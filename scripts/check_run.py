#!/usr/bin/env python
"""Acceptance gate for a finished training run (convergence, reload, output quality).

    python scripts/check_run.py experiments/baseline_simplecnn_v4 [--n 16]

Writes results/checks/<run>.json and results/checks/<run>_samples.png and exits
non-zero if any hard check fails. Everything is computed on the run's
model-selection HOLDOUT (ids from <run>/holdout_manifest.json), never on the
MSRS test split.

Hard checks
  convergence   finite losses; completed the configured number of epochs; the
                training loss on fixed holdout images is at least 1% below a
                fresh init of the same model (fusionedge/engine/learning_check.py;
                replaced the init-dependent "tail5 >= 10% below epoch 0" rule on
                2026-10-15, see docs/PROTOCOL.md -- that ratio is still reported)
  reload        last.pt loads STRICTLY into a fresh model; two forward passes are
                bit-identical; checkpoint epoch == configured epochs
  quality       outputs finite and in [0,1]; <5% of pixels within 1e-3 of 0 or 1;
                fused luma positively correlated with BOTH source lumas; fused
                luma contrast (std) >= 50% of the weaker source's; output is not
                a copy of the visible image (mean |fused-vi| luma > 0.005)
Reported (not gating): holdout metrics for the run and for the identity-visible
reference on the same images, so "better than doing nothing" is visible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from fusionedge.data import FusionPairDataset, load_registry  # noqa: E402
from fusionedge.engine.learning_check import learning_check  # noqa: E402
from fusionedge.engine.evaluate import fuse_batch, tensor_to_uint8, tensor_to_uint8_gray  # noqa: E402
from fusionedge.metrics import compute_all  # noqa: E402
from fusionedge.utils.checkpoint import load_model_weights  # noqa: E402
from fusionedge.utils.registry import build_model  # noqa: E402


def luma(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x[:, 0]
    w = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype).view(1, 3, 1, 1)
    return (x * w).sum(1)


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, default=16, help="holdout images to check")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    run = Path(args.run_dir)
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    out_dir = REPO_ROOT / "results" / "checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<34} {detail}")

    # ---------------- convergence ------------------------------------------
    rows = list(csv.DictReader((run / "metrics.csv").open(encoding="utf-8")))
    losses = [float(r["loss"]) for r in rows if r.get("phase") == "train"]
    epochs_cfg = int(cfg["train"]["epochs"])
    check("all train losses finite", all(math.isfinite(v) for v in losses),
          f"{len(losses)} epochs logged")
    check("completed configured epochs", len(losses) >= epochs_cfg,
          f"{len(losses)}/{epochs_cfg}")
    if len(losses) >= 6:
        tail = float(np.mean(losses[-5:]))
        drop = 1.0 - tail / losses[0]
        checks.append({"check": "train loss tail5 vs ep0", "pass": True,
                       "detail": f"{losses[0]:.4f} -> {tail:.4f} ({drop:+.1%}) (info)"})
        tail_cv = float(np.std(losses[-5:]) / max(tail, 1e-12))
        checks.append({"check": "tail5 loss coefficient of variation", "pass": True,
                       "detail": f"{tail_cv:.4f} (info)"})
    val_cols = sorted(k for k in rows[0] if k.startswith("val_")) if rows else []
    val_rows = [r for r in rows if r.get(val_cols[0])] if val_cols else []
    curve = {k: [round(float(r[k]), 5) for r in val_rows] for k in val_cols}

    # ---------------- reload -------------------------------------------------
    ckpt = run / "checkpoints" / "last.pt"
    model = build_model(cfg["model"]["name"], cfg["model"].get("cfg", {}))
    try:
        load_model_weights(model, ckpt, device=args.device)
        check("last.pt strict reload", True, str(ckpt))
    except Exception as exc:  # noqa: BLE001
        check("last.pt strict reload", False, f"{type(exc).__name__}: {exc}")
        return finish(run, out_dir, checks, {}, curve)
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    check("checkpoint epoch == configured", int(payload.get("epoch", -1)) == epochs_cfg,
          f"epoch={payload.get('epoch')} global_step={payload.get('global_step')}")
    check("checkpoint has scheduler state", "scheduler_state" in payload, "")

    # ---------------- holdout images ----------------------------------------
    reg = load_registry(cfg["dataset"]["id"])
    hm = json.loads((run / "holdout_manifest.json").read_text(encoding="utf-8"))
    ids = hm["holdout_ids"][: args.n]
    ds = FusionPairDataset(reg["root"], hm["source_split"], pairs=ids)
    dev = torch.device(args.device)
    ident = build_model("identity_vi", {})

    stats = {"sat": [], "c_ir": [], "c_vi": [], "std_ratio": [], "dvi": [],
             "c_vi_ir": [], "chroma": []}
    m_run, m_id = [], []
    det_ok = True
    samples = []
    for i in range(len(ds)):
        b = ds[i]
        ir, vi = b["ir"][None], b["vi"][None]
        f1 = fuse_batch(model, ir, vi, device=dev, amp=False).cpu()
        if i < 2:
            f2 = fuse_batch(model, ir, vi, device=dev, amp=False).cpu()
            det_ok &= torch.equal(f1, f2)
        fi = fuse_batch(ident, ir, vi, device=torch.device("cpu"), amp=False)
        if not torch.isfinite(f1).all() or f1.min() < 0 or f1.max() > 1:
            stats["sat"].append(1.0)
            continue
        stats["sat"].append(float(((f1 < 1e-3) | (f1 > 1 - 1e-3)).float().mean()))
        lf, li, lv = luma(f1), luma(ir), luma(vi)
        stats["c_ir"].append(corr(lf, li))
        stats["c_vi"].append(corr(lf, lv))
        stats["std_ratio"].append(float(lf.std() / max(min(li.std(), lv.std()), 1e-6)))
        stats["dvi"].append(float((lf - lv).abs().mean()))
        stats["c_vi_ir"].append(corr(lv, li))
        from fusionedge.models.colour import rgb_to_ycbcr
        stats["chroma"].append(float((rgb_to_ycbcr(f1)[:, 1:] - rgb_to_ycbcr(vi)[:, 1:]).abs().mean()))
        r8, v8 = tensor_to_uint8_gray(ir), tensor_to_uint8(vi)
        m_run.append(compute_all(tensor_to_uint8(f1), r8, v8).as_dict())
        m_id.append(compute_all(tensor_to_uint8(fi), r8, v8).as_dict())
        if len(samples) < 4:
            samples.append((r8, v8, tensor_to_uint8(f1)))

    check("forward pass deterministic", det_ok, "2 images, fp32, run twice")
    lc = learning_check(model, cfg, [(ds[i]["ir"][None], ds[i]["vi"][None]) for i in range(len(ds))], dev)
    check("loss below fresh init >=1% (fixed)", lc["pass"],
          f"{lc['init']:.4f} -> {lc['trained']:.4f} ({lc['rel_drop']:+.2%}, noise {lc['noise']:.1e}, {lc['n']} holdout imgs)")
    mean = lambda k: float(np.mean(stats[k])) if stats[k] else float("nan")  # noqa: E731
    check("saturation < 5%", mean("sat") < 0.05, f"{mean('sat'):.4f}")
    check("corr(fused, ir) > 0 (mean)", mean("c_ir") > 0, f"{mean('c_ir'):.3f}")
    check("corr(fused, vi) > 0 (mean)", mean("c_vi") > 0, f"{mean('c_vi'):.3f}")
    check("contrast >= 50% of weaker source", mean("std_ratio") >= 0.5, f"{mean('std_ratio'):.3f}")
    check("not a copy of visible", mean("dvi") > 0.005, f"mean|dY|={mean('dvi'):.4f}")
    # Added 2026-10-14 after v4 passed every check above yet visibly failed:
    # thermal content went into false colour instead of luminance.
    check("carries IR beyond visible",
          mean("c_ir") >= mean("c_vi_ir") + 0.05,
          f"corr(f,ir)={mean('c_ir'):.3f} vs corr(vi,ir)={mean('c_vi_ir'):.3f} (need +0.05)")
    check("chroma faithful to visible", mean("chroma") < 0.03,
          f"mean|dCbCr|={mean('chroma'):.4f}")

    agg = lambda rows_: {k: float(np.mean([r[k] for r in rows_])) for k in rows_[0]} if rows_ else {}  # noqa: E731
    holdout = {"n": len(m_run), "run": agg(m_run), "identity_vi": agg(m_id)}
    print("\n  holdout metrics (run vs identity_vi):")
    for k in holdout["run"]:
        print(f"    {k:<12} {holdout['run'][k]:8.4f}   {holdout['identity_vi'][k]:8.4f}")

    if samples:
        import cv2

        tiles = []
        for r8, v8, f8 in samples:
            r3 = np.repeat(r8[..., None], 3, axis=2)
            tiles.append(np.concatenate([r3, v8, f8], axis=1))
        grid = np.concatenate(tiles, axis=0)
        grid = cv2.resize(grid, (grid.shape[1] // 2, grid.shape[0] // 2), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"{run.name}_samples.png"), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    return finish(run, out_dir, checks, holdout, curve)


def finish(run: Path, out_dir: Path, checks: list, holdout: dict, curve: dict) -> int:
    ok = all(c["pass"] for c in checks)
    (out_dir / f"{run.name}.json").write_text(json.dumps(
        {"run": run.name, "passed": ok, "checks": checks, "holdout": holdout,
         "val_curve": curve}, indent=2), encoding="utf-8")
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}  -> results/checks/{run.name}.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
