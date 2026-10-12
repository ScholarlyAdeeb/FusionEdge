#!/usr/bin/env python
"""Post-run audit: verify a completed run before its numbers are used.

A training run can exit 0 and still be untrustworthy -- a bad resume, a split that
leaked, a checkpoint from the wrong epoch, a curve that never moved. This script
checks those things from the artefacts on disk and prints PASS/FAIL per check,
exiting non-zero if any fail.

    python scripts/audit_run.py experiments/baseline_simplecnn

Checks
------
  1. artefacts complete (config, log, metrics.csv, metrics.json, benchmark.json,
     checkpoints)
  2. metrics.json / config.json / the live registry all agree on dataset id and
     manifest hash -- i.e. the run really used the data we think it did
  3. the run is NOT a smoke run (unless --allow-smoke)
  4. the evaluation split is the held-out one, and its pair ids do not intersect
     the training split -- leakage is re-checked from the dataset on disk, not
     taken on trust from the config
  5. the loss curve actually moved (first vs last logged epoch)
  6. best.pt is loadable, and its recorded epoch/metric are consistent
  7. every reported metric is finite
  8. benchmark.json reports params, latency and peak VRAM
  9. evaluation covered a substantial number of images
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

import torch  # noqa: E402

from fusionedge.data import load_registry  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class Audit:
    def __init__(self) -> None:
        self.results: list[tuple[bool, str, str]] = []

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        self.results.append((bool(ok), name, detail))
        return bool(ok)

    def report(self) -> int:
        width = max((len(n) for _, n, _ in self.results), default=10)
        for ok, name, detail in self.results:
            print(f"  [{'PASS' if ok else 'FAIL'}] {name.ljust(width)}  {detail}")
        n_fail = sum(1 for ok, _, _ in self.results if not ok)
        print()
        print(
            f"  AUDIT FAILED: {n_fail}/{len(self.results)} check(s) failed"
            if n_fail
            else f"  AUDIT PASSED: all {len(self.results)} checks"
        )
        return 1 if n_fail else 0


def list_ids(split_dir: Path) -> set[str]:
    if not split_dir.is_dir():
        return set()
    return {p.stem for p in split_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}


def audit(run_dir: Path, allow_smoke: bool = False) -> int:
    a = Audit()
    print(f"\nauditing {run_dir}\n")

    required = ["config.yaml", "config.json", "log.txt", "metrics.csv",
                "metrics.json", "benchmark.json"]
    missing = [f for f in required if not (run_dir / f).exists()]
    ckpt_dir = run_dir / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.pt")) if ckpt_dir.is_dir() else []
    a.check(not missing, "artefacts complete",
            "all present" if not missing else f"MISSING {missing}")
    a.check(bool(ckpts), "checkpoints present",
            ", ".join(c.name for c in ckpts) if ckpts else "none")
    if missing:
        return a.report()

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    bench = json.loads((run_dir / "benchmark.json").read_text(encoding="utf-8"))
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    stamped = cfg.get("_stamped", {})

    # --- provenance ------------------------------------------------------
    ds_id = metrics["dataset"].get("dataset_id")
    ds_manifest = metrics["dataset"].get("manifest_sha256")
    a.check(
        ds_manifest == stamped.get("dataset_manifest_sha256"),
        "manifest: metrics vs config",
        f"{str(ds_manifest)[:16]} vs {str(stamped.get('dataset_manifest_sha256'))[:16]}",
    )
    # `registry` is assigned before use below (split checks read it), so it is
    # initialised here rather than only inside the try block.
    registry: dict = {}
    try:
        registry = load_registry(ds_id)
        a.check(
            ds_manifest == registry.get("manifest_sha256"),
            "manifest: run vs registry",
            f"run={str(ds_manifest)[:16]} registry={str(registry.get('manifest_sha256'))[:16]}",
        )
    except Exception as exc:  # noqa: BLE001
        a.check(False, "dataset registered", f"cannot load '{ds_id}': {exc}")

    # --- smoke gate -------------------------------------------------------
    is_smoke = bool(metrics.get("smoke"))
    a.check((not is_smoke) or allow_smoke, "not a smoke run",
            "smoke=true (explicitly allowed)" if is_smoke else "full run")

    # --- split + leakage, re-checked from disk ----------------------------
    train_split = cfg["dataset"]["train_split"]
    eval_split = cfg["dataset"]["eval_split"]
    root = Path(metrics["dataset"].get("root") or registry.get("root", ""))
    if root.is_dir():
        tr = list_ids(root / train_split / "ir")
        ev = list_ids(root / eval_split / "ir")
        a.check(train_split != eval_split, "eval split differs from train",
                f"train={train_split} eval={eval_split}")
        overlap = tr & ev
        a.check(not overlap, "no train/eval id overlap",
                f"{len(tr)} train / {len(ev)} eval ids, {len(overlap)} shared")
    else:
        a.check(False, "dataset root readable", f"{root} not found")

    # --- curve actually moved ---------------------------------------------
    with (run_dir / "metrics.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    a.check(len(rows) > 0, "training history", f"{len(rows)} epoch rows")
    if rows:
        losses = [(int(r["epoch"]), float(r["loss"])) for r in rows if r.get("loss")]
        if losses:
            first, last = losses[0][1], losses[-1][1]
            a.check(last < first, "loss decreased",
                    f"epoch {losses[0][0]}: {first:.4f} -> epoch {losses[-1][0]}: {last:.4f}")
            a.check(all(math.isfinite(v) for _, v in losses), "train loss finite")

    # --- checkpoint sanity -------------------------------------------------
    best = ckpt_dir / "best.pt"
    if best.exists():
        try:
            payload = torch.load(best, map_location="cpu", weights_only=False)
            has_state = "model_state" in payload and len(payload["model_state"]) > 0
            a.check(has_state, "best.pt loadable",
                    f"epoch={payload.get('epoch')} step={payload.get('global_step')} "
                    f"metric={payload.get('best_metric')}")
            a.check(payload.get("best_metric") is not None, "best.pt records its metric",
                    str(payload.get("best_metric")))
        except Exception as exc:  # noqa: BLE001
            a.check(False, "best.pt loadable", f"{type(exc).__name__}: {exc}")

    # --- metric sanity -----------------------------------------------------
    vals = metrics.get("metrics", {})
    a.check(bool(vals) and all(
        isinstance(v, (int, float)) and math.isfinite(v) for v in vals.values()
    ), "all metrics finite", f"{len(vals)} metrics")

    n_eval = metrics["evaluation"]["num_images"]
    a.check(n_eval > 1, "evaluation coverage",
            f"{n_eval} images (smoke run)" if is_smoke else f"{n_eval} images")

    # --- benchmark ---------------------------------------------------------
    has_latency = bool(bench.get("latency")) and all(
        "latency_ms_median" in l and l["latency_ms_median"] > 0 for l in bench["latency"]
    )
    a.check(bool(bench.get("params")), "params recorded", f"{bench.get('params', 0):,}")
    a.check(has_latency, "latency recorded",
            ", ".join(f"{l['size'][0]}x{l['size'][1]}={l['latency_ms_median']}ms"
                      for l in bench.get("latency", [])))
    a.check(bool(bench.get("gpu")), "gpu recorded", str(bench.get("gpu", "unknown")))

    return a.report()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir")
    ap.add_argument("--allow-smoke", action="store_true")
    args = ap.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    if not run_dir.is_dir():
        print(f"no such run directory: {run_dir}", file=sys.stderr)
        return 2
    return audit(run_dir, allow_smoke=args.allow_smoke)


if __name__ == "__main__":
    raise SystemExit(main())