#!/usr/bin/env python
"""Holdout-only analysis of finished runs: metrics, failure modes, and (for the
FusionEdge inject head) where the learned gate opens.

    python scripts/analyse_fusion.py experiments/fusionedge_main experiments/baseline_simplecnn_v6

Everything uses each run's `last.pt` (the protocol's reported checkpoint), fp32,
full resolution, on ALL ids of the run's model-selection holdout
(<run>/holdout_manifest.json). The MSRS test split is never read. Every run must
declare the same holdout sha, or the script refuses to compare them.

Writes results/analysis/<stamp>/ (never overwrites): summary.json, summary.md,
panels_<k>.png (IR | VI | each run | gate | dY).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402

from fusionedge.data import FusionPairDataset, load_registry  # noqa: E402
from fusionedge.engine.evaluate import tensor_to_uint8, tensor_to_uint8_gray  # noqa: E402
from fusionedge.metrics import compute_all  # noqa: E402
from fusionedge.models.colour import rgb_to_ycbcr  # noqa: E402
from fusionedge.utils.checkpoint import load_model_weights  # noqa: E402
from fusionedge.utils.registry import build_model  # noqa: E402

IR_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
GAIN_BINS = [0.0, 1e-6, 0.02, 0.05, 0.1, 0.2, 0.4, 1.0001]


def sobel(y: torch.Tensor) -> torch.Tensor:
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=y.dtype, device=y.device).view(1, 1, 3, 3)
    gx, gy = F.conv2d(y, kx, padding=1), F.conv2d(y, kx.transpose(2, 3), padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-12)


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


class Acc:
    """Pixel-weighted accumulators for binned statistics."""

    def __init__(self, edges: list[float]):
        self.edges = edges
        n = len(edges) - 1
        self.cnt = np.zeros(n)
        self.sums: dict[str, np.ndarray] = {}

    def add(self, key_vals: torch.Tensor, **vals: torch.Tensor) -> None:
        k = key_vals.flatten().cpu().numpy()
        idx = np.clip(np.searchsorted(self.edges, k, side="right") - 1, 0, len(self.edges) - 2)
        self.cnt += np.bincount(idx, minlength=len(self.cnt))
        for name, v in vals.items():
            s = self.sums.setdefault(name, np.zeros(len(self.cnt)))
            s += np.bincount(idx, weights=v.flatten().double().cpu().numpy(), minlength=len(self.cnt))

    def table(self) -> list[dict]:
        out = []
        for i in range(len(self.cnt)):
            row = {"bin": f"[{self.edges[i]:.2g},{self.edges[i + 1]:.2g})", "pixel_frac": 0.0}
            if self.cnt[i] > 0:
                row["pixel_frac"] = float(self.cnt[i] / self.cnt.sum())
                for k, s in self.sums.items():
                    row[k] = float(s[i] / self.cnt[i])
            out.append(row)
        return out


def load_run(run: Path, dev: torch.device):
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    m = build_model(cfg["model"]["name"], cfg["model"].get("cfg", {})).to(dev).eval()
    load_model_weights(m, run / "checkpoints" / "last.pt", device=str(dev))
    return cfg, m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--panels", type=int, default=8)
    ap.add_argument("--out", default=None, help="output dir (default results/analysis/<stamp>)")
    ap.add_argument("--limit", type=int, default=0, help="debug: first N holdout ids only")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    runs = [Path(r) for r in args.runs]

    manifests = [json.loads((r / "holdout_manifest.json").read_text(encoding="utf-8")) for r in runs]
    shas = {m["holdout_sha256"] for m in manifests}
    if len(shas) != 1:
        print(f"REFUSING: runs declare different holdouts: {shas}")
        return 2
    hm = manifests[0]
    cfg0 = yaml.safe_load((runs[0] / "config.yaml").read_text(encoding="utf-8"))
    ds = FusionPairDataset(load_registry(cfg0["dataset"]["id"])["root"], hm["source_split"], pairs=hm["holdout_ids"][: args.limit or None])

    models = {}
    for r in runs:
        cfg, m = load_run(r, dev)
        gate = {}
        mc = cfg["model"].get("cfg", {})
        inject = ((cfg["model"]["name"] == "fusionedge" and mc.get("residual_mode") == "inject")
                  or (cfg["model"]["name"] == "baseline" and mc.get("output_space") == "inject"))
        if inject:
            m.head.register_forward_hook(lambda mod, i, o, g=gate: g.__setitem__("g", torch.sigmoid(o)))
        models[r.name] = {"cfg": cfg, "m": m, "gate": gate, "inject": inject,
                          "metrics": [], "ir_bins": Acc(IR_BINS), "gain_bins": Acc(GAIN_BINS),
                          "px": {k: 0.0 for k in ("n", "bright01", "dark01", "blown_new", "dchroma",
                                                   "elig", "dy_sum", "gain_sum", "pos_sum", "pos_elig_sum")},
                          "img": {k: [] for k in ("c_ir", "c_vi", "grad_corr_vi", "edge_keep_nonelig",
                                                  "edge_keep_all", "gate_mean_elig", "gate_sd_elig",
                                                  "gate_corr_gain_elig", "inj_share")}}
    ident = {"metrics": []}

    panels, hot_rank = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(len(ds)):
            b = ds[i]
            ir, vi = b["ir"][None].to(dev), b["vi"][None].to(dev)
            y_vi = rgb_to_ycbcr(vi)[:, :1]
            gain = F.relu(ir - y_vi)
            elig = gain > 0.02
            r8, v8 = tensor_to_uint8_gray(ir.cpu()), tensor_to_uint8(vi.cpu())
            ident["metrics"].append(compute_all(v8, r8, v8).as_dict())
            blown_vi = (vi >= 254.5 / 255).any(1, keepdim=True)
            outs = {}
            for name, d in models.items():
                f = d["m"](ir, vi).float().clamp(0, 1)
                outs[name] = f
                y_f = rgb_to_ycbcr(f)[:, :1]
                dy = y_f - y_vi
                d["metrics"].append(compute_all(tensor_to_uint8(f.cpu()), r8, v8).as_dict())
                px = d["px"]
                n = dy.numel()
                px["n"] += n
                px["bright01"] += float((dy > 0.01).sum())
                px["dark01"] += float((dy < -0.01).sum())
                px["blown_new"] += float(((f >= 254.5 / 255).any(1, keepdim=True) & ~blown_vi).sum())
                px["dchroma"] += float((rgb_to_ycbcr(f)[:, 1:] - rgb_to_ycbcr(vi)[:, 1:]).abs().mean(1).sum())
                px["elig"] += float(elig.sum())
                px["dy_sum"] += float(dy.sum())
                px["gain_sum"] += float(gain.sum())
                pos = dy.clamp_min(0)
                px["pos_sum"] += float(pos.sum())
                px["pos_elig_sum"] += float(pos[elig].sum()) if elig.any() else 0.0
                im = d["img"]
                im["c_ir"].append(corr(y_f, ir))
                im["c_vi"].append(corr(y_f, y_vi))
                sf, sv = sobel(y_f), sobel(y_vi)
                im["grad_corr_vi"].append(corr(sf, sv))
                im["edge_keep_all"].append(float(sf.mean() / sv.mean().clamp_min(1e-8)))
                ne = ~elig
                if ne.any():
                    im["edge_keep_nonelig"].append(float(sf[ne].mean() / sv[ne].mean().clamp_min(1e-8)))
                if float(gain.sum()) > 0:
                    im["inj_share"].append(float(dy.clamp_min(0).sum() / gain.sum()))
                extra = {"dy": dy, "gain": gain}
                if d["inject"]:
                    g = d["gate"]["g"]
                    extra["gate"] = g
                    if elig.any():
                        im["gate_mean_elig"].append(float(g[elig].mean()))
                        im["gate_sd_elig"].append(float(g[elig].std()) if int(elig.sum()) > 1 else 0.0)
                        if int(elig.sum()) > 10:
                            im["gate_corr_gain_elig"].append(corr(g[elig], gain[elig]))
                d["ir_bins"].add(ir, **extra)
                d["gain_bins"].add(gain, **extra)
            hot_rank.append((float(elig.float().mean()), i))
            if i < 4:
                panels.append((i, r8, v8, {k: v.cpu() for k, v in outs.items()},
                               {k: (models[k]["gate"].get("g").cpu() if models[k]["inject"] else None) for k in outs}))

        # add the images with the most IR-brighter area (where fusion matters most)
        extra_ids = [i for _, i in sorted(hot_rank, reverse=True) if i >= 4][: max(0, args.panels - 4)]
        for i in extra_ids:
            b = ds[i]
            ir, vi = b["ir"][None].to(dev), b["vi"][None].to(dev)
            outs = {k: d["m"](ir, vi).float().clamp(0, 1).cpu() for k, d in models.items()}
            gates = {k: (d["gate"].get("g").cpu() if d["inject"] else None) for k, d in models.items()}
            panels.append((i, tensor_to_uint8_gray(ir.cpu()), tensor_to_uint8(vi.cpu()), outs, gates))

    agg = lambda rows: {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}  # noqa: E731
    summary = {"holdout_sha256": hm["holdout_sha256"], "n_images": len(ds), "checkpoint": "last.pt",
               "split": f"MSRS {hm['source_split']} holdout (model selection only; test split not read)",
               "identity_vi": agg(ident["metrics"]), "runs": {}, "seconds": round(time.time() - t0, 1)}
    for name, d in models.items():
        px = d["px"]
        mean = lambda k: float(np.mean(d["img"][k])) if d["img"][k] else float("nan")  # noqa: E731
        summary["runs"][name] = {
            "model": d["cfg"]["model"],
            "metrics": agg(d["metrics"]),
            "pixels": {
                "frac_brightened_gt_0.01": px["bright01"] / px["n"],
                "frac_darkened_gt_0.01": px["dark01"] / px["n"],
                "frac_new_blown_highlights": px["blown_new"] / px["n"],
                "mean_abs_dCbCr": px["dchroma"] / px["n"],
                "frac_ir_brighter_by_0.02": px["elig"] / px["n"],
                "mean_dY": px["dy_sum"] / px["n"],
                "share_of_brightening_inside_ir_brighter": px["pos_elig_sum"] / max(px["pos_sum"], 1e-12),
                "mean_available_gain": px["gain_sum"] / px["n"],
            },
            "per_image_mean": {k: mean(k) for k in d["img"]},
            "by_ir_intensity": d["ir_bins"].table(),
            "by_available_gain": d["gain_bins"].table(),
        }

    out = Path(args.out) if args.out else REPO_ROOT / "results" / "analysis" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # panels: IR | VI | each run | gate (inject) | dY of the inject run (red = brighter)
    names = list(models)
    for k, (i, r8, v8, outs, gates) in enumerate(panels):
        tiles = [cv2.cvtColor(r8, cv2.COLOR_GRAY2RGB), v8] + [tensor_to_uint8(outs[n]) for n in names]
        for n in names:
            if gates[n] is not None:
                g8 = (gates[n][0, 0].numpy() * 255).astype(np.uint8)
                tiles.append(cv2.cvtColor(cv2.applyColorMap(g8, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB))
                vi_t = torch.from_numpy(v8).permute(2, 0, 1)[None].float() / 255
                dy = (rgb_to_ycbcr(outs[n])[:, :1] - rgb_to_ycbcr(vi_t)[:, :1])[0, 0].numpy()
                d8 = np.clip(127.5 + dy * 127.5 / 0.3, 0, 255).astype(np.uint8)
                tiles.append(cv2.cvtColor(cv2.applyColorMap(d8, cv2.COLORMAP_COOLWARM if hasattr(cv2, "COLORMAP_COOLWARM") else cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB))
        h, w = tiles[0].shape[:2]
        tiles = [cv2.resize(t, (w // 2, h // 2), interpolation=cv2.INTER_AREA) for t in tiles]
        row = np.concatenate(tiles, axis=1)
        cv2.imwrite(str(out / f"panel_{k:02d}_{Path(hm['holdout_ids'][i]).stem}.png"), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    lines = [f"# Holdout analysis ({len(ds)} images, last.pt, holdout sha {hm['holdout_sha256'][:12]})", "",
             "Panels: IR | VI | " + " | ".join(names) + " | then gate, dY for each inject run in that order (JET: green = 0, red = brighter, ±0.3 full scale)", "",
             "| metric | identity_vi | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for mk in summary["identity_vi"]:
        lines.append(f"| {mk} | {summary['identity_vi'][mk]:.4f} | "
                     + " | ".join(f"{summary['runs'][n]['metrics'][mk]:.4f}" for n in names) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({n: {"metrics": s["metrics"], "pixels": s["pixels"], "per_image_mean": s["per_image_mean"]}
                      for n, s in summary["runs"].items()}, indent=1))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
