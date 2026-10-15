#!/usr/bin/env python
"""Generate result tables from experiment artefacts. Never by hand.

This script is the only sanctioned way to produce a numbers table. It reads
`experiments/<run>/metrics.json` and `benchmark.json` and writes markdown. Two
properties make the tables trustworthy:

  1. **Provenance gating.** A table may only contain rows from a single
     `dataset_id` + `manifest_sha256`, and a single YOLO backend. Mixing MSRS and
     MFNet rows, or ultralytics and ONNX rows, raises rather than silently
     producing a plausible-looking table. Each emitted table is stamped with the
     dataset it belongs to.

  2. **Smoke-run gating.** A run marked `"smoke": true` is excluded unless
     `--include-smoke` is passed, and then loudly labelled. Smoke numbers are
     pipeline-plumbing checks, not results.

     python scripts/make_figures.py
     python scripts/make_figures.py --include-smoke
     python scripts/make_figures.py --out results/tables.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Metrics shown in the quality table, in reporting order.
QUALITY_COLS = [
    ("EN", "EN ↑"),
    ("AG", "AG ↑"),
    ("SF", "SF ↑"),
    ("MI", "MI ↑"),
    ("MS_SSIM_ir", "MS-SSIM(ir) ↑"),
    ("MS_SSIM_vi", "MS-SSIM(vi) ↑"),
]

RUN_LABELS = {
    "baseline_simplecnn": "SimpleCNN baseline",
    "fusionedge_main": "FusionEdge (full)",
    "ablation_no_edge": "ablation: no edge guidance",
    "ablation_avg_fusion": "ablation: average fusion (no adaptive gate)",
    "ablation_no_cross": "ablation: no cross-modal exchange",
    "ablation_no_se": "ablation: no SE-lite",
    "ablation_dense_conv": "ablation: dense 3x3 (not separable)",
    "ablation_rgb_head": "ablation: direct RGB head",
    "ablation_unshared_stem": "ablation: unshared stem",
}


def load_runs(exp_dir: Path, include_smoke: bool, protocol_only: bool = False) -> list[dict]:
    runs = []
    for d in sorted(exp_dir.iterdir()):
        mj = d / "metrics.json"
        if not d.is_dir() or not mj.exists():
            continue
        try:
            payload = json.loads(mj.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("smoke") and not include_smoke:
            continue
        # Only runs that follow docs/PROTOCOL.md (holdout selection, test split
        # evaluated once) and are not marked superseded may enter a table.
        if (d / "SUPERSEDED").exists():
            print(f"[make_figures] skip {d.name}: marked SUPERSEDED", file=sys.stderr)
            continue
        if protocol_only and (payload.get("protocol") or {}).get("selection_split") != "holdout of train":
            print(f"[make_figures] skip {d.name}: pre-protocol run", file=sys.stderr)
            continue
        bj = d / "benchmark.json"
        payload["_bench"] = (
            json.loads(bj.read_text(encoding="utf-8")) if bj.exists() else None
        )
        payload["_dir"] = d
        payload["_name"] = d.name
        runs.append(payload)
    return runs


def check_single_dataset(runs: list[dict]) -> tuple[str, str, str | None]:
    """Return (dataset_id, manifest, backend). Raises if the runs are not
    mutually comparable."""
    datasets = {r["dataset"].get("dataset_id") for r in runs}
    manifests = {r["dataset"].get("manifest_sha256") for r in runs}
    if len(datasets) > 1:
        raise SystemExit(
            f"REFUSING to build a table spanning datasets {sorted(datasets)}.\n"
            f"Results from different datasets are not comparable; generate one "
            f"table per dataset."
        )
    if len(manifests) > 1:
        raise SystemExit(
            f"REFUSING: runs span {len(manifests)} different dataset manifests. "
            f"They were computed on different bytes of data."
        )

    backends = set()
    for r in runs:
        y = (r.get("downstream") or {}).get("yolov5x6") or {}
        if y.get("status") == "unavailable":
            backends.add("unavailable")
        elif y.get("backend"):
            backends.add(y["backend"])
    if len(backends) > 1:
        raise SystemExit(
            f"REFUSING to mix inference backends {sorted(backends)} in one table."
        )

    dataset_id = next(iter(datasets), "unknown")
    manifest = next(iter(manifests), "unknown")
    backend = next(iter(backends), None)
    return dataset_id, manifest, backend


def fmt(x, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:  # NaN
        return "n/a"
    return f"{v:.{digits}f}"


def quality_table(runs: list[dict], dataset_id: str, manifest: str, backend: str | None) -> str:
    lines = [
        f"### Fusion quality — dataset `{dataset_id}`",
        "",
        f"*Held-out split: `{'test' if dataset_id == 'MSRS' else 'val'}`. "
        f"Manifest `{str(manifest)[:16]}`. "
        f"{'Generated from `experiments/*/metrics.json`; do not edit by hand.'}*",
        "",
        "| model | " + " | ".join(lbl for _, lbl in QUALITY_COLS) + " | images |",
        "|---|" + "---|" * (len(QUALITY_COLS) + 1),
    ]
    for r in runs:
        m = r["metrics"]
        cells = [fmt(m.get(k)) for k, _ in QUALITY_COLS]
        label = RUN_LABELS.get(r["_name"], r["_name"])
        lines.append(
            f"| {label} | " + " | ".join(cells) + f" | {r['evaluation']['num_images']} |"
        )
    if backend:
        lines += ["", f"*Downstream-task metric backend: `{backend}`.*"]
    return "\n".join(lines)


def ablation_table(runs: list[dict], full_name: str) -> str | None:
    """Deltas of each ablation against the full model.

    Only valid when every row shares a dataset (checked by the caller). Deltas
    are stated per metric with an explicit sign, and `--` marks a run whose
    numbers are missing rather than treating it as zero.
    """
    full = next((r for r in runs if r["_name"] == full_name), None)
    if full is None:
        return None
    ablations = [
        r for r in runs
        if r["_name"] != full_name and r["_name"].startswith("ablation_")
    ]
    if not ablations:
        return None

    lines = [
        "### Ablations — change vs. the full model",
        "",
        "*Positive = the ablation is better than the full model, i.e. the removed "
        "component was **hurting**. Negative = the component was helping. "
        "A delta near zero means the component earned nothing.*",
        "",
        "| ablation | " + " | ".join(f"Δ {lbl}" for _, lbl in QUALITY_COLS) + " |",
        "|---|" + "---|" * len(QUALITY_COLS),
    ]
    for r in sorted(ablations, key=lambda x: x["_name"]):
        cells = []
        for k, _ in QUALITY_COLS:
            a, b = r["metrics"].get(k), full["metrics"].get(k)
            if a is None or b is None:
                cells.append("--")
            else:
                try:
                    d = float(a) - float(b)
                    cells.append(f"{d:+.4f}")
                except (TypeError, ValueError):
                    cells.append("--")
        lines.append(
            f"| {RUN_LABELS.get(r['_name'], r['_name'])} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines)


def benchmark_table(runs: list[dict]) -> str:
    rows = [r for r in runs if r.get("_bench")]
    if not rows:
        return ""
    size_key = None
    for r in rows:
        for lat in r["_bench"].get("latency", []):
            if lat["size"] == [480, 640]:
                size_key = "480x640"
    lines = [
        "### Efficiency",
        "",
        "| model | params | size (MB) | median ms | p95 ms | fps | peak VRAM (MiB) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        b = r["_bench"]
        lat = None
        for cand in b.get("latency", []):
            if size_key and cand["size"] == [480, 640]:
                lat = cand
        lat = lat or (b.get("latency") or [{}])[0]
        lines.append(
            f"| {RUN_LABELS.get(r['_name'], r['_name'])} | {b.get('params', 0):,} | "
            f"{b.get('checkpoint_mb', 0):.2f} | "
            f"{fmt(lat.get('latency_ms_median'), 2)} | {fmt(lat.get('latency_ms_p95'), 2)} | "
            f"{fmt(lat.get('fps'), 1)} | {fmt(lat.get('peak_vram_mb'), 0)} |"
        )
    gpu = next(
        (r["_bench"].get("gpu") for r in rows if r["_bench"].get("gpu")), None
    )
    if gpu:
        lines += ["", f"*Measured on {gpu}, batch 1, fp32, CUDA-event timed.*"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", default="experiments")
    ap.add_argument("--out", default=None, help="write markdown here (default: stdout)")
    ap.add_argument("--full", default="fusionedge_main", help="the run ablations compare to")
    ap.add_argument("--include-smoke", action="store_true")
    ap.add_argument("--protocol-only", action="store_true",
                    help="only runs following docs/PROTOCOL.md (holdout selection)")
    args = ap.parse_args()

    exp_dir = REPO_ROOT / args.exp
    if not exp_dir.is_dir():
        print(f"no experiments directory at {exp_dir}", file=sys.stderr)
        return 2

    runs = load_runs(exp_dir, args.include_smoke, protocol_only=args.protocol_only)
    if not runs:
        print(
            f"no completed runs with metrics.json in {exp_dir}"
            + ("" if args.include_smoke else " (use --include-smoke to include smoke runs)"),
            file=sys.stderr,
        )
        return 2

    dataset_id, manifest, backend = check_single_dataset(runs)

    header = [
        f"<!-- GENERATED by scripts/make_figures.py -- do not edit by hand -->",
        f"<!-- dataset: {dataset_id} | manifest: {manifest} -->",
        "",
    ]
    body = [quality_table(runs, dataset_id, manifest, backend), ""]
    abl = ablation_table(runs, args.full)
    if abl:
        body += [abl, ""]
    bench = benchmark_table(runs)
    if bench:
        body += [bench, ""]

    if args.include_smoke:
        body = [
            "> **WARNING: smoke runs included.** These numbers come from short "
            "pipeline-verification runs on a handful of images. They are NOT "
            "results.\n"
        ] + body

    text = "\n".join(header + body)
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())