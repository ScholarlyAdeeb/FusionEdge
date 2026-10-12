#!/usr/bin/env python
"""Run inference on a directory of real IR/VI pairs and build comparison grids.

This is the *real-image* path: it reads actual image files you supply, runs the
weights, and writes side-by-side comparison figures plus a metrics CSV.

What it deliberately does NOT do:

  * It does not train. Nothing here touches an optimizer.
  * It does not invent a dataset. If --ir-dir/--vi-dir are not given it refuses
    to run, rather than falling back to synthetic images.
  * It does not claim quality. The output is labelled with whether the weights
    are trained, and the metrics CSV is marked as an inference-set summary, not a
    benchmark result. Benchmark numbers require a registered held-out split (see
    scripts/run_experiment.py); these grids are for visual inspection only.

Usage
-----
    # trained weights on a folder of real pairs
    python scripts/infer_pairs.py \
        --ir-dir  path/to/ir --vi-dir path/to/vi \
        --weights experiments/fusionedge_main/checkpoints/best.pt \
        --out results/qualitative

    # plumbing check on real images with an untrained model (clearly marked)
    python scripts/infer_pairs.py --ir-dir ... --vi-dir ... \
        --weights experiments/smoke/checkpoints/best.pt --allow-untrained
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from fusionedge.data.fusion_dataset import _to_ir_tensor, read_image  # noqa: E402
from fusionedge.engine.infer import fuse_pair, is_trained, load_model  # noqa: E402
from fusionedge.metrics import compute_all  # noqa: E402
from fusionedge.models.classical import CLASSICAL_METHODS  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(d: Path) -> list[str]:
    return sorted(p.name for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def pair_files(ir_dir: Path, vi_dir: Path) -> list[tuple[str, Path, Path]]:
    """Pair on filename stem, matching the acquisition path's convention."""
    ir_map = {Path(n).stem: n for n in list_images(ir_dir)}
    vi_map = {Path(n).stem: n for n in list_images(vi_dir)}
    shared = sorted(ir_map.keys() & vi_map.keys())
    return [(s, ir_dir / ir_map[s], vi_dir / vi_map[s]) for s in shared]


def label_strip(width: int, text: str, height: int = 22) -> np.ndarray:
    """Caption bar. An unlabelled multi-panel grid is unreadable and invites
    misreading which column is which -- especially for a figure whose whole
    purpose is to be compared against another model."""
    strip = np.full((height, width, 3), 24, np.uint8)
    cv2.putText(strip, text[:38], (6, height - 7), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (235, 235, 235), 1, cv2.LINE_AA)
    return strip


def build_grid(panels: list[tuple[str, np.ndarray]]) -> np.ndarray:
    """Stack (title, image) panels horizontally with captions, equal heights."""
    target_h = min(p.shape[0] for _, p in panels)
    resized = []
    for title, img in panels:
        if img.shape[0] != target_h:
            scale = target_h / img.shape[0]
            img = cv2.resize(img, (max(1, int(img.shape[1] * scale)), target_h))
        resized.append((title, img))
    w = sum(p.shape[1] for _, p in resized)
    out = np.full((target_h + 22, w, 3), 24, np.uint8)
    x = 0
    for title, img in resized:
        out[22 : 22 + target_h, x : x + img.shape[1]] = img
        out[0:22, x : x + img.shape[1]] = label_strip(img.shape[1], title)
        x += img.shape[1]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ir-dir", required=True)
    ap.add_argument("--vi-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", default="fusionedge")
    ap.add_argument("--model-cfg", default="{}")
    ap.add_argument("--out", default="results/qualitative")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--grids", type=int, default=6, help="how many comparison grids")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp16", action="store_true", help="use autocast fp16")
    ap.add_argument("--no-classical", action="store_true")
    ap.add_argument("--allow-untrained", action="store_true")
    args = ap.parse_args()

    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            args.device = "cpu"

    ir_dir, vi_dir = Path(args.ir_dir), Path(args.vi_dir)
    for d, flag in ((ir_dir, "--ir-dir"), (vi_dir, "--vi-dir")):
        if not d.is_dir():
            print(f"{flag} is not a directory: {d}\n"
                  f"This script needs real paired images on disk. It will not "
                  f"generate substitute data.", file=sys.stderr)
            return 2

    pairs = pair_files(ir_dir, vi_dir)
    if not pairs:
        print(f"no paired files between {ir_dir} and {vi_dir} "
              f"(pairing is on filename stem)", file=sys.stderr)
        return 2
    print(f"{len(pairs)} paired scene(s) found")

    trained, reason = is_trained(args.weights)
    print(f"weights: {reason}")
    if not trained and not args.allow_untrained:
        print("\nREFUSING: these weights look untrained. An untrained FusionEdge "
              "returns the visible image unchanged, which looks like a plausible "
              "fusion but carries no infrared information.\nPass --allow-untrained "
              "only for a plumbing test.", file=sys.stderr)
        return 2

    model = load_model(args.weights, args.model, json.loads(args.model_cfg), args.device)

    out_dir = Path(args.out)
    grids_dir = out_dir / "grids"
    grids_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    subset = pairs[: min(args.limit, len(pairs))]
    n_grids = max(1, min(args.grids, len(subset)))

    for stem, ir_path, vi_path in subset:
        ir_img = read_image(ir_path)
        vi_img = read_image(vi_path)

        fused = fuse_pair(model, ir_path, vi_path, device=args.device, amp=args.fp16)

        ir_t = _to_ir_tensor(ir_img).numpy()[0]
        ir_rgb = (np.clip(ir_t, 0, 1) * 255).astype(np.uint8)
        ir_rgb = cv2.cvtColor(ir_rgb, cv2.COLOR_GRAY2RGB)
        ir_u8 = cv2.cvtColor(ir_rgb, cv2.COLOR_RGB2GRAY)

        panels: list[tuple[str, np.ndarray]] = [
            ("infrared (input)", ir_rgb),
            ("visible (input)", vi_img),
            (f"{args.model} (ours)", fused),
        ]
        if not args.no_classical:
            for cname in ("average", "max_abs", "laplacian_pyramid"):
                panels.append((f"classical: {cname}", CLASSICAL_METHODS[cname](ir_img, vi_img)))

        idx = len(rows)
        if idx < n_grids:
            cv2.imwrite(
                str(grids_dir / f"grid_{idx:02d}_{stem}.png"),
                cv2.cvtColor(build_grid(panels), cv2.COLOR_RGB2BGR),
            )

        res = compute_all(fused, ir_u8, vi_img).as_dict()
        rows.append({"pair_id": stem, "weights_trained": int(trained), **res})
        print(f"  {stem}: EN={res['EN']:.3f} AG={res['AG']:.4f} "
              f"MS-SSIM_ir={res['MS_SSIM_ir']:.4f}")

    csv_path = out_dir / "inference_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    meta = {
        "purpose": "qualitative inspection of real pairs; NOT a benchmark result",
        "weights": str(args.weights),
        "weights_trained": trained,
        "weights_note": reason,
        "model": args.model,
        "model_cfg": json.loads(args.model_cfg),
        "ir_dir": str(ir_dir),
        "vi_dir": str(vi_dir),
        "num_pairs_evaluated": len(subset),
        "num_pairs_available": len(pairs),
        "device": args.device,
        "fp16": args.fp16,
        "dataset_id": "UNREGISTERED (ad-hoc directory, not a benchmark split)",
        "metrics_csv": str(csv_path),
    }
    (out_dir / "inference_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"\nwrote {n_grids} comparison grid(s) to {grids_dir}")
    print(f"wrote {csv_path}")
    print("\nNOTE: these metrics describe the images you pointed at, not a "
          "registered held-out split.\nFor a benchmark result use "
          "scripts/run_experiment.py on a validated dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())