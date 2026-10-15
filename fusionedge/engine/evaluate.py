"""Evaluation engine.

Runs a model over a held-out split and writes `metrics.json` containing:
  * per-metric means over the split,
  * the dataset id and manifest hash it was computed on,
  * the model config and parameter count,
  * the backend used for any downstream-task metric.

The provenance fields are not decoration: `scripts/make_figures.py` refuses to
build a comparison table spanning two datasets or two inference backends, so a
number can never be presented as if it came from data it did not come from.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from ..data.transforms import pad_to_multiple
from ..metrics import compute_all
from ..utils.checkpoint import load_model_weights


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) float [0,1] on any device -> HWC uint8 RGB numpy."""
    if x.dim() != 4:
        raise ValueError(f"expected (B,C,H,W), got {tuple(x.shape)}")
    arr = x.detach().float().clamp(0, 1).cpu().numpy()[0]
    arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)


def tensor_to_uint8_gray(x: torch.Tensor) -> np.ndarray:
    """(1,1,H,W) -> H,W uint8."""
    arr = x.detach().float().clamp(0, 1).cpu().numpy()[0, 0]
    return np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)


@torch.inference_mode()
def fuse_batch(
    model: nn.Module,
    ir: torch.Tensor,
    vi: torch.Tensor,
    *,
    device: torch.device,
    amp: bool = True,
    amp_dtype: str = "bfloat16",
    pad_multiple: int = 8,
) -> torch.Tensor:
    """Pad -> forward -> unpad, so odd input sizes are handled by the model
    rather than crashing it. Returns the fused tensor at the *original* size.

    `amp_dtype` defaults to bfloat16 for the same reason training does: fp16
    overflows here too (see Trainer.AMP_DTYPES). Evaluation in fp32 is ~2x
    slower and remains available via amp=False.
    """
    h, w = ir.shape[-2:]
    ir_p = pad_to_multiple(ir, pad_multiple)
    vi_p = pad_to_multiple(vi, pad_multiple)

    ir_d, vi_d = ir_p.to(device, non_blocking=True), vi_p.to(device, non_blocking=True)
    if amp and device.type == "cuda":
        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }.get(amp_dtype, torch.bfloat16)
        with torch.autocast("cuda", dtype=dtype):
            out = model(ir_d, vi_d)
    else:
        out = model(ir_d, vi_d)
    return out[..., :h, :w].float()


def evaluate_model(
    model: nn.Module,
    loader,
    *,
    device: torch.device | str = "cuda",
    amp: bool = True,
    amp_dtype: str = "bfloat16",
    per_image: Callable[[np.ndarray, np.ndarray, np.ndarray], dict[str, float]] | None = None,
    max_images: int | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Compute fusion metrics over a loader.

    The loader is streamed (never all-in-RAM): on this 8 GB GPU, holding a few
    hundred 480x640 triples as float tensors is unnecessary and wasteful.

    Args:
        per_image: optional extra metric hook, e.g. the YOLO mAP pass. Kept out
            of the main loop so a slow detector cannot slow every other metric.
    """
    device = torch.device(device)
    model = model.to(device).eval()

    totals: dict[str, float] = {}
    count = 0
    per_image_rows: list[dict[str, Any]] = []
    t0 = time.time()

    for batch in loader:
        ir, vi = batch["ir"], batch["vi"]
        fused = fuse_batch(
            model, ir, vi, device=device, amp=amp, amp_dtype=amp_dtype
        )
        for i in range(fused.shape[0]):
            if max_images is not None and count >= max_images:
                break
            f = tensor_to_uint8(fused[i : i + 1])
            r = tensor_to_uint8_gray(ir[i : i + 1])
            v = tensor_to_uint8(vi[i : i + 1])
            res = compute_all(f, r, v).as_dict()
            if per_image is not None:
                res.update(per_image(f, r, v))
            row = {"pair_id": batch["pair_id"][i] if "pair_id" in batch else str(count), **res}
            per_image_rows.append(row)
            for k, val in res.items():
                if isinstance(val, (int, float)) and np.isfinite(val):
                    totals[k] = totals.get(k, 0.0) + float(val)
            count += 1
        if max_images is not None and count >= max_images:
            break
        if progress is not None:
            progress(count, len(loader))

    means = {k: v / max(count, 1) for k, v in totals.items()}
    return {
        "num_images": count,
        "metrics": means,
        "per_image": per_image_rows,
        "elapsed_s": round(time.time() - t0, 2),
    }


def run_evaluation(
    model_path: str | Path,
    model_name: str,
    model_cfg: dict,
    loader,
    registry: dict,
    *,
    device: str = "cuda",
    amp: bool = True,
    amp_dtype: str = "bfloat16",
    out_json: str | Path | None = None,
    max_images: int | None = None,
    yolo: dict | None = None,
    yolo_images: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Load weights, evaluate, and write a fully-provenanced metrics.json."""
    device_t = torch.device(device)
    from ..utils.registry import build_model

    model = build_model(model_name, model_cfg)
    if model_path:
        load_model_weights(model, model_path, device=str(device_t))

    result = evaluate_model(
        model, loader, device=device_t, amp=amp, amp_dtype=amp_dtype,
        max_images=max_images,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    payload: dict[str, Any] = {
        "model": {
            "name": model_name,
            "config": model_cfg,
            "params": n_params,
            "weights": str(model_path) if model_path else None,
        },
        "dataset": {
            "dataset_id": registry.get("dataset_id"),
            "manifest_sha256": registry.get("manifest_sha256"),
            "splits": registry.get("splits"),
            "retrieved_at": registry.get("retrieved_at"),
        },
        "evaluation": {
            "num_images": result["num_images"],
            "elapsed_s": result["elapsed_s"],
            "amp": amp,
            "amp_dtype": amp_dtype if amp else "float32",
            "device": str(device_t),
            "torch": torch.__version__,
        },
        "metrics": result["metrics"],
        "per_image": result["per_image"],
    }

    if yolo is not None and yolo_images:
        from ..metrics.yolo_map import attach_yolo_metrics

        attach_yolo_metrics(payload, yolo, yolo_images)

    if out_json:
        out = Path(out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload