"""Inference: fuse image pairs from disk.

Used for the qualitative comparison grids and by the Gradio demo. Kept separate
from training so a demo or a figure can never accidentally re-train anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..data.fusion_dataset import read_image, _to_ir_tensor, _to_vi_tensor
from ..utils.checkpoint import load_model_weights
from ..utils.registry import build_model
from .evaluate import fuse_batch, tensor_to_uint8


def load_model(
    weights: str | Path, model_name: str, model_cfg: dict | None, device: str = "cuda"
) -> torch.nn.Module:
    model = build_model(model_name, model_cfg or {})
    load_model_weights(model, weights, device=device)
    return model.to(device).eval()


@torch.inference_mode()
def fuse_pair(
    model: torch.nn.Module,
    ir_img,
    vi_img,
    device: str = "cuda",
    amp: bool = True,
) -> np.ndarray:
    """ir_img / vi_img: file paths, PIL images, HWC arrays, or CHW tensors.
    Returns a uint8 HWC RGB array."""
    dev = torch.device(device)

    def prep(x, which):
        if isinstance(x, (str, Path)):
            arr = read_image(Path(x))
        elif isinstance(x, np.ndarray):
            arr = x
        elif isinstance(x, torch.Tensor):
            t = x.float()
            if t.dim() == 3 and t.shape[0] in (1, 3):
                t = t.unsqueeze(0)
            return t.clamp(0, 1)
        else:  # PIL
            arr = np.asarray(x)
        if which == "ir":
            return _to_ir_tensor(arr).unsqueeze(0)
        return _to_vi_tensor(arr).unsqueeze(0)

    ir = prep(ir_img, "ir").to(dev)
    vi = prep(vi_img, "vi").to(dev)
    fused = fuse_batch(model, ir, vi, device=dev, amp=amp)
    return tensor_to_uint8(fused)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ir", required=True)
    ap.add_argument("--vi", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--model", default="fusionedge")
    ap.add_argument("--model-cfg", default="{}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    import cv2

    model = load_model(args.weights, args.model, json.loads(args.model_cfg), args.device)
    fused = fuse_pair(model, args.ir, args.vi, device=args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(fused, cv2.COLOR_RGB2BGR))
    print(f"wrote {out} {fused.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())