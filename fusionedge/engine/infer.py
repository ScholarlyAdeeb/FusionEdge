"""Inference: fuse image pairs from disk.

Used for the qualitative comparison grids and by the Gradio demo. Kept separate
from training so a demo or a figure can never accidentally re-train anything.
"""

from __future__ import annotations

import argparse
import json
import sys
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


def load_model(
    weights: str | Path,
    model_name: str,
    model_cfg: dict | None,
    device: str = "cuda",
) -> torch.nn.Module:
    model = build_model(model_name, model_cfg or {})
    load_model_weights(model, weights, device=device)
    return model.to(device).eval()


def is_trained(weights: str | Path) -> tuple[bool, str]:
    """Whether a checkpoint looks trained, and why.

    An untrained FusionEdge has a zero-initialised output head, so its output is a
    fixed rule rather than anything learned: the visible image ('add' head), or
    the visible image lifted half-way towards brighter IR ('inject' head). Either
    way it is a plausible, reasonable-looking picture that no training shaped. That
    is the most dangerous failure mode in this project: a user runs inference on
    real pairs, sees a fine image, and concludes the model works.

    So checkpoints are labelled. `trainer_state` is written by the Trainer; a
    checkpoint without it, or with all-zero output-head weights, is reported as
    untrained. This is a heuristic on the weights, not proof of training quality
    -- it says "you probably have not trained this", not "this model is good".
    """
    path = Path(weights)
    if not path.exists():
        return False, f"checkpoint does not exist: {path}"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return False, f"checkpoint could not be read: {exc}"

    if "model_state" not in payload:
        return False, "checkpoint has no 'model_state' -- not a FusionEdge checkpoint"
    state = payload["model_state"]
    head_key = next((k for k in state if k.endswith("head.weight")), None)
    if head_key is None:
        return False, "no output head found in state_dict"
    if torch.count_nonzero(state[head_key]) == 0:
        return (
            False,
            "output head is all-zero -- this is the untrained initialisation, and "
            "fusing with it returns the visible image unchanged",
        )
    epochs = payload.get("epoch", 0)
    return True, f"output head is non-zero; checkpoint reports epoch={epochs}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ir", required=True)
    ap.add_argument("--vi", required=True)
    ap.add_argument("--run", default=None,
                    help="experiment dir: use its config.yaml model and checkpoints/last.pt "
                         "(the protocol's reported checkpoint); replaces --weights/--model/--model-cfg")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--model", default="fusionedge")
    ap.add_argument("--model-cfg", default="{}")
    ap.add_argument("--fp32", action="store_true",
                    help="no autocast (matches the holdout evaluation); default is bf16 autocast on CUDA")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--allow-untrained",
        action="store_true",
        help="proceed even if the checkpoint looks untrained (for smoke tests only)",
    )
    args = ap.parse_args()
    if args.run:
        import yaml
        run_cfg = yaml.safe_load((Path(args.run) / "config.yaml").read_text(encoding="utf-8"))
        args.weights = str(Path(args.run) / "checkpoints" / "last.pt")
        args.model = run_cfg["model"]["name"]
        args.model_cfg = json.dumps(run_cfg["model"].get("cfg", {}))
    if not args.weights:
        ap.error("give --run <experiment dir> or --weights <checkpoint>")

    trained, reason = is_trained(args.weights)
    if not trained:
        print(f"REFUSING to fuse: {reason}", file=sys.stderr)
        print(
            "\nAn untrained model returns a fixed, unlearned blend, which looks like "
            "a plausible fusion but reflects no training.\n"
            "Train first (python scripts/run_experiment.py --config configs/"
            "fusionedge.yaml), or pass --allow-untrained for a plumbing test.",
            file=sys.stderr,
        )
        if not args.allow_untrained:
            return 2

    import cv2

    model = load_model(args.weights, args.model, json.loads(args.model_cfg), args.device)
    fused = fuse_pair(model, args.ir, args.vi, device=args.device, amp=not args.fp32)

    # How different is the output from the visible input? If the model is barely
    # using the infrared channel the answer is ~0 and that is worth knowing.
    delta = np.abs(
        fused.astype(np.float64) - cv2.cvtColor(cv2.imread(args.vi), cv2.COLOR_BGR2RGB)
    ).mean()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(fused, cv2.COLOR_RGB2BGR))
    print(f"wrote {out} {fused.shape}")
    print(f"mean |fused - visible| = {delta:.2f} (0.0 means no infrared contribution)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())