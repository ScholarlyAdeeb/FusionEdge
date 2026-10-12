"""Paired geometric transforms.

Every transform here takes `(ir, vi)` and returns `(ir, vi)` with the *same*
geometric change applied to both. That is not a detail: unaligned crops would
silently bias the intensity/gradient losses toward whichever modality got
shifted, and the resulting model looks fine on EN/AG while producing colour
fringes on real scenes.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def to_tensor_image(img: np.ndarray) -> torch.Tensor:
    """HWC uint8/float ndarray -> CHW float32 tensor in [0, 1]."""
    if img.dtype == np.uint8:
        arr = img.astype(np.float32) / 255.0
    else:
        arr = np.clip(img.astype(np.float32), 0.0, 1.0)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


def pad_to_multiple(x: torch.Tensor, multiple: int = 8, mode: str = "reflect") -> torch.Tensor:
    """Pad (C,H,W) or (B,C,H,W) so H,W are multiples of `multiple`.

    Reflection is preferred over zero-padding because a zero border creates a
    hard artificial edge in the border region, which the gradient and edge
    losses then try to reproduce in the fused output.

    The pad is applied one pixel at a time on purpose. torch's reflect/replicate
    modes require pad < dimension, so a 4x4 image cannot be padded by 4 in one
    call -- but it can be padded by 1 four times. `multiple` is 8, so this loop
    runs at most 7 iterations and is not a hot path.
    """
    if multiple <= 1:
        return x
    h, w = x.shape[-2:]
    need_h = (multiple - h % multiple) % multiple
    need_w = (multiple - w % multiple) % multiple
    if need_h == 0 and need_w == 0:
        return x  # already aligned: hand back the caller's tensor, no copy

    squeeze = x.dim() == 3
    if squeeze:
        x = x.unsqueeze(0)
    for _ in range(need_h):
        x = F.pad(x, (0, 0, 0, 1), mode=mode)
    for _ in range(need_w):
        x = F.pad(x, (0, 1, 0, 0), mode=mode)

    return x.squeeze(0) if squeeze else x


def paired_random_crop(
    ir: torch.Tensor, vi: torch.Tensor, size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random (size, size) crop at a shared origin.

    Tries a few offsets when the crop is only slightly smaller than the image,
    so we do not systematically sample only the top-left region of scenes.
    """
    _, h, w = ir.shape
    if h < size or w < size:
        ph, pw = max(0, size - h), max(0, size - w)
        ir = F.pad(ir, (0, pw, 0, ph), mode="reflect")
        vi = F.pad(vi, (0, pw, 0, ph), mode="reflect")
        h, w = ir.shape[-2:]
    top = left = 0
    if h > size or w > size:
        top = np.random.randint(0, h - size + 1)
        left = np.random.randint(0, w - size + 1)
    return ir[:, top : top + size, left : left + size], vi[:, top : top + size, left : left + size]


def paired_random_hflip(ir: torch.Tensor, vi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Horizontal flip with probability 0.5, applied to both modalities."""
    if np.random.rand() < 0.5:
        return torch.flip(ir, dims=[-1]), torch.flip(vi, dims=[-1])
    return ir, vi


def paired_random_vflip(ir: torch.Tensor, vi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if np.random.rand() < 0.5:
        return torch.flip(ir, dims=[-2]), torch.flip(vi, dims=[-2])
    return ir, vi