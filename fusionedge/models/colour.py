"""BT.601 full-range (JPEG) RGB <-> YCbCr, used to fuse luminance only.

Standard practice on MSRS (SeAFusion, PIAFusion, ...): the network produces the
fused luminance Y and the chroma (Cb, Cr) is taken unchanged from the visible
image. Infrared has no colour, so any chroma a network "invents" is an artefact;
fusing Y only removes that failure mode for every model equally.

Note: adding the same scalar delta to R, G and B changes Y by exactly delta and
leaves Cb, Cr unchanged (the Cb/Cr rows of the matrix sum to zero), so
FusionEdge's residual head `vi + delta` is already Y-only fusion.
"""

from __future__ import annotations

import torch


def rgb_to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) RGB in [0,1] -> (B,3,H,W) [Y, Cb, Cr], Cb/Cr centred on 0.5."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 + (-0.168736 * r - 0.331264 * g + 0.5 * b)
    cr = 0.5 + (0.5 * r - 0.418688 * g - 0.081312 * b)
    return torch.cat([y, cb, cr], dim=1)


def ycbcr_to_rgb(y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor) -> torch.Tensor:
    """Inverse of rgb_to_ycbcr. Every row has Y-coefficient 1. Not clamped."""
    cb, cr = cb - 0.5, cr - 0.5
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.cat([r, g, b], dim=1)


def recolour(y: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
    """Fused luminance (B,1,H,W) + visible chroma -> RGB in [0,1]."""
    ycc = rgb_to_ycbcr(vi)
    return torch.clamp(ycbcr_to_rgb(y, ycc[:, 1:2], ycc[:, 2:3]), 0.0, 1.0)


def inject_ir(logit: torch.Tensor, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
    """IR-injection head shared by every model that uses it (docs/PROTOCOL.md).

    g = sigmoid(logit) gates how much of the IR excess over the visible luminance
    is added: Y = Y_vi + g * relu(IR - Y_vi), recoloured with the visible chroma.
    It can only brighten, and only where IR is brighter than the visible image.
    """
    y_vi = rgb_to_ycbcr(vi)[:, :1]
    return recolour(y_vi + torch.sigmoid(logit) * torch.relu(ir - y_vi), vi)
