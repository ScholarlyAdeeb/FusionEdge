"""Individual loss terms + the configurable composite `FusionLoss`.

Everything is implemented in torch so the losses are differentiable w.r.t. the
fused output only, run under autocast, and are unit-tested against independent
reference implementations (scikit-image for SSIM, an explicit Sobel matrix for
gradients) in tests/test_losses.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# gradient / edge primitives
# --------------------------------------------------------------------------- #
_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_SOBEL_Y = _SOBEL_X.t().contiguous()


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    kx = _SOBEL_X.to(device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(device=device, dtype=dtype).view(1, 1, 3, 3)
    return kx, ky


def sobel_magnitude(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """First-order gradient magnitude via 3x3 Sobel. (B,C,H,W) -> (B,1,H,W).

    Accepts any channel count; the response is computed per channel and reduced
    by max, which matches the usual fusion convention: an edge exists if it is
    strong in *any* channel (a red edge is an edge).

    Border handling matters here. Padding with zeros (the torch default) makes
    the kernel see a step from 0 up to the image value at every border pixel, so
    a *constant* image reports a strong edge all the way round -- and the
    gradient loss would then reward the network for synthesising a bright frame.
    The border is replicate-padded instead, which is consistent for any constant
    or smooth image.
    """
    b, c, h, w = x.shape
    kx, ky = _sobel_kernels(x.device, x.dtype)  # each (1,1,3,3)
    # collapse (B,C,H,W) -> (B*C,1,H,W) so one single-channel kernel applies to
    # every plane at once; conv2d then sees in_channels == 1, matching the kernel.
    xf = F.pad(x.reshape(b * c, 1, h, w), (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(xf, kx)
    gy = F.conv2d(xf, ky)
    mag = torch.sqrt(gx * gx + gy * gy + eps)
    return mag.view(b, c, h, w).amax(dim=1, keepdim=True)


def _luminance(x: torch.Tensor) -> torch.Tensor:
    """(B,C,H,W) -> (B,1,H,W) luma with Rec.601 weights (ITU-R BT.601)."""
    if x.shape[1] == 1:
        return x
    w = torch.tensor([0.299, 0.587, 0.114], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


# --------------------------------------------------------------------------- #
# loss terms
# --------------------------------------------------------------------------- #
def intensity_loss(fused: torch.Tensor, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
    """L1 of fused luma against the *pixel-wise max* of source luma.

    Using max rather than mean is what pushes thermal targets into the output
    even where the visible image is bright; a mean target would make the network
    hedge between the two in exactly the salient regions.
    """
    f = _luminance(fused)
    target = torch.max(_luminance(ir), _luminance(vi))
    return F.l1_loss(f, target)


def gradient_loss(fused: torch.Tensor, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
    """L1 of fused gradient magnitude against the max of the source magnitudes.

    Classic fusion gradient objective (Ma et al.), kept as a building block
    rather than copied from any single network's implementation.
    """
    gf = sobel_magnitude(fused)
    gi = sobel_magnitude(_luminance(ir))
    gv = sobel_magnitude(vi)
    gmax = torch.max(gi, gv)
    return F.l1_loss(gf, gmax)


def _ssim_map(
    x: torch.Tensor, y: torch.Tensor, window: int = 11, sigma: float = 1.5, data_range: float = 1.0
) -> torch.Tensor:
    """Per-pixel SSIM map for (B,C,H,W) inputs in [0,1]."""
    coords = torch.arange(window, dtype=x.dtype, device=x.device) - (window - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    k = torch.outer(g, g).view(1, 1, window, window).expand(x.shape[1], 1, window, window)

    pad = window // 2
    mu_x = F.conv2d(x, k, padding=pad, groups=x.shape[1])
    mu_y = F.conv2d(y, k, padding=pad, groups=y.shape[1])
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sx = F.conv2d(x * x, k, padding=pad, groups=x.shape[1]) - mu_x2
    sy = F.conv2d(y * y, k, padding=pad, groups=y.shape[1]) - mu_y2
    sxy = F.conv2d(x * y, k, padding=pad, groups=x.shape[1]) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mu_xy + c1) * (2 * sxy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sx + sy + c2)
    return num / den


def ssim_loss(fused: torch.Tensor, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
    """1 - mean SSIM(fused, source) averaged over the two sources.

    Averaged (not max) over sources: we want structural agreement with *both*
    inputs, which discourages the "fused output is structurally equal to the
    infrared image" failure mode that a max-based SSIM term can fall into.
    """
    fi = _luminance(fused)
    s_ir = _ssim_map(fi, _luminance(ir)).mean()
    s_vi = _ssim_map(fi, _luminance(vi)).mean()
    return 1.0 - 0.5 * (s_ir + s_vi)


def edge_loss(fused: torch.Tensor, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
    """Penalise fused edges that are *weaker* than the strongest source edge.

    one-sided on purpose: fused is allowed (and desired) to add edges, but not to
    lose them. A symmetric penalty would also punish the desirable case of a
    fused image with more detail than either input.
    """
    gf = sobel_magnitude(fused)
    gmax = torch.max(sobel_magnitude(_luminance(ir)), sobel_magnitude(vi))
    return F.relu(gmax - gf).mean()


def color_loss(fused: torch.Tensor, vi: torch.Tensor, ir: torch.Tensor | None = None) -> torch.Tensor:
    """Chroma distance between fused and visible.

    Chroma (colour without luminance) is separated from luma so the term cannot
    fight the intensity term over the same pixels. Infrared has no colour, so
    chroma is compared against the visible image only; that is the reason a fused
    output keeps plausible colour instead of turning thermal targets orange.
    """
    cf, cv = _chroma(fused), _chroma(vi)
    return F.l1_loss(cf, cv)


def _chroma(x: torch.Tensor) -> torch.Tensor:
    """Opponency chroma: [R-G, (R+G)/2 - B].

    Both components are differences, so the term is invariant to a global
    additive brightness shift (the classic "the fused image came out too
    bright/dark" error) and therefore does not fight the intensity term, which
    owns luma. It is deliberately *not* invariant to a multiplicative gain: a
    fused image that is uniformly 0.5x the visible brightness has genuinely lost
    colour contrast, and the term should say so.
    """
    if x.shape[1] == 1:
        return torch.zeros_like(x)
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return torch.cat([r - g, 0.5 * (r + g) - b], dim=1)


# --------------------------------------------------------------------------- #
# composite
# --------------------------------------------------------------------------- #
@dataclass
class FusionLossConfig:
    w_intensity: float = 8.0
    w_gradient: float = 10.0
    w_ssim: float = 2.0
    w_edge: float = 1.0
    w_color: float = 1.0
    use_edge: bool = True

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "FusionLossConfig":
        if not d:
            return cls()
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown loss keys: {sorted(unknown)}; valid: {sorted(known)}")
        return cls(**d)


def fp32_loss(loss_fn, fused: torch.Tensor, ir: torch.Tensor, vi: torch.Tensor):
    """Evaluate `loss_fn` in genuine float32, even if called inside an autocast block.

    Casting the *inputs* with `.float()` is not enough: autocast re-casts every
    conv2d inside the loss (Sobel, SSIM Gaussian) back to bf16/fp16. SSIM's
    variance E[x^2] - mu^2 then suffers catastrophic cancellation, and the
    gradients of these ops are corrupted -- NaN on CPU, finite-but-wrong on CUDA.
    Measured 2026-10-14: the same baseline trained 300 steps reaches intensity
    L1 0.032 with the loss in fp32, and NaN with the loss inside bf16 autocast.
    All GPU baseline runs up to v5 trained under the corrupted loss.
    """
    with torch.autocast(device_type=fused.device.type, enabled=False):
        return loss_fn(fused.float(), ir.float(), vi.float())


class FusionLoss(nn.Module):
    """Weighted sum of the fusion loss terms.

    `forward` returns (total, parts) so the training loop can log each term's
    contribution -- otherwise a silently-misconfigured weight is invisible.
    """

    def __init__(self, config: FusionLossConfig | dict | None = None):
        super().__init__()
        if isinstance(config, FusionLossConfig):
            self.cfg = config
        else:
            self.cfg = FusionLossConfig.from_dict(config)

    def forward(
        self, fused: torch.Tensor, ir: torch.Tensor, vi: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        parts: dict[str, torch.Tensor] = {}
        total = fused.new_zeros(())

        if self.cfg.w_intensity:
            parts["intensity"] = intensity_loss(fused, ir, vi)
            total = total + self.cfg.w_intensity * parts["intensity"]
        if self.cfg.w_gradient:
            parts["gradient"] = gradient_loss(fused, ir, vi)
            total = total + self.cfg.w_gradient * parts["gradient"]
        if self.cfg.w_ssim:
            parts["ssim"] = ssim_loss(fused, ir, vi)
            total = total + self.cfg.w_ssim * parts["ssim"]
        if self.cfg.use_edge and self.cfg.w_edge:
            parts["edge"] = edge_loss(fused, ir, vi)
            total = total + self.cfg.w_edge * parts["edge"]
        if self.cfg.w_color:
            parts["color"] = color_loss(fused, vi)
            total = total + self.cfg.w_color * parts["color"]

        return total, parts

    def extra_repr(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in asdict(self.cfg).items())