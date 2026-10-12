"""Fusion losses.

All terms take `fused` (B,3,H,W), `ir` (B,1,H,W), `vi` (B,3,H,W) and return a
scalar. Nothing here depends on a specific architecture, so the same loss object
is reused by the baseline and by FusionEdge -- which is what makes the
architecture comparison a controlled one.

The default objective is

    L = w_int  * L_intensity        (fused should retain the salient pixels)
      + w_grad * L_gradient         (fused should keep the strongest gradient)
      + w_ssim * L_ssim             (fused should stay structurally consistent)
      + w_edge * L_edge             (fused edges should dominate source edges)
      + w_color * L_color           (fused chroma should stay close to visible)

Weighted L1 is used rather than L2 because a squared penalty on the handful of
pixels where the two modalities disagree (glare, saturated lamps, hot exhaust)
otherwise dominates the gradient, which is exactly where fusion behaviour is
decided. The gradient and SSIM terms are what keep the result from degenerating
into a blurry average.
"""

from .fusion_loss import (
    FusionLoss,
    FusionLossConfig,
    edge_loss,
    gradient_loss,
    intensity_loss,
    ssim_loss,
    color_loss,
    sobel_magnitude,
)

__all__ = [
    "FusionLoss",
    "FusionLossConfig",
    "edge_loss",
    "gradient_loss",
    "intensity_loss",
    "ssim_loss",
    "color_loss",
    "sobel_magnitude",
]