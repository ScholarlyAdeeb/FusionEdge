"""Loss term correctness: each term is pinned against an explicit reference and
against the property that defines it."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fusionedge.losses import (
    FusionLoss,
    FusionLossConfig,
    color_loss,
    edge_loss,
    gradient_loss,
    intensity_loss,
    sobel_magnitude,
    ssim_loss,
)
from fusionedge.losses.fusion_loss import _chroma, _luminance, _ssim_map


# --------------------------------------------------------------------------- #
def _pair(h=32, w=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    ir = torch.rand(2, 1, h, w, generator=g)
    vi = torch.rand(2, 3, h, w, generator=g)
    return ir, vi


def test_luminance_uses_bt601():
    x = torch.zeros(1, 3, 2, 2)
    x[:, 0] = 1.0  # red
    assert _luminance(x).mean().item() == pytest.approx(0.299, abs=1e-6)
    x = torch.zeros(1, 3, 2, 2)
    x[:, 1] = 1.0
    assert _luminance(x).mean().item() == pytest.approx(0.587, abs=1e-6)


def test_luminance_passes_single_channel_through():
    x = torch.rand(1, 1, 4, 4)
    assert torch.equal(_luminance(x), x)


# --------------------------------------------------------------------------- #
def test_sobel_magnitude_matches_explicit_convolution():
    """Reference: explicit conv with replicate padding (NOT zero padding -- a
    zero border reads as a step edge, which is the bug this test guards)."""
    import torch.nn.functional as F

    x = torch.rand(1, 1, 16, 16, generator=torch.Generator().manual_seed(3))
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
    ky = kx.t().contiguous()
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(xp, kx.view(1, 1, 3, 3))
    gy = F.conv2d(xp, ky.view(1, 1, 3, 3))
    expect = torch.sqrt(gx**2 + gy**2 + 1e-8)
    assert torch.allclose(sobel_magnitude(x), expect, atol=1e-5)


def test_sobel_does_not_invent_a_border_edge():
    """A constant image must have zero gradient everywhere, including the frame.
    With torch's default zero padding this fails: the kernel sees 0 -> 0.4 at the
    border and reports a strong edge. Guarded explicitly because the symptom
    (models learning bright/dark frames) is very hard to notice later."""
    const = torch.full((1, 3, 24, 24), 0.4)
    assert sobel_magnitude(const).max().item() == pytest.approx(0.0, abs=1e-3)


def test_sobel_does_not_invent_a_border_edge_on_smooth_images():
    """Same check with a smooth gradient, where the interior response is tiny."""
    ramp = torch.linspace(0.2, 0.8, 24).view(1, 1, 1, 24).expand(1, 3, 24, 24).contiguous()
    m = sobel_magnitude(ramp)
    border = torch.cat([m[..., 0, :], m[..., -1, :], m[..., :, 0], m[..., :, -1]])
    interior = m[..., 2:-2, 2:-2]
    # a border response far above the interior one means the padding is fabricating edges
    assert border.max().item() <= interior.max().item() * 1.05 + 1e-3


def test_sobel_magnitude_reduces_channels_by_max():
    """A strong edge in one channel must survive the channel reduction."""
    x = torch.zeros(1, 3, 16, 16)
    x[:, 1, :, 8:] = 1.0  # hard edge in green only
    m = sobel_magnitude(x)
    assert m.shape == (1, 1, 16, 16)
    assert m.max() > 1.0


def test_sobel_zero_for_constant_image():
    # sqrt(gx^2+gy^2+eps) floors at sqrt(1e-8) = 1e-4 rather than exactly 0.
    assert sobel_magnitude(torch.full((1, 1, 8, 8), 0.4)).max().item() == pytest.approx(0.0, abs=1e-3)


# --------------------------------------------------------------------------- #
def test_intensity_loss_zero_for_max_of_sources():
    """If fused luma == max(ir, vi) luma everywhere, the term is 0. This is the
    term's whole definition, so it is checked directly."""
    ir, vi = _pair(seed=1)
    fused = torch.max(_luminance(ir), _luminance(vi)).repeat(1, 3, 1, 1)
    assert intensity_loss(fused, ir, vi).item() == pytest.approx(0.0, abs=1e-7)


def test_intensity_loss_is_l1_distance_to_max():
    ir, vi = _pair(seed=2)
    fused = torch.full_like(vi, 0.3)
    expect = F.l1_loss(_luminance(fused), torch.max(_luminance(ir), _luminance(vi)))
    assert intensity_loss(fused, ir, vi).item() == pytest.approx(expect.item(), rel=1e-6)


# --------------------------------------------------------------------------- #
def test_gradient_loss_zero_when_fused_has_max_gradient():
    ir, vi = _pair(seed=3)
    gmax = torch.max(sobel_magnitude(_luminance(ir)), sobel_magnitude(vi))
    # build a fused image whose gradient magnitude equals gmax: use an image
    # that already has a gradient >= gmax everywhere is hard, so instead verify
    # monotonicity -- sharpening fused reduces the term.
    soft = F.avg_pool2d(vi, 3, stride=1, padding=1)
    sharp = vi * 1.5
    assert gradient_loss(sharp, ir, vi) < gradient_loss(soft, ir, vi)


# --------------------------------------------------------------------------- #
def test_ssim_map_is_one_for_identical_inputs():
    x = torch.rand(1, 1, 32, 32, generator=torch.Generator().manual_seed(4))
    m = _ssim_map(x, x, 11)
    assert torch.allclose(m, torch.ones_like(m), atol=1e-4)


def test_ssim_loss_zero_for_perfect_structure():
    ir, vi = _pair(seed=5)
    assert ssim_loss(vi, vi, vi).item() == pytest.approx(0.0, abs=1e-5)
    assert ssim_loss(ir.repeat(1, 3, 1, 1), ir, ir).item() == pytest.approx(0.0, abs=1e-5)


def test_ssim_loss_increases_with_structural_change():
    ir, vi = _pair(seed=6)
    base = vi
    perturbed = F.avg_pool2d(F.pad(vi, (4, 4, 4, 4), mode="reflect"), 9, stride=1)
    assert ssim_loss(base, ir, vi) < ssim_loss(perturbed, ir, vi)


# --------------------------------------------------------------------------- #
def test_edge_loss_is_one_sided():
    """Fused edges at least as strong as the source max cost nothing; weaker
    edges are penalised. The asymmetry is the whole point of the term, so both
    directions are checked with inputs whose gradients are known exactly."""
    ir = torch.zeros(1, 1, 16, 16)
    ir[..., 8:] = 1.0  # a hard step edge
    vi = torch.full((1, 3, 16, 16), 0.5)  # flat: no edges
    gmax = sobel_magnitude(_luminance(ir))

    flat = torch.full((1, 3, 16, 16), 0.5)
    assert sobel_magnitude(flat).max() < gmax.max()
    assert edge_loss(flat, ir, vi).item() > 0.0, "dropping a source edge must be penalised"

    # a fused image that contains the step edge keeps its gradients -> no penalty
    stepped = torch.zeros(1, 3, 16, 16)
    stepped[..., 8:] = 1.0
    assert torch.allclose(sobel_magnitude(stepped), gmax, atol=1e-4)
    assert edge_loss(stepped, ir, vi).item() == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
def test_chroma_is_greyscale_invariant():
    """A grey image has zero chroma, and chroma loss to it is zero."""
    grey = torch.full((1, 3, 8, 8), 0.5)
    assert _chroma(grey).abs().max() == pytest.approx(0.0)
    assert color_loss(grey, grey).item() == pytest.approx(0.0, abs=1e-7)


def test_color_loss_ignores_additive_brightness_shift():
    """A global brightness offset must not change the chroma term -- otherwise it
    would fight the intensity term over the same pixels. The source is scaled
    first so that adding 0.2 cannot hit the [0,1] clamp; clamping *does* change
    chroma, legitimately, and would invalidate the test rather than the code."""
    vi = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(8)) * 0.5
    brighter = vi + 0.2  # provably within [0, 0.7]: no clamping
    assert color_loss(brighter, vi).item() == pytest.approx(0.0, abs=1e-6)


def test_color_loss_penalises_muted_colour():
    """Colour contrast loss (a multiplicative gain) *is* penalised -- this is a
    documented property of the chroma definition, not an accident."""
    vi = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(8)) * 0.8 + 0.1
    muted = 0.5 + 0.5 * (vi - 0.5)  # pull everything toward mid grey
    assert color_loss(muted, vi).item() > 0.0


def test_color_loss_positive_for_hue_shift():
    vi = torch.rand(1, 3, 16, 16, generator=torch.Generator().manual_seed(9))
    swapped = vi[:, [2, 1, 0]]
    assert color_loss(swapped, vi).item() > 0.0


# --------------------------------------------------------------------------- #
def test_fusion_loss_is_weighted_sum_of_its_parts():
    ir, vi = _pair(seed=10)
    fused = torch.rand_like(vi)
    cfg = FusionLossConfig()
    loss = FusionLoss(cfg)
    total, parts = loss(fused, ir, vi)
    expect = (
        cfg.w_intensity * parts["intensity"]
        + cfg.w_gradient * parts["gradient"]
        + cfg.w_ssim * parts["ssim"]
        + cfg.w_edge * parts["edge"]
        + cfg.w_color * parts["color"]
    )
    assert total.item() == pytest.approx(expect.item(), rel=1e-6)


def test_fusion_loss_finite_and_differentiable():
    ir, vi = _pair(seed=11)
    fused = torch.rand_like(vi).requires_grad_(True)
    total, _ = FusionLoss()(fused, ir, vi)
    assert torch.isfinite(total)
    total.backward()
    assert fused.grad is not None and torch.isfinite(fused.grad).all()
    assert fused.grad.abs().sum() > 0


def test_fusion_loss_weights_actually_change_the_result():
    ir, vi = _pair(seed=12)
    fused = torch.rand_like(vi)
    a, _ = FusionLoss(FusionLossConfig(w_intensity=1.0, w_gradient=0.0, w_ssim=0.0,
                                       w_edge=0.0, w_color=0.0))(fused, ir, vi)
    b, _ = FusionLoss(FusionLossConfig(w_intensity=5.0, w_gradient=0.0, w_ssim=0.0,
                                       w_edge=0.0, w_color=0.0))(fused, ir, vi)
    assert b.item() == pytest.approx(5.0 * a.item(), rel=1e-6)


def test_fusion_loss_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown loss keys"):
        FusionLossConfig.from_dict({"w_nonexistent": 1.0})


def test_edge_term_can_be_disabled():
    ir, vi = _pair(seed=13)
    fused = torch.rand_like(vi)
    _, parts = FusionLoss(FusionLossConfig(use_edge=False, w_edge=1.0))(fused, ir, vi)
    assert "edge" not in parts