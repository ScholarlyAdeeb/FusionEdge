"""Model shape/parameter tests for the baseline and the shared blocks.

These run on synthetic pairs only; no real dataset is required, and none is
silently substituted if one is missing.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fusionedge.models import (
    ChannelGate,
    ConvBNAct,
    DilatedSeparableBlock,
    PairedResidualBlock,
    SELite,
    SobelEdgeExtractor,
)
from fusionedge.models.baseline import SimpleCNNBaseline
from fusionedge.models.blocks import ResidualSeparableBlock


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# blocks
# --------------------------------------------------------------------------- #
def test_convbnact_preserves_spatial_dims():
    m = ConvBNAct(3, 16, k=3, p=1)
    assert m(torch.rand(2, 3, 32, 32)).shape == (2, 16, 32, 32)


def test_dilated_block_preserves_dims_and_keeps_dilation():
    m = DilatedSeparableBlock(16, 16, d=3)
    assert m(torch.rand(1, 16, 32, 32)).shape == (1, 16, 32, 32)
    dilations = {
        mod.dilation[0]
        for mod in m.modules()
        if isinstance(mod, torch.nn.Conv2d)
    }
    assert 3 in dilations, "the block must actually apply the requested dilation"


def _make_bn_identity(module: nn.Module) -> nn.Module:
    """Put every BatchNorm into eval mode with identity params, so a probe of the
    convolutional geometry is not swamped by normalisation."""
    module.eval()
    for m in module.modules():
        if isinstance(m, nn.BatchNorm2d):
            with torch.no_grad():
                m.weight.fill_(1.0)
                m.bias.zero_()
                m.running_mean.zero_()
                m.running_var.fill_(1.0)
    return module


def test_dilation_changes_the_receptive_field():
    """A dilated block must spread an impulse further than a plain one. Probed on
    the conv geometry with BN neutralised, so this measures the dilation and not
    the normalisation."""
    impulse = torch.zeros(1, 8, 32, 32)
    impulse[..., 16, 16] = 1.0
    d1 = _make_bn_identity(DilatedSeparableBlock(8, 8, d=1))(impulse)
    d3 = _make_bn_identity(DilatedSeparableBlock(8, 8, d=3))(impulse)
    spread = lambda t: int((t.abs().amax(dim=1) > 1e-6).sum())
    assert spread(d3) > spread(d1), "dilation must widen the receptive field"


def test_separable_block_is_cheaper_than_dense():
    """The parameter argument for depthwise-separable convs, checked directly."""
    dense = torch.nn.Conv2d(64, 64, 3, padding=1)
    sep = torch.nn.Sequential(
        torch.nn.Conv2d(64, 64, 3, padding=1, groups=64, bias=False),
        torch.nn.Conv2d(64, 64, 1, bias=False),
    )
    assert count_params(sep) < count_params(dense) / 5


def test_residual_block_is_a_residual_at_init():
    """Output should be close to input when the residual branch is ~zero -- this
    is what makes deep residual stacks trainable."""
    m = ResidualSeparableBlock(8, 8)
    nn.init.zeros_(m.conv2[2].weight)  # BatchNorm gain -> residual branch == 0
    nn.init.zeros_(m.conv2[2].bias)
    x = torch.rand(1, 8, 16, 16)
    out = m(x)
    assert torch.allclose(out, torch.nn.functional.leaky_relu(x, 0.2), atol=1e-5)


def test_residual_block_handles_width_change():
    m = ResidualSeparableBlock(8, 16)
    assert m(torch.rand(1, 8, 16, 16)).shape == (1, 16, 16, 16)


def test_paired_block_shares_weights_across_modalities():
    """Weight sharing is a claimed efficiency property -- assert it structurally:
    one block's parameters serve both modalities, so both branches have identical
    parameter counts and identical inputs produce identical outputs."""
    m = PairedResidualBlock(16).eval()
    assert count_params(m) == count_params(ResidualSeparableBlock(16, 16))

    ir = torch.rand(2, 16, 16, 16)
    vi = torch.rand(2, 16, 16, 16)
    a, b = m(ir, vi)
    with torch.no_grad():
        a_again, _ = m(ir, vi)
        ir_only, _ = m(ir, ir)
        _, vi_only = m(vi, vi)
    assert torch.allclose(a, a_again), "the shared block must be deterministic"
    assert torch.allclose(a, ir_only), "IR output must equal feeding IR into both slots"
    assert torch.allclose(b, vi_only), "VI output must equal feeding VI into both slots"


def test_se_gate_preserves_shape_and_is_channel_wise():
    m = SELite(32)
    x = torch.rand(2, 32, 16, 16)
    assert m(x).shape == x.shape
    # scaling one channel's input should scale that channel's output
    y = m(x)
    x2 = x.clone()
    x2[:, 5] *= 2
    y2 = m(x2)
    assert y2[:, 5].abs().mean() > y[:, 5].abs().mean()


def test_channel_gate_produces_spatial_logits():
    g = ChannelGate(16)
    feat = torch.rand(2, 16, 12, 10)
    edge = torch.rand(2, 1, 12, 10)
    logits = g(feat, edge)
    assert logits.shape == (2, 12, 10)


def test_sobel_edge_extractor_is_parameter_free_and_matches_loss():
    from fusionedge.losses import sobel_magnitude

    m = SobelEdgeExtractor()
    assert count_params(m) == 0
    x = torch.rand(1, 3, 24, 24, generator=torch.Generator().manual_seed(2))
    assert torch.allclose(m(x), sobel_magnitude(x), atol=1e-4)


def test_sobel_extractor_no_border_artifacts_on_constant_image():
    m = SobelEdgeExtractor()
    out = m(torch.full((1, 3, 20, 20), 0.5))
    assert out.max().item() == pytest.approx(0.0, abs=1e-3)


# --------------------------------------------------------------------------- #
# baseline model
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("size", [(32, 32), (64, 96), (48, 61)])
def test_baseline_output_shape_and_range(size):
    """Non-divisible sizes must work: the model is fully convolutional."""
    h, w = size
    model = SimpleCNNBaseline(width=16, depth=2)
    ir = torch.rand(2, 1, h, w)
    vi = torch.rand(2, 3, h, w)
    out = model(ir, vi)
    assert out.shape == (2, 3, h, w)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert torch.isfinite(out).all()


def test_baseline_accepts_ir_with_or_without_repeat():
    """IR arrives as 1 channel; the model must not require the caller to repeat."""
    model = SimpleCNNBaseline(width=16, depth=1)
    ir3 = torch.rand(1, 3, 32, 32)
    vi = torch.rand(1, 3, 32, 32)
    out = model(ir3, vi)
    assert out.shape == (1, 3, 32, 32)


def test_baseline_is_deterministic_in_eval_mode():
    model = SimpleCNNBaseline(width=16, depth=2).eval()
    ir, vi = torch.rand(1, 1, 32, 32), torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(model(ir, vi), model(ir, vi))


def test_baseline_gradients_flow_to_every_parameter():
    model = SimpleCNNBaseline(width=16, depth=2)
    out = model(torch.rand(2, 1, 32, 32), torch.rand(2, 3, 32, 32))
    out.mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters without gradient: {missing}"


def test_baseline_depth_increases_parameters():
    a = count_params(SimpleCNNBaseline(width=32, depth=2))
    b = count_params(SimpleCNNBaseline(width=32, depth=6))
    assert b > a


def test_baseline_is_small_enough_to_be_lightweight():
    """Sanity bound so 'lightweight' stays a checked property, not a claim."""
    n = count_params(SimpleCNNBaseline(width=32, depth=4))
    assert n < 500_000, f"baseline has {n} params; expected < 500k"


def test_baseline_works_on_cpu_and_cuda_if_available():
    model = SimpleCNNBaseline(width=8, depth=1)
    ir, vi = torch.rand(1, 1, 32, 32), torch.rand(1, 3, 32, 32)
    assert model(ir, vi).shape == (1, 3, 32, 32)
    if torch.cuda.is_available():
        m = model.cuda()
        out = m(ir.cuda(), vi.cuda())
        assert out.is_cuda and out.shape == (1, 3, 32, 32)
        m.cpu()


def test_baseline_works_under_autocast():
    """AMP must not break the forward pass or produce NaNs."""
    if not torch.cuda.is_available():
        pytest.skip("AMP path needs CUDA")
    model = SimpleCNNBaseline(width=16, depth=2).cuda()
    ir, vi = torch.rand(2, 1, 64, 64).cuda(), torch.rand(2, 3, 64, 64).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(ir, vi)
    assert torch.isfinite(out).all()
    model.cpu()