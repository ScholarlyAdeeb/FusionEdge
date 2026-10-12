"""Output-head behaviour: range, gradient flow, and actual learning.

Context
-------
The SimpleCNN baseline originally ended in `clamp(x, 0, 1)`. `clamp` has zero
gradient outside its range, and on real IR/VI data ~58% of output pixels ended up
pinned exactly on 0.0 or 1.0, which froze the network: 60 Adam steps moved the
loss by 0.000. The head is now a sigmoid.

These tests guard the *property* that was violated, not the exact architecture.
The bar is deliberately not "zero saturated pixels": a well-trained model may
legitimately output near-0 or near-1 values in dark or blown-out regions, and a
test that forbids that would be brittle and wrong. What must hold is that the
output is in range, that gradient reaches the head, and that the loss actually
descends on real data.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fusionedge.losses import FusionLoss
from fusionedge.models.baseline import SimpleCNNBaseline
from fusionedge.models.fusionedge import FusionEdge
from fusionedge.utils.registry import build_model

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def saturation_fraction(x: torch.Tensor, eps: float = 1e-3) -> float:
    """Fraction of pixels at (or within `eps` of) the [0,1] boundaries."""
    return float(((x <= eps) | (x >= 1 - eps)).float().mean())


def real_batch(batch_size: int = 4, crop: int = 128):
    """A batch of REAL MSRS crops, or a clear skip if the dataset is absent.

    Statistics on synthetic noise would not reproduce the failure: random images
    have a very different gradient distribution from aligned road scenes, and the
    original bug only showed up on the real distribution.
    """
    from fusionedge.data import build_dataloaders, load_registry

    try:
        load_registry("MSRS")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"MSRS not available: {exc}")
    loader, _, _ = build_dataloaders(
        "MSRS", batch_size=batch_size, num_workers=0, crop=crop, limit_train=batch_size
    )
    return next(iter(loader))


# --------------------------------------------------------------------------- #
# output range
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model_fn", ["baseline", "fusionedge"])
def test_output_is_within_unit_range(model_fn):
    m = build_model(model_fn, {"width": 16, "depth": 1})
    ir, vi = torch.rand(2, 1, 64, 64), torch.rand(2, 3, 64, 64)
    out = m(ir, vi)
    assert out.min() >= 0.0, "output escaped below 0"
    assert out.max() <= 1.0, "output escaped above 1"


def test_baseline_output_is_strictly_interior():
    """sigmoid is strictly (0,1), so no pixel can be exactly on a boundary.
    This is the property clamp violated."""
    m = SimpleCNNBaseline(width=16, depth=1).eval()
    with torch.no_grad():
        out = m(torch.rand(4, 1, 64, 64), torch.rand(4, 3, 64, 64))
    assert out.min() > 0.0 and out.max() < 1.0


def test_saturation_is_low_on_real_data_not_zero():
    """The realistic assertion: saturation should be *low*, not zero.

    Requiring exactly zero would be brittle -- a genuinely dark night scene can
    legitimately have near-zero output. What must not recur is the clamp regime.
    """
    try:
        b = real_batch()
    except pytest.skip.Exception:
        raise
    m = build_model("baseline", {"width": 32, "depth": 4}).eval()
    with torch.no_grad():
        out = m(b["ir"], b["vi"])
    sat = saturation_fraction(out)
    assert sat < 0.10, (
        f"{sat:.1%} of pixels sit at the [0,1] boundary -- if this is high, the "
        f"output head is saturating again and gradients are being lost"
    )


# --------------------------------------------------------------------------- #
# gradient flow
# --------------------------------------------------------------------------- #
def test_gradient_reaches_every_baseline_parameter():
    m = SimpleCNNBaseline(width=16, depth=2)
    out = m(torch.rand(2, 1, 32, 32), torch.rand(2, 3, 32, 32))
    out.mean().backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_head_gradient_is_not_trivially_small():
    """A saturated head produces a near-zero gradient. Pin the head's gradient
    magnitude relative to the overall gradient so a regression is visible."""
    m = SimpleCNNBaseline(width=16, depth=2)
    out = m(torch.rand(2, 1, 32, 32), torch.rand(2, 3, 32, 32))
    out.mean().backward()
    head_grad = m.head.weight.grad.norm().item()
    total = sum(
        p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None
    ) ** 0.5
    assert head_grad > 0.0
    assert total > 0.0
    assert head_grad / total > 1e-3, "head is receiving a negligible share of gradient"


def test_gradient_survives_extreme_inputs():
    """Dark and saturated inputs are the worst case for a saturating output head.

    Note the deliberate exclusion of input == 0.0. With an all-zero input the
    stem's bias-free convolutions produce all-zero activations, so the head's
    weight gradient is exactly zero -- the convolution sees no input to correlate.
    That is a property of convolutions, not of the output head, and including it
    here would assert something false. Small-but-nonzero darkness is the real
    worst case for saturation and is what this checks.
    """
    for val in (0.01, 0.5, 1.0):
        m = SimpleCNNBaseline(width=16, depth=1)
        ir = torch.full((2, 1, 32, 32), val)
        vi = torch.full((2, 3, 32, 32), val)
        out = m(ir, vi)
        out.mean().backward()
        assert m.head.weight.grad is not None
        assert m.head.weight.grad.abs().sum() > 0, f"no head gradient for input={val}"


def test_dark_inputs_do_not_saturate_the_output():
    """A near-black scene must still produce interior values, so the loss keeps
    a gradient there."""
    m = SimpleCNNBaseline(width=16, depth=1).eval()
    with torch.no_grad():
        out = m(torch.full((1, 1, 32, 32), 0.01), torch.full((1, 3, 32, 32), 0.01))
    assert out.min() > 0.0 and out.max() < 1.0
    assert saturation_fraction(out) < 0.05


# --------------------------------------------------------------------------- #
# actual learning (the property the bug violated)
# --------------------------------------------------------------------------- #
def test_baseline_loss_decreases_on_real_msrs_batch():
    """The direct regression test for the original bug.

    60 Adam steps on one real MSRS batch must measurably reduce the fusion loss.
    Under the clamp head this moved by 0.000.
    """
    try:
        b = real_batch(batch_size=4, crop=128)
    except pytest.skip.Exception:
        raise
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ir, vi = b["ir"].to(device), b["vi"].to(device)

    torch.manual_seed(0)
    m = build_model("baseline", {"width": 32, "depth": 4}).to(device)
    loss_fn = FusionLoss().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)

    losses = []
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        loss, _ = loss_fn(m(ir, vi), ir, vi)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    first = float(np.mean(losses[:5]))
    last = float(np.mean(losses[-5:]))
    assert last < first * 0.95, (
        f"loss did not meaningfully decrease: {first:.4f} -> {last:.4f} "
        f"(requires a >5% drop over 60 steps)"
    )


def test_baseline_reduces_loss_on_synthetic_data_too():
    """Not a substitute for the real-data test -- a guard so the property holds
    even when MSRS is unavailable."""
    from fusionedge.data.synthetic import make_pair

    ir, vi = make_pair(seed=0, height=128, width=128)
    ir, vi = ir.unsqueeze(0), vi.unsqueeze(0)

    torch.manual_seed(0)
    m = SimpleCNNBaseline(width=16, depth=2)
    loss_fn = FusionLoss()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)

    first = None
    for i in range(40):
        opt.zero_grad(set_to_none=True)
        loss, _ = loss_fn(m(ir, vi), ir, vi)
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
    assert loss.item() < first


def test_fusionedge_starts_from_the_visible_image():
    """FusionEdge's residual head means an untrained model is exactly the visible
    input -- which is why it never hit the saturation problem."""
    m = FusionEdge(width=16, depth=1).eval()
    vi = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        out = m(torch.rand(1, 1, 32, 32), vi)
    assert torch.allclose(out, vi, atol=1e-5)
    assert saturation_fraction(out) < 0.05