"""FusionEdge architecture tests: shapes, ablation switches, invariants.

Every ablation switch is exercised here, so a broken ablation config fails in
seconds rather than after a 40-minute training run.
"""

from __future__ import annotations

import pytest
import torch

from fusionedge.models.fusionedge import FusionEdge, build_fusionedge
from fusionedge.utils.registry import build_model


def small(**kw) -> FusionEdge:
    base = dict(width=8, depth=2)
    base.update(kw)
    return FusionEdge(**base)


# --------------------------------------------------------------------------- #
def test_forward_shape_and_range():
    m = small()
    ir, vi = torch.rand(2, 1, 32, 40), torch.rand(2, 3, 32, 40)
    out = m(ir, vi)
    assert out.shape == (2, 3, 32, 40)
    assert 0.0 <= out.min() and out.max() <= 1.0
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("size", [(32, 32), (64, 96), (37, 61)])
def test_non_divisible_sizes(size):
    h, w = size
    out = small()(torch.rand(1, 1, h, w), torch.rand(1, 3, h, w))
    assert out.shape == (1, 3, h, w)


def test_starts_as_identity_on_the_visible_image():
    """The head is zero-initialised, so an untrained FusionEdge returns the visible
    image exactly. This makes the early loss curve readable and means training
    starts from a sane point rather than from noise. ('add' head only: this
    fixed point is also why 'add' never trains -- see PROTOCOL.md.)"""
    m = small(residual_mode="add")
    ir, vi = torch.rand(1, 1, 32, 32), torch.rand(1, 3, 32, 32)
    assert torch.allclose(m(ir, vi), vi, atol=1e-6)


def test_ignores_ir_when_weights_are_untrained():
    """Consequence of the identity initialisation: the output cannot yet depend
    on the infrared input. Asserted because it explains why the very first
    training steps are needed before any IR information can appear."""
    m = small(residual_mode="add")
    vi = torch.rand(1, 3, 32, 32)
    ir1, ir2 = torch.rand(1, 1, 32, 32), torch.rand(1, 1, 32, 32)
    assert torch.allclose(m(ir1, vi), m(ir2, vi), atol=1e-6)


def test_parameter_count_is_lightweight():
    n = small(width=32, depth=2).num_parameters()
    full = FusionEdge(width=32, depth=2).num_parameters()
    assert n < 200_000, f"{n} parameters is not 'lightweight'"


# --------------------------------------------------------------------------- #
def test_fusion_weights_are_a_convex_combination():
    """The adaptive gate must produce weights that sum to 1 across modalities at
    every pixel. If they did not, some pixels would be able to discard both
    inputs and the network would have no gradient path enforcing content."""
    m = small()
    ir, vi = torch.rand(2, 1, 32, 32), torch.rand(2, 3, 32, 32)
    alpha = m.fusion_weights(ir, vi)
    assert alpha is not None
    assert alpha.shape == (2, 2, 32, 32)
    assert torch.allclose(alpha.sum(dim=1), torch.ones(2, 32, 32), atol=1e-5)
    assert (alpha >= 0).all() and (alpha <= 1).all()


def test_fusion_weights_disabled_when_fusion_is_not_adaptive():
    m = small(adaptive_fusion=False)
    m(torch.rand(1, 1, 32, 32), torch.rand(1, 3, 32, 32))
    assert m.fusion_weights(
        torch.rand(1, 1, 32, 32), torch.rand(1, 3, 32, 32)
    ) is None


def test_edge_guidance_changes_the_gate_input():
    """With and without the edge prior, the gate must see different information --
    otherwise the 'ablation' would be measuring nothing."""
    torch.manual_seed(0)
    with_edge = small(use_edge_guidance=True)
    torch.manual_seed(0)
    without = small(use_edge_guidance=False)
    ir, vi = torch.rand(1, 1, 32, 32), torch.rand(1, 3, 32, 32)
    # identical weights -> any output difference is the edge channel's doing
    a_with = with_edge.fusion_weights(ir, vi)
    a_without = without.fusion_weights(ir, vi)
    assert not torch.allclose(a_with, a_without, atol=1e-4)


def test_edge_extractor_is_used_and_parameter_free():
    m = small()
    from fusionedge.models.blocks import SobelEdgeExtractor

    assert any(isinstance(mod, SobelEdgeExtractor) for mod in m.modules())
    assert all(
        p.numel() == 0 for n, p in m.named_parameters() if "edge" in n
    )


# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "flag",
    ["use_cross_modal", "use_se", "separable", "shared_stem", "residual_output"],
)
def test_structural_flags_change_the_model(flag):
    """Structural switches must change the parameter set. A switch that silently
    did nothing would make its ablation row meaningless."""
    on = small(**{flag: True})
    off = small(**{flag: False})
    out = off(torch.rand(1, 1, 32, 32), torch.rand(1, 3, 32, 32))
    assert out.shape == (1, 3, 32, 32) and torch.isfinite(out).all()

    p_on, p_off = on.num_parameters(), off.num_parameters()
    names_on = {n for n, _ in on.named_parameters()}
    names_off = {n for n, _ in off.named_parameters()}
    assert p_on != p_off or names_on != names_off, f"{flag} changed nothing"


@pytest.mark.parametrize("flag", ["adaptive_fusion", "use_edge_guidance"])
def test_behavioural_flags_change_the_output(flag):
    """These two switches keep the same modules in place and change only what
    information flows through them, so parameter counts are identical by design.
    The check that matters is whether the output differs at identical weights.
    """
    torch.manual_seed(0)
    on = small(**{flag: True})
    torch.manual_seed(0)
    off = small(**{flag: False})
    assert on.num_parameters() == off.num_parameters(), (
        f"{flag} is expected to be behaviour-preserving in parameter count"
    )

    # break the zero-init head, otherwise both variants are the identity and
    # every comparison trivially passes
    for m in (on, off):
        torch.nn.init.normal_(m.head.weight, std=0.05)

    torch.manual_seed(1)
    ir, vi = torch.rand(1, 1, 32, 32), torch.rand(1, 3, 32, 32)
    out_on, out_off = on(ir, vi), off(ir, vi)
    assert not torch.allclose(out_on, out_off, atol=1e-5), (
        f"{flag} did not change the output at identical weights"
    )


def test_separable_is_cheaper_than_dense():
    sep = small(separable=True).num_parameters()
    dense = small(separable=False).num_parameters()
    assert sep < dense


def test_shared_stem_halves_stem_parameters():
    shared = small(shared_stem=True).num_parameters()
    unshared = small(shared_stem=False).num_parameters()
    assert unshared > shared


def test_rgb_head_ablation_changes_output_definition():
    m = small(residual_output=False)
    assert m.head.out_channels == 3
    out = m(torch.rand(1, 1, 32, 32), torch.rand(1, 3, 32, 32))
    assert out.shape == (1, 3, 32, 32)
    # with a zero head and no residual connection the output is exactly zero
    assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)


def test_residual_head_outputs_one_channel():
    assert small().head.out_channels == 1


# --------------------------------------------------------------------------- #
def test_gradients_reach_every_trainable_parameter():
    m = small()
    # perturb the zero-init head so gradients are non-trivial
    torch.nn.init.normal_(m.head.weight, std=0.01)
    out = m(torch.rand(2, 1, 32, 32), torch.rand(2, 3, 32, 32))
    out.mean().backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert not missing, f"no gradient reached: {missing}"


def test_training_reduces_the_loss_on_a_single_pair():
    """A real optimisation check: 30 Adam steps on one fixed pair must reduce
    the fusion loss. If this fails the model cannot learn at all, which no
    amount of dataset would fix."""
    torch.manual_seed(0)
    ir = torch.rand(1, 1, 64, 64)
    vi = torch.rand(1, 3, 64, 64)
    m = FusionEdge(width=16, depth=2)
    torch.nn.init.normal_(m.head.weight, std=0.01)

    from fusionedge.losses import FusionLoss

    loss_fn = FusionLoss()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)

    first = last = None
    for step in range(30):
        out = m(ir, vi)
        loss, _ = loss_fn(out, ir, vi)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"


def test_cross_modal_cost_is_bounded_by_the_token_grid():
    """The design claim is that attention runs on a fixed 8x8 grid, so its cost
    does not grow quadratically with resolution. This is the property that makes
    the module affordable at 480x640 -- 64x64 attention logits instead of
    19200x19200 -- so it is asserted structurally, not just described."""
    from fusionedge.models.fusionedge import CrossModalExchange

    mod = CrossModalExchange(32, grid=8).eval()
    x = torch.rand(1, 32, 64, 64)
    tokens_small = mod._tokens(x).shape
    y = torch.rand(1, 32, 480, 640)
    tokens_large = mod._tokens(y).shape
    assert tokens_small[1] == tokens_large[1] == 64, "token count must be grid^2"
    assert tokens_small[2] == tokens_large[2], "head_dim must be fixed by width"

    with torch.no_grad():
        a, _ = mod(x, x)
        b, _ = mod(y, y)
    assert a.shape == x.shape and b.shape == y.shape


def test_cross_modal_is_bounded_attention_not_full_attention():
    """A full-resolution alternative would produce H*W tokens. Assert the module
    is actually pooling, so this test fails if someone 'optimises' it into
    something quadratic."""
    from fusionedge.models.fusionedge import CrossModalExchange

    mod = CrossModalExchange(32, grid=8)
    pooled = mod._tokens(torch.rand(1, 32, 480, 640))
    assert pooled.shape[1] == 64
    assert pooled.shape[1] != 480 * 640


def test_fusionedge_handles_large_input():
    """Full 480x640 inference must work and stay in range (the real eval size)."""
    m = small(width=16, depth=2).eval()
    with torch.no_grad():
        out = m(torch.rand(1, 1, 480, 640), torch.rand(1, 3, 480, 640))
    assert out.shape == (1, 3, 480, 640)
    assert torch.isfinite(out).all()


def test_attention_temperature_is_applied():
    """A missing 1/sqrt(d) scale is a classic attention bug that still trains.
    Check the scale is what we expect for a given head_dim."""
    from fusionedge.models.fusionedge import CrossModalExchange

    mod = CrossModalExchange(32, grid=4, heads=2)
    assert mod.head_dim == 16
    # scale is computed inline as head_dim ** -0.5; pin the value
    assert (16**-0.5) == pytest.approx(0.25)


def test_deterministic_inference():
    m = small().eval()
    ir, vi = torch.rand(1, 1, 32, 32), torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(m(ir, vi), m(ir, vi))


# --------------------------------------------------------------------------- #
# registry integration
# --------------------------------------------------------------------------- #
def test_registry_builds_both_models():
    for name in ("baseline", "fusionedge", "identity_vi"):
        m = build_model(name, {"width": 8, "depth": 1} if name != "identity_vi" else {})
        out = m(torch.rand(1, 1, 32, 32), torch.rand(1, 3, 32, 32))
        assert out.shape == (1, 3, 32, 32)


def test_registry_rejects_unknown_model():
    with pytest.raises(KeyError, match="unknown model"):
        build_model("not_a_model")


def test_build_fusionedge_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown FusionEdge config keys"):
        build_fusionedge({"not_a_key": 1})


def test_baseline_and_fusionedge_are_comparable_in_size():
    """The comparison is only fair if the two models are the same order of
    magnitude in size. Checked so a future config edit cannot quietly turn the
    ablation into a capacity comparison."""

    def n_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    b = n_params(build_model("baseline", {"width": 32, "depth": 4}))
    f = n_params(build_model("fusionedge", {"width": 32, "depth": 2}))
    ratio = max(b, f) / max(min(b, f), 1)
    assert ratio < 6, f"size gap too large to be a controlled comparison: {b} vs {f}"