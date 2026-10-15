"""Regression tests for the fixes made during the 2026-10-13 recovery.

Each test fails on the pre-fix code:
  * cross-attention routed the wrong values and scrambled heads,
  * MS-SSIM was not the standard product-of-scales definition,
  * the baseline head had a LeakyReLU in front of the sigmoid,
  * the model-selection holdout must be deterministic and disjoint,
  * checkpoints must carry the LR-scheduler state for a faithful resume.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from fusionedge.data.fusion_dataset import holdout_manifest, split_holdout
from fusionedge.metrics.metrics import MS_SSIM, _gaussian_window, _prep_pair, _ssim_and_cs
from fusionedge.models.baseline import SimpleCNNBaseline
from fusionedge.models.fusionedge import CrossModalExchange
from fusionedge.utils.checkpoint import load_checkpoint, save_checkpoint


def _identity_exchange(c: int = 8, grid: int = 4, heads: int = 2) -> CrossModalExchange:
    """q=k=0 (uniform attention), v=proj=identity: output = x + mean of y tokens."""
    m = CrossModalExchange(c, grid=grid, heads=heads)
    with torch.no_grad():
        for conv in (m.q, m.k):
            conv.weight.zero_(); conv.bias.zero_()
        for conv in (m.v, m.proj):
            conv.weight.copy_(torch.eye(c).view(c, c, 1, 1)); conv.bias.zero_()
    return m.eval()


def test_cross_attention_x_branch_receives_y_values():
    m = _identity_exchange()
    x = torch.zeros(1, 8, 16, 16)
    y = torch.rand(1, 8, 16, 16)
    with torch.no_grad():
        x_out, _ = m(x, y)
    # uniform attention over y's tokens -> per-channel mean of y, broadcast
    expect = y.mean(dim=(2, 3), keepdim=True).expand_as(x)
    assert torch.allclose(x_out, expect, atol=1e-5), "x must aggregate y's values"


def test_cross_attention_keeps_channels_separate_across_heads():
    m = _identity_exchange(c=8, heads=2)
    y = torch.zeros(1, 8, 16, 16)
    y[:, 3] = 1.0  # only channel 3 carries signal
    with torch.no_grad():
        x_out, _ = m(torch.zeros_like(y), y)
    per_channel = x_out.mean(dim=(2, 3))[0]
    assert per_channel[3] == pytest.approx(1.0, abs=1e-5)
    others = torch.cat([per_channel[:3], per_channel[4:]])
    assert torch.allclose(others, torch.zeros_like(others), atol=1e-5)


def _img(seed, h=128, w=128):
    return np.random.default_rng(seed).random((h, w, 3))


def test_ms_ssim_is_the_standard_product_of_scales():
    a = _img(1)
    b = np.clip(a + 0.1 * _img(2), 0, 1)
    ta, tb = _prep_pair(a, b)
    ta, tb = ta.double(), tb.double()
    levels = 4  # 128 >= 11*2**3 = 88 but < 11*2**4 = 176
    w = np.array([0.0448, 0.2856, 0.3001, 0.2363])
    w = w / w.sum()
    css, last = [], None
    for i in range(levels):
        s, cs = _ssim_and_cs(ta, tb, _gaussian_window(11, 1.5, dtype=torch.float64))
        css.append(max(float(cs), 0.0)); last = max(float(s), 0.0)
        ta = torch.nn.functional.avg_pool2d(ta, 2); tb = torch.nn.functional.avg_pool2d(tb, 2)
    expect = np.prod(np.array(css[:-1]) ** w[:-1]) * last ** w[-1]
    assert MS_SSIM(a, b) == pytest.approx(expect, rel=1e-9)


def test_baseline_head_has_no_activation_before_sigmoid():
    m = SimpleCNNBaseline()
    assert isinstance(m.act, nn.Identity)
    legacy = SimpleCNNBaseline(use_relu_head=True)
    assert isinstance(legacy.act, nn.LeakyReLU)  # old behaviour still reproducible


def test_holdout_is_deterministic_disjoint_and_order_independent():
    ids = [f"{i:05d}D.png" for i in range(1083)]
    fit, hold = split_holdout(ids, 0.1, seed=0)
    fit2, hold2 = split_holdout(list(reversed(ids)), 0.1, seed=0)
    assert (fit, hold) == (fit2, hold2)
    assert len(hold) == 108 and len(fit) == 975
    assert not set(fit) & set(hold) and set(fit) | set(hold) == set(ids)
    _, hold3 = split_holdout(ids, 0.1, seed=1)
    assert hold3 != hold
    man = holdout_manifest(fit, hold, 0.1, 0, "train")
    assert man["n_fit"] == 975 and len(man["holdout_sha256"]) == 64


def test_checkpoint_round_trips_scheduler_state(tmp_path):
    model = nn.Linear(2, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-6)
    for _ in range(37):
        opt.step(); sch.step()
    save_checkpoint(tmp_path / "c.pt", model, optimizer=opt, scheduler=sch, best_epoch=4)

    model2 = nn.Linear(2, 2)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=100, eta_min=1e-6)
    payload = load_checkpoint(tmp_path / "c.pt", model2, optimizer=opt2, scheduler=sch2,
                              restore_rng=True)
    assert sch2.last_epoch == 37
    assert opt2.param_groups[0]["lr"] == pytest.approx(opt.param_groups[0]["lr"])
    assert payload["best_epoch"] == 4


# --------------------------------------------------------------------------- #
# luminance-only output (2026-10-14)
# --------------------------------------------------------------------------- #
from fusionedge.models.colour import recolour, rgb_to_ycbcr, ycbcr_to_rgb  # noqa: E402
from fusionedge.models.fusionedge import FusionEdge  # noqa: E402


def test_ycbcr_round_trip():
    x = torch.rand(2, 3, 16, 16)
    ycc = rgb_to_ycbcr(x)
    back = ycbcr_to_rgb(ycc[:, :1], ycc[:, 1:2], ycc[:, 2:3])
    assert torch.allclose(back, x, atol=1e-5)


def test_residual_head_is_already_luminance_only_fusion():
    """vi + delta (FusionEdge's residual output) == recolour(Y_vi + delta, vi)."""
    vi = 0.2 + 0.6 * torch.rand(1, 3, 16, 16)  # interior, so no clamping
    delta = 0.05 * torch.randn(1, 1, 16, 16)
    y = rgb_to_ycbcr(vi)[:, :1]
    assert torch.allclose(vi + delta, recolour(y + delta, vi), atol=1e-5)


def test_y_output_baseline_keeps_visible_chroma():
    m = SimpleCNNBaseline(width=8, depth=1, output_space="y").eval()
    assert m.head.out_channels == 1
    vi = 0.45 + 0.1 * torch.rand(1, 3, 32, 32)  # low chroma, so clamping is rare
    with torch.no_grad():
        out = m(torch.rand(1, 1, 32, 32), vi)
    # compare chroma only where no channel was clamped
    ok = ((out > 1e-4) & (out < 1 - 1e-4)).all(dim=1, keepdim=True)
    assert ok.float().mean() > 0.5
    d = (rgb_to_ycbcr(out)[:, 1:] - rgb_to_ycbcr(vi)[:, 1:]).abs()
    assert d[ok.expand_as(d)].max() < 1e-5


def test_direct_y_ablation_head():
    m = FusionEdge(width=8, depth=1, dilations=(1,), residual_output=False, output_space="y").eval()
    assert m.head.out_channels == 1
    with torch.no_grad():
        out = m(torch.rand(1, 1, 32, 32), torch.rand(1, 3, 32, 32))
    assert out.shape == (1, 3, 32, 32) and torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# loss precision (2026-10-14): the loss must not run under autocast
# --------------------------------------------------------------------------- #
def test_fp32_loss_is_exact_fp32_even_inside_autocast():
    from fusionedge.losses import FusionLoss
    from fusionedge.losses.fusion_loss import fp32_loss

    g = torch.Generator().manual_seed(0)
    ir, vi = torch.rand(2, 1, 64, 64, generator=g), torch.rand(2, 3, 64, 64, generator=g)
    f = (0.5 * vi + 0.5 * ir).requires_grad_(True)
    L = FusionLoss({})
    ref, _ = L(f, ir, vi)
    ref_grad, = torch.autograd.grad(ref, f)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got, _ = fp32_loss(L, f, ir, vi)
    got_grad, = torch.autograd.grad(got, f)
    assert got.dtype == torch.float32
    assert torch.equal(got, ref) and torch.equal(got_grad, ref_grad)


def test_trainer_computes_loss_outside_autocast():
    """Structural guard: the training step must call fp32_loss, and must not call
    the loss inside the autocast block (the v3-v5 bug)."""
    import inspect
    from fusionedge.engine import train as T

    src = inspect.getsource(T.Trainer.train_epoch)
    assert "fp32_loss(self.loss_fn" in src
    block = src.split('with torch.autocast("cuda", dtype=self.amp_dtype):')[1].split("\n")
    inside = [ln for ln in block[1:] if ln.startswith(" " * 20)]
    assert not any("loss_fn" in ln for ln in inside), "loss called inside autocast"


# --------------------------------------------------------------------------- #
# IR-injection head (residual_mode='inject')
def _inject(**kw):
    from fusionedge.models.fusionedge import FusionEdge
    return FusionEdge(width=8, depth=1, dilations=(1,), residual_mode="inject", **kw)


def test_inject_head_starts_halfway_to_ir():
    from fusionedge.models.colour import rgb_to_ycbcr
    torch.manual_seed(0)
    m = _inject().eval()
    vi = 0.3 + 0.2 * torch.rand(1, 3, 16, 16)
    ir = torch.rand(1, 1, 16, 16)
    y_vi = rgb_to_ycbcr(vi)[:, :1]
    with torch.no_grad():
        y = rgb_to_ycbcr(m(ir, vi))[:, :1]
    assert torch.allclose(y, y_vi + 0.5 * torch.relu(ir - y_vi), atol=1e-5)


def test_inject_head_only_brightens_towards_ir_and_keeps_chroma():
    from fusionedge.models.colour import rgb_to_ycbcr
    torch.manual_seed(0)
    m = _inject().eval()
    with torch.no_grad():
        m.head.weight.normal_(0, 1.0)  # arbitrary gate values
    vi = 0.45 + 0.1 * torch.rand(2, 3, 32, 32)  # low chroma: no clamping
    ir = torch.rand(2, 1, 32, 32)
    with torch.no_grad():
        out = m(ir, vi)
    yo, yv = rgb_to_ycbcr(out), rgb_to_ycbcr(vi)
    assert (yo[:, :1] >= yv[:, :1] - 1e-5).all()
    assert (yo[:, :1] <= torch.max(yv[:, :1], ir) + 1e-5).all()
    assert (yo[:, 1:] - yv[:, 1:]).abs().max() < 1e-5


def test_inject_head_gets_gradient_at_init():
    """The 'add' head's trap: identity sits on L1 kinks. 'inject' must get a usable signal."""
    torch.manual_seed(0)
    m = _inject().train()
    ir, vi = torch.rand(2, 1, 32, 32), torch.rand(2, 3, 32, 32)
    m(ir, vi).mean().backward()
    assert m.head.weight.grad.abs().sum() > 0


def test_residual_mode_is_validated():
    from fusionedge.models.fusionedge import FusionEdge
    with pytest.raises(ValueError):
        FusionEdge(residual_mode="bogus")


# --------------------------------------------------------------------------- #
# learning check (replaced the init-dependent "loss fell >=X% since epoch 0" rule)
def _lc_cfg():
    return {"seed": 1234, "loss": {}, "model": {"name": "fusionedge",
            "cfg": {"width": 8, "depth": 1, "dilations": [1]}}}


def test_learning_check_fails_an_untrained_model():
    from fusionedge.engine.learning_check import fresh_model, learning_check
    torch.manual_seed(0)
    pairs = [(torch.rand(1, 1, 32, 32), torch.rand(1, 3, 32, 32)) for _ in range(2)]
    cfg = _lc_cfg()
    r = learning_check(fresh_model(cfg, 1234, torch.device("cpu")), cfg, pairs, torch.device("cpu"))
    assert r["rel_drop"] == 0.0 and not r["pass"]


def test_learning_check_passes_when_loss_falls():
    from fusionedge.engine.learning_check import fresh_model, learning_check
    torch.manual_seed(0)
    ir = torch.rand(1, 1, 32, 32)
    vi = 0.2 * torch.rand(1, 3, 32, 32)  # IR mostly brighter: opening the gate lowers the loss
    cfg = _lc_cfg()
    m = fresh_model(cfg, 1234, torch.device("cpu"))
    with torch.no_grad():
        m.head.bias.fill_(6.0)  # gate ~1: full IR injection
    r = learning_check(m, cfg, [(ir, vi)], torch.device("cpu"))
    assert r["pass"] and r["rel_drop"] > 0.01


# --------------------------------------------------------------------------- #
# baseline backbone + FusionEdge's inject head (output_space='inject')
def test_baseline_inject_has_same_params_as_y_head():
    a = SimpleCNNBaseline(width=8, depth=2, output_space="y")
    b = SimpleCNNBaseline(width=8, depth=2, output_space="inject")
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


def test_baseline_inject_starts_at_the_same_rule_as_fusionedge():
    """Both zero-initialised: g = 0.5 everywhere, whatever the backbone weights."""
    from fusionedge.models.fusionedge import FusionEdge
    torch.manual_seed(0)
    ir, vi = torch.rand(2, 1, 32, 32), torch.rand(2, 3, 32, 32)
    b = SimpleCNNBaseline(width=8, depth=2, output_space="inject").eval()
    f = FusionEdge(width=8, depth=1, dilations=(1,), residual_mode="inject").eval()
    with torch.no_grad():
        assert torch.allclose(b(ir, vi), f(ir, vi), atol=1e-6)


def test_baseline_inject_only_brightens_and_keeps_chroma():
    from fusionedge.models.colour import rgb_to_ycbcr
    torch.manual_seed(0)
    m = SimpleCNNBaseline(width=8, depth=1, output_space="inject").eval()
    with torch.no_grad():
        m.head.weight.normal_(0, 1.0)
        vi = 0.45 + 0.1 * torch.rand(2, 3, 32, 32)
        ir = torch.rand(2, 1, 32, 32)
        yo, yv = rgb_to_ycbcr(m(ir, vi)), rgb_to_ycbcr(vi)
    assert (yo[:, :1] >= yv[:, :1] - 1e-5).all()
    assert (yo[:, :1] <= torch.max(yv[:, :1], ir) + 1e-5).all()
    assert (yo[:, 1:] - yv[:, 1:]).abs().max() < 1e-5


# --------------------------------------------------------------------------- #
# loader_seed: the train loader's generator drives shuffle AND worker aug seeds
def test_loader_seed_drives_the_train_generator(monkeypatch):
    import fusionedge.data.fusion_dataset as fd
    seen = []
    real = torch.utils.data.DataLoader

    def spy(*a, **kw):
        if kw.get("shuffle"):
            seen.append(kw["generator"].initial_seed())
        return real(*a, **kw)

    monkeypatch.setattr(fd, "DataLoader", spy)
    for s in (0, 1235):
        try:
            fd.build_dataloaders("MSRS", num_workers=0, limit_train=8, limit_eval=2,
                                 holdout_frac=0.1, loader_seed=s)
        except Exception as exc:  # dataset not present on this machine
            pytest.skip(f"MSRS unavailable: {exc}")
    assert seen == [0, 1235]
