"""Metric correctness: validated against independent references.

The point of these tests is that a number in results/ can be defended. If EN
disagreed with the textbook formula, or SSIM disagreed with scikit-image, no
comparison table would mean anything.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from skimage.metrics import structural_similarity as sk_ssim

from fusionedge.metrics import AG, EN, MI, MS_SSIM, SF, SSIM, compute_all, to_uint8, to_y_channel
from fusionedge.metrics.metrics import _mutual_information


def _rand_img(seed: int, h: int = 64, w: int = 64, channels: int = 3) -> np.ndarray:
    g = torch.Generator().manual_seed(seed)
    a = torch.rand(h, w, channels, generator=g)
    if channels == 3:
        # add some structure so gradients are not pure noise
        a[..., 0] = torch.linspace(0, 1, w)[None, :].repeat(h, 1) * 0.5 + a[..., 0] * 0.5
        a[..., 1] = torch.linspace(0, 1, h)[:, None].repeat(1, w) * 0.4 + a[..., 1] * 0.6
    return a.numpy().astype(np.float64)


# --------------------------------------------------------------------------- #
def test_to_y_channel_weights():
    """BT.601: pure red -> 0.299, pure green -> 0.587, pure blue -> 0.114."""
    red = np.zeros((1, 1, 3)); red[..., 0] = 1.0
    green = np.zeros((1, 1, 3)); green[..., 1] = 1.0
    blue = np.zeros((1, 1, 3)); blue[..., 2] = 1.0
    assert to_y_channel(red)[0, 0] == pytest.approx(0.299)
    assert to_y_channel(green)[0, 0] == pytest.approx(0.587)
    assert to_y_channel(blue)[0, 0] == pytest.approx(0.114)


def test_to_uint8_roundtrip_is_lossless():
    img = _rand_img(3, 32, 32)
    u8 = to_uint8(img)
    assert u8.dtype == np.uint8
    assert np.abs(to_float_of(u8) - img).max() < 1.0 / 255.0 + 1e-9


def to_float_of(u8: np.ndarray) -> np.ndarray:
    return u8.astype(np.float64) / 255.0


# --------------------------------------------------------------------------- #
def test_en_matches_textbook_entropy():
    """Recompute EN from the histogram directly and compare."""
    img = _rand_img(7, 48, 64)
    got = EN(img)
    y = to_y_channel(img)
    hist, _ = np.histogram(y, bins=256, range=(0.0, 1.0))
    p = hist / hist.sum()
    p = p[p > 0]
    expect = -(p * np.log2(p)).sum()
    assert got == pytest.approx(expect, rel=1e-12)


def test_en_maximal_for_uniform_histogram():
    """A perfectly uniform 256-bin histogram has entropy exactly 8 bits."""
    y = (np.arange(256) / 255.0).reshape(16, 16)
    img = np.repeat(y[:, :, None], 3, axis=2)
    assert EN(img) == pytest.approx(8.0, abs=1e-9)


def test_en_constant_image_is_zero():
    assert EN(np.full((8, 8, 3), 0.5)) == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
def test_ag_matches_explicit_definition():
    """AG is the mean of sqrt(dx^2 + dy^2) on first differences."""
    img = _rand_img(11, 32, 32)
    y = to_y_channel(img)
    gx = np.zeros_like(y); gx[:, :-1] = y[:, 1:] - y[:, :-1]
    gy = np.zeros_like(y); gy[:-1, :] = y[1:, :] - y[:-1, :]
    assert AG(img) == pytest.approx(np.mean(np.sqrt(gx**2 + gy**2)), rel=1e-12)


def test_ag_zero_for_constant_image():
    assert AG(np.full((16, 16, 3), 0.3)) == pytest.approx(0.0, abs=1e-12)


def test_ag_higher_for_sharper_image():
    sharp = _rand_img(5, 64, 64)
    blurred = np.stack(
        [
            np.add.reduceat(sharp[..., c], np.arange(0, 64, 4), axis=1)[:, :64]
            if False
            else sharp[..., c]
            for c in range(3)
        ],
        axis=-1,
    )
    # construct a genuinely blurred image instead of the no-op above
    blurred = np.copy(sharp)
    for c in range(3):
        blurred[..., c] = 0.25 * (
            np.roll(sharp[..., c], 1, 0) + sharp[..., c] + np.roll(sharp[..., c], -1, 0)
        )
    assert AG(blurred) < AG(sharp)


# --------------------------------------------------------------------------- #
def test_sf_nonnegative_and_scales_with_frequency():
    flat = np.full((32, 32, 3), 0.5)
    checker = np.indices((32, 32)).sum(axis=0) % 2
    hi = np.repeat(checker[:, :, None].astype(np.float64), 3, axis=2)
    assert SF(flat) == pytest.approx(0.0, abs=1e-12)
    assert SF(hi) > 0.5


# --------------------------------------------------------------------------- #
def test_mi_properties():
    a = _rand_img(21, 32, 32)
    b = _rand_img(22, 32, 32)
    # MI(x;x) equals the entropy of x's own histogram -- it is *not* necessarily
    # 8 bits, because a finite image may not populate all 256 bins.
    assert _mutual_information(a, a) == pytest.approx(EN(a), abs=1e-9)
    assert _mutual_information(a, b) >= 0.0
    assert MI(a, [a, a]) == pytest.approx(2 * _mutual_information(a, a), rel=1e-9)


def test_mi_is_a_sum_over_sources():
    """MI(fused; [ir, vi]) == MI(fused; ir) + MI(fused; vi) -- it is a sum of
    independent terms, not MI(fused; ir, vi). This distinction is easy to get
    wrong when reading fusion papers, so it is pinned here."""
    ir = _rand_img(23, 32, 32, channels=1)
    vi = _rand_img(24, 32, 32)
    fused = _rand_img(25, 32, 32)
    assert MI(fused, [ir, vi]) == pytest.approx(
        _mutual_information(fused, ir) + _mutual_information(fused, vi), rel=1e-9
    )


def test_mi_prefers_the_more_informative_source():
    """An output that is a copy of source A carries all of A's information, so
    the MI sum must be at least H(A) -- no fusion can do better than keeping
    one source intact."""
    ir = _rand_img(26, 48, 48, channels=1)
    vi = _rand_img(27, 48, 48)
    mi_if_fused_is_ir = MI(ir, [ir, vi])
    mi_if_fused_is_vi = MI(vi, [ir, vi])
    assert mi_if_fused_is_ir >= EN(ir) - 1e-9
    assert mi_if_fused_is_vi >= EN(vi) - 1e-9


# --------------------------------------------------------------------------- #
def test_ssim_matches_skimage():
    """Our SSIM must agree with scikit-image's reference implementation."""
    for seed in (31, 32, 33):
        a = _rand_img(seed, 64, 64)
        b = np.clip(a + 0.15 * _rand_img(seed + 100, 64, 64), 0, 1)
        ours = SSIM(a, b)
        theirs = sk_ssim(
            to_y_channel(a),
            to_y_channel(b),
            data_range=1.0,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
        assert ours == pytest.approx(theirs, abs=1e-6), f"seed {seed}: {ours} vs {theirs}"


def test_ssim_identity_is_one():
    a = _rand_img(41, 48, 48)
    assert SSIM(a, a) == pytest.approx(1.0, abs=1e-6)
    assert MS_SSIM(a, a) == pytest.approx(1.0, abs=1e-6)


def test_ms_ssim_bounds_and_monotonicity():
    a = _rand_img(51, 96, 96)
    prev = 2.0
    for eps in (0.0, 0.05, 0.15, 0.35):
        b = np.clip(a + eps * _rand_img(52, 96, 96), 0, 1)
        v = MS_SSIM(a, b)
        assert 0.0 <= v <= 1.0
        if eps == 0.0:
            assert v == pytest.approx(1.0, abs=1e-6)
        else:
            assert v < prev, f"MS-SSIM should decrease as distortion grows (eps={eps})"
            prev = v


def test_ms_ssim_is_symmetric():
    a = _rand_img(61, 64, 64)
    b = _rand_img(62, 64, 64)
    assert MS_SSIM(a, b) == pytest.approx(MS_SSIM(b, a), abs=1e-9)


# --------------------------------------------------------------------------- #
def test_compute_all_keys_and_types():
    fused, ir, vi = _rand_img(71), _rand_img(72, channels=1), _rand_img(73)
    res = compute_all(fused, ir, vi)
    d = res.as_dict()
    assert set(d) == {
        "EN", "AG", "SF", "MI", "SSIM_ir", "SSIM_vi", "MS_SSIM_ir", "MS_SSIM_vi",
    }
    assert all(isinstance(v, float) and np.isfinite(v) for v in d.values())


def test_identical_sources_give_identity_structure_metrics():
    """Fusing a perfect image against itself should not be penalised structurally."""
    a = _rand_img(81, 64, 64)
    res = compute_all(a, a, a)
    assert res.SSIM_ir == pytest.approx(1.0, abs=1e-6)
    assert res.MS_SSIM_ir == pytest.approx(1.0, abs=1e-6)
    assert res.MI == pytest.approx(2 * EN(a), rel=1e-9)  # 2 x 8 bits at most