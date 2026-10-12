"""Classical fusion baselines: contract, range and sanity.

These are the floor the learned models must beat, so their behaviour is pinned:
they must return uint8 RGB in a sensible range, and each must fail in the
specific way its name advertises (average blurs, max keeps the brighter pixel).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fusionedge.models.classical import (
    CLASSICAL_METHODS,
    average,
    laplacian_pyramid,
    max_abs,
    weighted_average,
)
from fusionedge.metrics import to_uint8, to_y_channel


@pytest.fixture
def pair():
    """An aligned pair with a hot spot that is dark in the visible image."""
    h = w = 48
    ir = np.full((h, w), 0.15, np.float64)
    ir[16:32, 16:32] = 0.95  # a thermal target invisible in VI
    yy, xx = np.mgrid[0:h, 0:w]
    vi = np.stack(
        [0.4 + 0.2 * np.sin(xx / 4), 0.4 + 0.2 * np.cos(yy / 4), 0.4 + 0.05 * (xx / w)],
        axis=-1,
    ).astype(np.float64)
    return ir, vi


@pytest.mark.parametrize("name", sorted(CLASSICAL_METHODS))
def test_all_methods_return_valid_uint8_rgb(pair, name):
    ir, vi = pair
    out = CLASSICAL_METHODS[name](ir, vi)
    assert out.dtype == np.uint8
    assert out.shape == vi.shape
    assert out.ndim == 3 and out.shape[2] == 3


@pytest.mark.parametrize("name", sorted(CLASSICAL_METHODS))
def test_all_methods_handle_grayscale_visible(pair, name):
    ir, vi = pair
    out = CLASSICAL_METHODS[name](ir, to_y_channel(vi))
    assert out.shape == (48, 48, 3) and out.dtype == np.uint8


@pytest.mark.parametrize("name", sorted(CLASSICAL_METHODS))
def test_all_methods_are_deterministic(pair, name):
    ir, vi = pair
    fn = CLASSICAL_METHODS[name]
    assert np.array_equal(fn(ir, vi), fn(ir, vi))


@pytest.mark.parametrize("name", sorted(CLASSICAL_METHODS))
def test_all_methods_handle_odd_sizes(name):
    rng = np.random.default_rng(0)
    ir = rng.random((37, 53))
    vi = rng.random((37, 53, 3))
    out = CLASSICAL_METHODS[name](ir, vi)
    assert out.shape == (37, 53, 3)


def test_average_is_the_midpoint_of_luma(pair):
    ir, vi = pair
    out = average(ir, vi)
    expect = 0.5 * (to_y_channel(ir) + to_y_channel(vi))
    got = to_y_channel(out)
    # luma is preserved only up to the [0,1] clamp, so compare loosely
    assert np.abs(got - expect).mean() < 0.08


def test_max_abs_recovers_the_invisible_thermal_target(pair):
    """The defining property: a hot IR target that is dark in VI must appear in
    the fused output."""
    ir, vi = pair
    out = to_y_channel(max_abs(ir, vi))
    assert out[20:28, 20:28].mean() > 0.7
    assert to_y_channel(vi)[20:28, 20:28].mean() < 0.6


def test_weighted_average_interpolates_between_endpoints(pair):
    ir, vi = pair
    a = to_y_channel(weighted_average(ir, vi, w=0.0))
    b = to_y_channel(weighted_average(ir, vi, w=1.0))
    m = to_y_channel(weighted_average(ir, vi, w=0.5))
    assert np.abs(m - 0.5 * (a + b)).mean() < 0.06
    assert b[20:28, 20:28].mean() > a[20:28, 20:28].mean()


def test_laplacian_pyramid_preserves_more_detail_than_average(pair):
    """Multi-scale fusion is supposed to keep more high-frequency energy than a
    flat average. This is the reason to prefer it as a reference method."""
    ir, vi = pair
    def hf_energy(img):
        y = to_y_channel(img).astype(np.float64)
        return np.abs(np.diff(y, axis=1)).mean()

    lap = hf_energy(laplacian_pyramid(ir, vi))
    avg = hf_energy(average(ir, vi))
    assert lap > avg


def test_laplacian_pyramid_handles_more_levels_than_the_image_allows():
    """Asking for a 6-level pyramid on a 32x32 image must not crash."""
    rng = np.random.default_rng(1)
    ir, vi = rng.random((32, 32)), rng.random((32, 32, 3))
    assert laplacian_pyramid(ir, vi, levels=6).shape == (32, 32, 3)