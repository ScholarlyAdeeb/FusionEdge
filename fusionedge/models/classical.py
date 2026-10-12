"""Classical (training-free) fusion baselines.

These exist for two reasons: they cost nothing, so they are the honest floor
that a learned model must beat; and their behaviour makes the failure modes of
learned models legible (average blurs, max-abs ghosts, Laplacian pyramid rings).

Each function returns a uint8 RGB image, matching what the learned models emit,
so all of them go through exactly the same metric path.
"""

from __future__ import annotations

import numpy as np

from ..metrics.metrics import to_float, to_uint8, to_y_channel


def _prep(ir: np.ndarray, vi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (ir_y, vi_rgb) as float [0,1], shaped for fusion."""
    ir_y = to_y_channel(ir)  # (H,W)
    vi_rgb = to_float(vi)
    if vi_rgb.ndim == 2:
        vi_rgb = np.repeat(vi_rgb[:, :, None], 3, axis=2)
    vi_rgb = vi_rgb[:, :, :3]
    return ir_y, vi_rgb


def average(ir: np.ndarray, vi: np.ndarray) -> np.ndarray:
    """F = (Y_ir + Y_vi)/2, chroma from visible."""
    ir_y, vi_rgb = _prep(ir, vi)
    y = 0.5 * (ir_y + to_y_channel(vi_rgb))
    # scale the visible chroma by the ratio of target luma to source luma
    src = np.maximum(to_y_channel(vi_rgb), 1e-4)
    out = vi_rgb * (y / src)[:, :, None]
    return to_uint8(np.clip(out, 0, 1))


def max_abs(ir: np.ndarray, vi: np.ndarray) -> np.ndarray:
    """F = max(Y_ir, Y_vi) with visible chroma -- the classic intensity-max rule."""
    ir_y, vi_rgb = _prep(ir, vi)
    y = np.maximum(ir_y, to_y_channel(vi_rgb))
    src = np.maximum(to_y_channel(vi_rgb), 1e-4)
    return to_uint8(np.clip(vi_rgb * (y / src)[:, :, None], 0, 1))


def weighted_average(ir: np.ndarray, vi: np.ndarray, w: float = 0.5) -> np.ndarray:
    """F = w*Y_ir + (1-w)*Y_vi with visible chroma. `w` is the IR share."""
    ir_y, vi_rgb = _prep(ir, vi)
    y = w * ir_y + (1.0 - w) * to_y_channel(vi_rgb)
    src = np.maximum(to_y_channel(vi_rgb), 1e-4)
    return to_uint8(np.clip(vi_rgb * (y / src)[:, :, None], 0, 1))


def laplacian_pyramid(
    ir: np.ndarray, vi: np.ndarray, levels: int = 4, sigma: float = 1.0
) -> np.ndarray:
    """Classic multi-scale Laplacian pyramid fusion (Burt & Adelson).

    Per level: Gaussian blur both inputs, keep the Laplacian residual, fuse
    residuals by max-abs, then collapse from the coarsest level up. Chroma is
    taken from the visible image unchanged.

    Implemented with numpy + scipy.ndimage so the baseline has no torch
    dependency and cannot accidentally benefit from anything the learned models
    have.
    """
    from scipy.ndimage import gaussian_filter

    ir_y, vi_rgb = _prep(ir, vi)
    a, b = ir_y.copy(), to_y_channel(vi_rgb)

    gauss_pyr_a: list[np.ndarray] = []
    gauss_pyr_b: list[np.ndarray] = []
    lap_a: list[np.ndarray] = []
    lap_b: list[np.ndarray] = []
    cur_a, cur_b = a, b
    for _ in range(levels):
        nxt_a = gaussian_filter(cur_a, sigma)
        nxt_b = gaussian_filter(cur_b, sigma)
        gauss_pyr_a.append(nxt_a)
        gauss_pyr_b.append(nxt_b)
        lap_a.append(cur_a - nxt_a)
        lap_b.append(cur_b - nxt_b)
        cur_a, cur_b = nxt_a, nxt_b

    fused_lap: list[np.ndarray] = []
    for la, lb in zip(lap_a, lap_b):
        fused_lap.append(np.where(np.abs(la) >= np.abs(lb), la, lb))

    y = fused_lap[-1] + cur_a
    for lvl in range(levels - 2, -1, -1):
        y = fused_lap[lvl] + gaussian_filter(y, sigma)

    src = np.maximum(to_y_channel(vi_rgb), 1e-4)
    return to_uint8(np.clip(vi_rgb * (y / src)[:, :, None], 0, 1))


CLASSICAL_METHODS = {
    "average": average,
    "max_abs": max_abs,
    "weighted_average": weighted_average,
    "laplacian_pyramid": laplacian_pyramid,
}