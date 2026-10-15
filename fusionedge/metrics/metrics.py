"""Metric implementations (numpy/torch, validated against reference code)."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# conversions
# --------------------------------------------------------------------------- #
_BT601 = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def to_float(x: np.ndarray) -> np.ndarray:
    """uint8 or float -> float64 in [0, 1]."""
    a = np.asarray(x)
    if a.dtype == np.uint8:
        return a.astype(np.float64) / 255.0
    return np.clip(a.astype(np.float64), 0.0, 1.0)


def to_uint8(x: np.ndarray) -> np.ndarray:
    """float in [0,1] (or uint8) -> uint8, with rounding rather than truncation."""
    a = np.asarray(x)
    if a.dtype == np.uint8:
        return a
    return np.clip(np.round(to_float(a) * 255.0), 0, 255).astype(np.uint8)


def to_y_channel(x: np.ndarray) -> np.ndarray:
    """RGB (or already-gray) -> luma in [0,1] using BT.601 weights."""
    a = to_float(x)
    if a.ndim == 2:
        return a
    if a.shape[2] >= 3:
        return a[..., :3] @ _BT601
    return a[..., 0]


# --------------------------------------------------------------------------- #
# EN -- entropy of the intensity histogram
# --------------------------------------------------------------------------- #
def EN(fused: np.ndarray, bins: int = 256, **_: Any) -> float:
    """Entropy of the fused image's intensity histogram (bits).

    Higher = the fused image uses more of the dynamic range, i.e. it preserved
    more distinct structure. It is easy to inflate (adding noise raises EN), so
    on its own it is a weak metric and is always reported alongside AG.
    """
    y = to_y_channel(fused)
    hist, _ = np.histogram(y, bins=bins, range=(0.0, 1.0), density=False)
    p = hist.astype(np.float64) / max(y.size, 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# --------------------------------------------------------------------------- #
# AG -- average gradient
# --------------------------------------------------------------------------- #
def AG(fused: np.ndarray, **_: Any) -> float:
    """Mean Sobel gradient magnitude of the luma channel (0..~1).

    This is the standard "sharpness" metric in fusion papers: it rewards clear
    edges, which is exactly the failure mode of over-smoothed fusion output.
    """
    y = to_y_channel(fused)
    gx = np.zeros_like(y)
    gy = np.zeros_like(y)
    gx[:, :-1] = y[:, 1:] - y[:, :-1]
    gy[:-1, :] = y[1:, :] - y[:-1, :]
    return float(np.mean(np.sqrt(gx**2 + gy**2)))


# --------------------------------------------------------------------------- #
# SF -- spatial frequency
# --------------------------------------------------------------------------- #
def SF(fused: np.ndarray, **_: Any) -> float:
    """Spatial frequency, in the standard four-direction formulation.

    Row/column frequencies are measured by first-difference energies; the total
    is sqrt(RF^2 + CF^2) with RF, CF themselves the diagonal combination of the
    horizontal and vertical components.
    """
    y = to_y_channel(fused)
    dx = np.diff(y, axis=1)
    dy = np.diff(y, axis=0)

    rf = np.sqrt(np.mean(dx**2) + np.mean(dy**2))
    cf = np.sqrt(np.mean(np.mean(dx**2, axis=1)) + np.mean(np.mean(dy**2, axis=0)))
    return float(np.sqrt(rf**2 + cf**2))


# --------------------------------------------------------------------------- #
# MI -- mutual information with the sources
# --------------------------------------------------------------------------- #
def _mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 256) -> float:
    """MI(a,b) in bits from a joint 256x256 histogram of the two luma channels."""
    ay, by = to_y_channel(a).ravel(), to_y_channel(b).ravel()
    idx_a = np.clip((ay * bins).astype(np.int64), 0, bins - 1)
    idx_b = np.clip((by * bins).astype(np.int64), 0, bins - 1)
    joint = np.bincount(idx_a * bins + idx_b, minlength=bins * bins).reshape(bins, bins)
    p = joint / joint.sum()
    pa, pb = p.sum(axis=1, keepdims=True), p.sum(axis=0, keepdims=True)
    nz = p > 0
    denom = (pa @ pb)[nz]
    return float((p[nz] * np.log2(p[nz] / denom)).sum())


def MI(fused: np.ndarray, sources: list[np.ndarray] | None = None, **kw: Any) -> float:
    """Sum of MI(fused, ir) + MI(fused, vi).

    Encodes "how much information from each source survives in the output". With
    a single source it degenerates to that pair's MI, which is why the fusion
    convention is to sum over both.
    """
    srcs = sources if sources is not None else []
    if not srcs:
        raise ValueError("MI requires the source images")
    return float(sum(_mutual_information(fused, s, kw.get("bins", 256)) for s in srcs))


# --------------------------------------------------------------------------- #
# SSIM / MS-SSIM (torch, differentiable-friendly)
# --------------------------------------------------------------------------- #
def _gaussian_window(window: int = 11, sigma: float = 1.5, device=None, dtype=None) -> torch.Tensor:
    coords = torch.arange(window, dtype=torch.float64 if dtype is None else dtype, device=device)
    coords = coords - (window - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None]


def _fit_window(win: int, h: int, w: int) -> int:
    """Largest odd window <= min(h, w), floor 3. SSIM cannot be computed with a
    window wider than the image; shrinking is better than an exception."""
    win = min(win, h, w)
    if win % 2 == 0:
        win -= 1
    return max(3, win)


def _ssim_maps(
    x: torch.Tensor, y: torch.Tensor, window: torch.Tensor, data_range: float = 1.0
) -> torch.Tensor:
    """Per-pixel SSIM map for (N,C,H,W), computed on the *valid* region.

    'valid' (no padding) is what the reference implementation does: it discards
    the border where the Gaussian window would hang off the image. Padding
    instead would change the mean and make our number disagree with the one a
    reader computes with scikit-image, which is the whole point of implementing
    it here rather than importing it.
    """
    n, c, h, w = x.shape
    groups = c

    def filt(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, window.to(t.dtype).expand(c, 1, -1, -1), groups=groups)

    mu_x, mu_y = filt(x), filt(y)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sx = filt(x * x) - mu_x2
    sy = filt(y * y) - mu_y2
    sxy = filt(x * y) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    return ((2 * mu_xy + c1) * (2 * sxy + c2)) / ((mu_x2 + mu_y2 + c1) * (sx + sy + c2))


def _prep_pair(a: np.ndarray, b: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    ta = torch.from_numpy(to_y_channel(a)[None, None]).float()
    tb = torch.from_numpy(to_y_channel(b)[None, None]).float()
    return ta, tb


def SSIM(a: np.ndarray, b: np.ndarray, **_: Any) -> float:
    """Mean SSIM between two images' luma channels (Wang et al. 2004).

    Validated against skimage.metrics.structural_similarity with matching
    settings (gaussian weights, sigma=1.5, population covariance) in
    tests/test_metrics.py -- agreement to <1e-6.
    """
    ta, tb = _prep_pair(a, b)
    win = _fit_window(11, ta.shape[-2], ta.shape[-1])
    return float(_ssim_maps(ta, tb, _gaussian_window(win, 1.5)).mean())


# MS-SSIM standard scale weights and contrast-sensitivity exponents
_MS_SSIM_WEIGHTS = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
_MS_SSIM_CS = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])


def _ssim_and_cs(
    x: torch.Tensor, y: torch.Tensor, window: torch.Tensor, data_range: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean SSIM and mean contrast-structure term on the valid region."""
    c = x.shape[1]

    def filt(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, window.to(t.dtype).expand(c, 1, -1, -1), groups=c)

    mu_x, mu_y = filt(x), filt(y)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sx = filt(x * x) - mu_x2
    sy = filt(y * y) - mu_y2
    sxy = filt(x * y) - mu_xy
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    cs_map = (2 * sxy + c2) / (sx + sy + c2)
    ssim_map = ((2 * mu_xy + c1) / (mu_x2 + mu_y2 + c1)) * cs_map
    return ssim_map.mean(), cs_map.mean()


def MS_SSIM(
    a: np.ndarray,
    b: np.ndarray,
    levels: int = 5,
    win_size: int = 11,
    **_: Any,
) -> float:
    """Multi-scale SSIM (Wang, Simoncelli & Bovik 2003), standard definition:

        MS-SSIM = l_M^{w_M} * prod_{j=1..M} cs_j^{w_j}

    i.e. the contrast-structure term at every scale and the luminance term only
    at the coarsest, combined as a weighted *product*. Negative cs values are
    clamped to 0 (as in the widely used pytorch-msssim) so the result stays in
    [0, 1]. The pyramid depth is reduced when the image is too small, and the
    weights are renormalised to sum to 1 so identical images still score 1.

    Corrected 2026-10-13: the previous version used the full SSIM at every level
    and returned the *mean* of per-level products, which is not MS-SSIM.
    """
    ta, tb = _prep_pair(a, b)
    ta, tb = ta.double(), tb.double()

    usable = levels
    while usable > 1 and min(ta.shape[-2], ta.shape[-1]) < win_size * (2 ** (usable - 1)):
        usable -= 1
    w = torch.tensor(_MS_SSIM_WEIGHTS[:usable], dtype=torch.float64)
    w = w / w.sum()

    cs_vals = []
    ssim_last = None
    for i in range(usable):
        win = _fit_window(win_size, ta.shape[-2], ta.shape[-1])
        s_mean, cs_mean = _ssim_and_cs(ta, tb, _gaussian_window(win, 1.5, dtype=torch.float64))
        cs_vals.append(cs_mean.clamp_min(0.0))
        ssim_last = s_mean.clamp_min(0.0)
        if i < usable - 1:
            ta = F.avg_pool2d(ta, kernel_size=2, stride=2)
            tb = F.avg_pool2d(tb, kernel_size=2, stride=2)

    cs = torch.stack(cs_vals)
    # coarsest scale contributes full SSIM (= l_M * cs_M); finer scales cs only
    val = torch.prod(cs[:-1] ** w[:-1]) * (ssim_last ** w[-1])
    return float(val.clamp(0.0, 1.0))


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
@dataclass
class MetricResult:
    """All metrics for one fused image. Every field is a plain float so the
    result can be serialised straight into metrics.json."""

    EN: float = 0.0
    AG: float = 0.0
    SF: float = 0.0
    MI: float = 0.0
    SSIM_ir: float = 0.0
    SSIM_vi: float = 0.0
    MS_SSIM_ir: float = 0.0
    MS_SSIM_vi: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_all(fused: np.ndarray, ir: np.ndarray, vi: np.ndarray) -> MetricResult:
    """Compute every no-GT metric for a single fused image."""
    return MetricResult(
        EN=EN(fused),
        AG=AG(fused),
        SF=SF(fused),
        MI=MI(fused, [ir, vi]),
        SSIM_ir=SSIM(fused, ir),
        SSIM_vi=SSIM(fused, vi),
        MS_SSIM_ir=MS_SSIM(fused, ir),
        MS_SSIM_vi=MS_SSIM(fused, vi),
    )


#: Metrics where higher is better. Stated explicitly so no sign error can creep
#: into a results table.
HIGHER_IS_BETTER = frozenset({"EN", "AG", "SF", "MI", "SSIM_ir", "SSIM_vi", "MS_SSIM_ir", "MS_SSIM_vi"})

AGGREGATE_METRICS: dict[str, Callable[..., float]] = {
    "EN": EN,
    "AG": AG,
    "SF": SF,
    "MS_SSIM_ir": lambda f, ir, vi, **kw: MS_SSIM(f, ir),
    "MS_SSIM_vi": lambda f, ir, vi, **kw: MS_SSIM(f, vi),
}


def compute_fusion_metrics(
    fused: np.ndarray,
    ir: np.ndarray,
    vi: np.ndarray,
    *,
    extra: Callable[[np.ndarray, np.ndarray, np.ndarray], dict[str, float]] | None = None,
) -> MetricResult:
    """Metric entry point used by the evaluation engine.

    `extra` is where the downstream-task metric (YOLOv5x6 mAP) is injected, so
    that the expensive detector never runs inside the per-image loop.
    """
    res = compute_all(fused, ir, vi)
    if extra is not None:
        for k, v in extra(fused, ir, vi).items():
            if hasattr(res, k):
                setattr(res, k, float(v))
    return res