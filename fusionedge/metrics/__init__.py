"""Fusion quality metrics.

Headline metrics are EN, AG, MS-SSIM and YOLOv5x6 mAP. EN/AG/MS-SSIM are the
standard no-reference / full-reference pair used in the IR-VL fusion literature;
mAP is the downstream-task metric that says whether the fusion actually helped.

Every metric is implemented here in numpy/torch rather than imported from a
third-party package, and each is validated against an independent reference in
tests/test_metrics.py (EN and AG against literal re-implementations, SSIM and
MS-SSIM against scikit-image). The reason is worth stating in an interview: a
fusion paper's numbers are only meaningful if you can say exactly what the metric
computes, and "we called skimage" is not an answer.

Conventions
-----------
Images are uint8 HWC RGB, or float arrays in [0, 1]. Metrics that need a single
channel take luma with ITU-R BT.601 weights, except EN and AG which are defined
on the grayscale conversion that is itself part of the metric.
"""

from .metrics import (
    AG,
    EN,
    MI,
    SF,
    MS_SSIM,
    SSIM,
    AGGREGATE_METRICS,
    HIGHER_IS_BETTER,
    MetricResult,
    compute_all,
    compute_fusion_metrics,
    to_uint8,
    to_y_channel,
)

__all__ = [
    "EN",
    "AG",
    "SF",
    "MI",
    "SSIM",
    "MS_SSIM",
    "AGGREGATE_METRICS",
    "HIGHER_IS_BETTER",
    "MetricResult",
    "compute_all",
    "compute_fusion_metrics",
    "to_uint8",
    "to_y_channel",
]