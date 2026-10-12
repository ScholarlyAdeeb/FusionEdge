"""Model registry.

One place that maps a config string to a constructed model, so that a run's
config is enough to rebuild the exact architecture, and so an unknown name fails
immediately with the list of valid options instead of silently training nothing.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def build_model(name: str, cfg: dict[str, Any] | None = None) -> nn.Module:
    cfg = dict(cfg or {})
    name = name.lower()

    if name in ("baseline", "simplecnn", "simple_cnn"):
        from ..models.baseline import build_baseline

        return build_baseline(cfg)

    if name in ("fusionedge", "fusion_edge"):
        from ..models.fusionedge import build_fusionedge

        return build_fusionedge(cfg)

    if name in ("identity_vi", "passthrough"):
        return _VisiblePassthrough()

    raise KeyError(f"unknown model '{name}'. Available: {sorted(MODEL_NAMES)}")


MODEL_NAMES = ["baseline", "fusionedge", "identity_vi"]


class _VisiblePassthrough(nn.Module):
    """Returns the visible image unchanged.

    Not a fusion method. It is the sanity floor: a learned model that cannot
    beat "return the visible image" on a quality metric has a problem, and this
    makes that floor measurable instead of rhetorical.
    """

    def forward(self, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        return vi.clamp(0.0, 1.0)