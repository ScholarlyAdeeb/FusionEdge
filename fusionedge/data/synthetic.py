"""Synthetic IR/VI pairs -- UNIT TESTS ONLY.

This module exists so shapes, dtypes, ranges, loss finiteness and metric
correctness can be tested on a machine with no dataset. It is fenced off from
the training and evaluation entry points on purpose:

  * `SyntheticPairDataset` requires `allow_synthetic=True` (default False).
  * It marks every sample with `"synthetic": True`.
  * `fusionedge.data.build_dataloaders` cannot return it.
  * `tests/test_no_fabricated_results.py` fails if a run directory produced from
    synthetic data is ever read as a real result.

The synthetic scenes are built from parametric primitives (rectangles, discs,
gradients, stripes, a sinusoidal "thermal blob") with the *same* structural
property real pairs have: IR emphasises hot blobs, VI carries fine texture, and
the two are pixel-aligned.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticDatasetNotPermitted(RuntimeError):
    """Raised when synthetic data is requested without an explicit opt-in."""


def make_pair(
    seed: int, height: int = 128, width: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate one aligned (ir, vi) pair as float tensors in [0, 1]."""
    rng = np.random.default_rng(seed)

    # --- visible: smooth background + fine texture -------------------------
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    base = 0.45 + 0.2 * np.sin(2 * np.pi * (xx / max(width, 1) * rng.uniform(1, 3)))
    base += 0.15 * np.cos(2 * np.pi * (yy / max(height, 1) * rng.uniform(1, 3)))
    tex = rng.normal(0, 0.06, size=(height, width)).astype(np.float32)
    vi = base[..., None] + tex[..., None] * np.array([1.0, 0.9, 0.8], np.float32)
    vi = vi + np.stack(
        [
            0.10 * ((xx % 17) < 8),  # stripes -> high frequency edges
            0.08 * ((yy % 23) < 11),
            0.06 * (((xx + yy) % 29) < 14),
        ],
        axis=-1,
    ).astype(np.float32)

    # --- infrared: hot blobs, no texture -----------------------------------
    ir = np.full((height, width), 0.22, np.float32)
    for _ in range(rng.integers(1, 4)):
        cy, cx = rng.integers(0, height), rng.integers(0, width)
        r = rng.integers(max(4, height // 16), max(6, height // 6))
        yy0, yy1 = max(0, cy - r), min(height, cy + r + 1)
        xx0, xx1 = max(0, cx - r), min(width, cx + r + 1)
        ir[yy0:yy1, xx0:xx1] = rng.uniform(0.65, 0.95)
    ir = np.clip(ir + rng.normal(0, 0.02, size=ir.shape).astype(np.float32), 0, 1)

    vi = np.clip(vi, 0, 1)
    ir = torch.from_numpy(ir).unsqueeze(0)  # (1,H,W)
    vi = torch.from_numpy(vi.transpose(2, 0, 1).copy())  # (3,H,W)
    return ir, vi


class SyntheticPairDataset(Dataset):
    """Deterministic synthetic pairs. Requires an explicit opt-in flag."""

    def __init__(
        self,
        length: int = 8,
        *,
        height: int = 128,
        width: int = 128,
        seed: int = 0,
        allow_synthetic: bool = False,
    ):
        if not allow_synthetic:
            raise SyntheticDatasetNotPermitted(
                "SyntheticPairDataset is for unit tests only. Pass "
                "allow_synthetic=True to acknowledge that the resulting numbers "
                "are NOT real fusion results, or use "
                "fusionedge.data.build_dataloaders(dataset_id) instead."
            )
        self.length = length
        self.height = height
        self.width = width
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # Without this guard the sequence protocol never terminates: iterating a
        # Dataset walks idx upward until __getitem__ raises IndexError, and
        # generating synthetic pairs "works" for any integer, so `for x in ds`
        # would run forever.
        if idx < 0:
            idx += self.length
        if idx < 0 or idx >= self.length:
            raise IndexError(
                f"SyntheticPairDataset index {idx} out of range for length {self.length}"
            )
        ir, vi = make_pair(self.seed + idx, self.height, self.width)
        return {"ir": ir, "vi": vi, "pair_id": f"syn_{self.seed + idx:06d}", "synthetic": True}