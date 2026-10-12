"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fusionedge.data.synthetic import SyntheticPairDataset, make_pair  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_pairs() -> list[tuple[torch.Tensor, torch.Tensor]]:
    """A handful of deterministic synthetic (ir, vi) pairs for shape/range tests."""
    return [make_pair(seed=s, height=64, width=80) for s in range(3)]


@pytest.fixture(scope="session")
def synthetic_dataset() -> SyntheticPairDataset:
    return SyntheticPairDataset(length=4, height=32, width=48, seed=1, allow_synthetic=True)


@pytest.fixture(scope="session")
def rand_rgb() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.rand(1, 3, 32, 32, generator=g)