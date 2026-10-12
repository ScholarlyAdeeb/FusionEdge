"""Reproducibility helpers.

The project treats a run as reproducible only if it can be re-created bit-for-bit
given the same config, dataset manifest and seed. Three things break that in
practice and are handled here:

  * global RNGs (python `random`, numpy, torch CPU/CUDA),
  * non-deterministic cuDNN/cuBLAS kernels (slower, but numerically stable),
  * dataloader worker RNGs, which are re-seeded per epoch per worker so that a
    resumed run sees the same augmentation stream as an uninterrupted one.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed all RNGs and optionally force deterministic kernels.

    Args:
        seed: base seed.
        deterministic: if True, forces cuDNN into deterministic algorithm mode
            and disables the TF32 fast paths. This costs throughput (we measured
            roughly 10-20% on this GPU) but makes A/B comparisons between two
            architectures trustworthy, which matters more than speed for the
            ablation table. Latency numbers in results/ are therefore always
            measured with deterministic=False -- see engine/benchmark.py.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def seed_worker(worker_id: int) -> None:
    """DataLoader `worker_init_fn`: give each worker a distinct, reproducible RNG."""
    base = torch.initial_seed() % (2**32)
    np.random.seed((base + worker_id) % (2**32))
    random.seed(base + worker_id)


def worker_init_fn(worker_init) -> object:
    """Adapter returning a picklable zero-arg callable for DataLoader."""

    def _init(worker_id: int) -> None:
        seed_worker(worker_id)
        if worker_init is not None:
            worker_init(worker_id)

    return _init