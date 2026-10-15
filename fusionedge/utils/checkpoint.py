"""Checkpoint save/load with full training state.

A checkpoint stores enough to resume bit-identically: model + optimizer + AMP
scaler state, the epoch/iteration counters, the RNG states of python/numpy/torch,
and the stamped config. `last.pt` is overwritten every epoch (crash safety);
`best.pt` is written only when the tracked validation metric improves; periodic
`epoch_<n>.pt` snapshots let you resume from an arbitrary past point or plot a
training curve from snapshots alone.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def load_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any = None,
    scheduler: Any = None,
    best_epoch: int | None = None,
    epoch: int = 0,
    global_step: int = 0,
    best_metric: float | None = None,
    config: dict | None = None,
    metrics: dict | None = None,
    include_rng: bool = True,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # weights_only=False: we intentionally persist RNG tuples + a config dict,
    # which are not plain tensors. Only ever load checkpoints you produced.
    payload: dict[str, Any] = {
        "format_version": 1,
        "model_state": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "config": config or {},
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if include_rng:
        payload["rng_state"] = rng_state()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic-ish: never leave a half-written last.pt
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module | None = None,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any = None,
    map_location: str | torch.device = "cpu",
    load_optimizer: bool = True,
    scheduler: Any = None,
    restore_rng: bool = False,
) -> dict[str, Any]:
    """Load a checkpoint. Returns the full payload for inspection."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None:
        state = payload["model_state"]
        # tolerate checkpoints saved before the head was renamed
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[checkpoint] non-strict load: missing={list(missing)} "
                f"unexpected={list(unexpected)}"
            )
    if optimizer is not None and load_optimizer and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scaler is not None and load_optimizer and "scaler_state" in payload:
        scaler.load_state_dict(payload["scaler_state"])
    if scheduler is not None and load_optimizer and "scheduler_state" in payload:
        scheduler.load_state_dict(payload["scheduler_state"])
    if restore_rng and payload.get("rng_state") is not None:
        st = payload["rng_state"]
        # torch.load(map_location=cuda) moves RNG ByteTensors to the GPU; the
        # setters require CPU tensors.
        if hasattr(st.get("torch"), "cpu"):
            st["torch"] = st["torch"].cpu()
        if st.get("cuda") is not None:
            st["cuda"] = [t.cpu() for t in st["cuda"]]
        load_rng_state(st)
    return payload


def load_model_weights(model: torch.nn.Module, path: str | Path, device: str = "cpu") -> None:
    """Strictly load inference weights; raise on any mismatch.

    Deliberately strict (unlike load_checkpoint): inference should never
    silently run with a partially-initialised head.
    """
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model_state", payload)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()