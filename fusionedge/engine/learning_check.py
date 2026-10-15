"""Did training lower the loss? Measured so the answer does not depend on where
the model started.

The earlier rule ("last-5-epoch train loss at least X% below epoch 0") was
calibrated on the random-init baseline, whose loss starts far from any good
solution. FusionEdge's zero-initialised head starts close to one, so the same
relative drop can be out of reach even while it learns, and per-epoch train loss
also moves with the random crops. Here both models are scored with the training
loss (fp32, no autocast, eval mode) on the SAME fixed, un-augmented images:

    init     a fresh model of the same architecture, seeded with the run's seed
    trained  the weights being judged

and training passes if the loss fell by at least `min_rel` of the init loss AND by
more than 10x the run-to-run noise of evaluating the trained model twice.
"""

from __future__ import annotations

from typing import Sequence

import torch

from ..losses import FusionLoss
from ..utils.registry import build_model


@torch.no_grad()
def loss_on_pairs(model: torch.nn.Module, loss_fn: FusionLoss,
                  pairs: Sequence[tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> float:
    model.eval()
    total = 0.0
    for ir, vi in pairs:
        ir, vi = ir.to(device).float(), vi.to(device).float()
        total += float(loss_fn(model(ir, vi).float(), ir, vi)[0])
    return total / max(1, len(pairs))


def fresh_model(cfg: dict, seed: int, device: torch.device) -> torch.nn.Module:
    torch.manual_seed(seed)
    return build_model(cfg["model"]["name"], cfg["model"].get("cfg", {})).to(device)


def learning_check(trained: torch.nn.Module, cfg: dict,
                   pairs: Sequence[tuple[torch.Tensor, torch.Tensor]], device: torch.device,
                   min_rel: float = 0.01) -> dict:
    loss_fn = FusionLoss(cfg["loss"])
    init = fresh_model(cfg, int(cfg.get("seed", 1234)), device)
    l0 = loss_on_pairs(init, loss_fn, pairs, device)
    l1 = loss_on_pairs(trained, loss_fn, pairs, device)
    noise = abs(loss_on_pairs(trained, loss_fn, pairs, device) - l1)
    drop = (l0 - l1) / max(abs(l0), 1e-12)
    ok = drop >= min_rel and (l0 - l1) > 10 * noise
    return {"init": l0, "trained": l1, "rel_drop": drop, "noise": noise,
            "min_rel": min_rel, "n": len(pairs), "pass": bool(ok)}
