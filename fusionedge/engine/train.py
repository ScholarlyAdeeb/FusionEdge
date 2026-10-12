"""Training engine.

Design points that matter for honesty of results:

  * The validation metric used for `best.pt` is fixed in the config and printed
    at startup. "Best" is whichever metric you say it is; a fusion paper that
    reports the max over {EN, AG, MS-SSIM} per image has selected the metric per
    image, which is not a metric. We pick one, declare it, and log all of them.
  * AMP is on by default with a GradScaler, and `channels_last` is used, both of
    which matter on an 8 GB laptop GPU.
  * Every epoch writes `last.pt`; a crash or a Ctrl-C loses at most one epoch,
    and `--resume` picks up from the exact RNG state.
  * Deterministic kernels are on during training (slower, but the ablation table
    is worthless if the only difference between two runs is kernel
    nondeterminism). Latency benchmarks turn them off explicitly.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..data import build_dataloaders
from ..losses import FusionLoss
from ..utils.checkpoint import load_checkpoint, save_checkpoint
from ..utils.logger import RunLogger
from ..utils.progress import ProgressBar, Timer
from ..utils.registry import build_model
from ..utils.seed import seed_everything
from .evaluate import evaluate_model


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    name = str(cfg.get("optimizer", "adam")).lower()
    lr = float(cfg.get("lr", 1e-4))
    wd = float(cfg.get("weight_decay", 0.0))
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=float(cfg.get("momentum", 0.9)), weight_decay=wd
        )
    raise KeyError(f"unknown optimizer '{name}' (adam | adamw | sgd)")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, epochs: int):
    """Build an LR scheduler stepped once per optimizer step.

    `T_max` is therefore the total number of *optimizer steps* over training, not
    the epoch count, so the cosine actually decays to `eta_min` by the last batch.
    `steps_per_epoch` is passed in by the Trainer once the loader exists; when it
    is unknown the scheduler falls back to an epoch-count estimate and the caller
    is expected to rebuild it.
    """
    name = str(cfg.get("scheduler", "cosine")).lower()
    if name in ("none", ""):
        return None
    total_steps = int(epochs) * int(cfg.get("steps_per_epoch", 1))
    total_steps = max(1, total_steps)

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=float(cfg.get("eta_min", 1e-6))
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(cfg.get("step_size", max(1, total_steps // 3))),
            gamma=float(cfg.get("gamma", 0.5)),
        )
    raise KeyError(f"unknown scheduler '{name}' (cosine | step | none)")


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float = -1e18
    best_epoch: int = -1


class Trainer:
    def __init__(
        self,
        cfg: dict,
        run_dir: str | Path,
        *,
        registry: dict | None = None,
        device: str | None = None,
        argv: list[str] | None = None,
    ):
        from ..utils.config import dump_config, stamp

        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.log = RunLogger(self.run_dir, name=str(cfg.get("run_name", "run")))
        self.device = torch.device(
            device or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )

        seed_everything(
            int(cfg.get("seed", 1234)), deterministic=bool(cfg.get("deterministic", True))
        )

        # provenance: config + dataset identity + environment, all in one file
        stamped = stamp(cfg, dataset_registry=registry, argv=argv)
        dump_config(stamped, self.run_dir / "config.yaml")
        self.stamped = stamped

        dcfg = cfg.get("dataset", {})
        data_cfg = cfg.get("data", {})
        self.train_loader, self.val_loader, self.registry = build_dataloaders(
            dataset_id=dcfg["id"],
            batch_size=int(data_cfg.get("batch_size", 8)),
            num_workers=int(data_cfg.get("num_workers", 4)),
            crop=data_cfg.get("crop", 128),
            augment=True,
            train_split=dcfg.get("train_split", "train"),
            eval_split=dcfg.get("eval_split", "val"),
            limit_train=data_cfg.get("limit_train"),
            limit_eval=data_cfg.get("limit_eval"),
        )

        self.model = build_model(cfg["model"]["name"], cfg["model"].get("cfg", {})).to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)

        self.loss_fn = FusionLoss(cfg.get("loss", {})).to(self.device)
        tcfg = dict(cfg.get("train", {}))
        # The LR schedule decays per optimizer step, so the scheduler needs the
        # real step count -- which is only known now that the loader exists.
        tcfg["steps_per_epoch"] = len(self.train_loader)
        self.tcfg = tcfg
        self.optimizer = build_optimizer(self.model, tcfg)
        self.scheduler = build_scheduler(self.optimizer, tcfg, int(tcfg.get("epochs", 100)))
        self.amp = bool(tcfg.get("amp", True)) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.grad_clip = float(tcfg.get("grad_clip", 0.0) or 0.0)
        self.eval_every = int(tcfg.get("eval_every", 1))
        self.total_epochs = int(tcfg.get("epochs", 100))
        #: 'auto' (bar only on an interactive terminal), 'always', or 'never'
        self.progress_mode = str(tcfg.get("progress", "auto"))
        #: the single metric `best.pt` is selected on -- declared, not chosen
        #: after the fact. Sign is applied so that "higher is better" always holds.
        self.best_key = str(tcfg.get("best_metric", "MS_SSIM_ir"))
        self.state = TrainState()
        self._log_setup()

    # ------------------------------------------------------------------ #
    def _log_setup(self) -> None:
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log.section("run setup")
        self.log.kv(
            {
                "run_dir": str(self.run_dir),
                "model": self.cfg["model"]["name"],
                "parameters": f"{n_params:,}",
                "device": str(self.device),
                "amp": self.amp,
                "channels_last": self.device.type == "cuda",
                "dataset_id": self.registry.get("dataset_id"),
                "dataset_manifest": str(self.registry.get("manifest_sha256"))[:16],
                "dataset_splits": json.dumps(self.registry.get("splits", {})),
                "train_batches": len(self.train_loader),
                "val_images": len(self.val_loader.dataset),
                "batch_size": self.tcfg.get("batch_size"),
                "crop": self.cfg.get("data", {}).get("crop"),
                "epochs": self.tcfg.get("epochs"),
                "optimizer": self.tcfg.get("optimizer", "adam"),
                "lr": self.tcfg.get("lr"),
                "loss": self.loss_fn.extra_repr(),
                "best_metric": self.best_key,
                "deterministic": self.cfg.get("deterministic", True),
            },
            title="configuration",
        )
        if self.device.type == "cuda":
            props = torch.cuda.get_device_properties(0)
            self.log.info(
                f"  gpu: {props.name} | {props.total_memory / 2**30:.1f} GiB VRAM | "
                f"torch {torch.__version__}"
            )

    # ------------------------------------------------------------------ #
    def _to_device(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        ir = batch["ir"].to(self.device, non_blocking=True)
        vi = batch["vi"].to(self.device, non_blocking=True)
        if self.device.type == "cuda":
            ir = ir.contiguous(memory_format=torch.channels_last)
            vi = vi.contiguous(memory_format=torch.channels_last)
        return ir, vi

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        timer = Timer()
        agg: dict[str, float] = {}
        n = 0

        total_steps = len(self.train_loader)
        bar = ProgressBar(
            total_steps,
            desc=f"epoch {epoch + 1}/{self.total_epochs}",
            mode=self.progress_mode,
            leave=False,
        )

        for i, batch in enumerate(self.train_loader):
            ir, vi = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            # Step the LR schedule per optimizer step, not per epoch. Stepping it
            # per epoch also triggers PyTorch's "scheduler.step() before
            # optimizer.step()" warning on the first batch, and conflates the
            # schedule with the epoch count.
            if self.scheduler is not None:
                self.scheduler.step()

            if self.amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    fused = self.model(ir, vi)
                    # losses are computed in fp32: the intensity/gradient terms
                    # compare near-identical tensors, and fp16 subtraction there
                    # loses the small differences the loss is trying to measure.
                    loss, parts = self.loss_fn(fused.float(), ir.float(), vi.float())
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                fused = self.model(ir, vi)
                loss, parts = self.loss_fn(fused, ir, vi)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            self.state.global_step += 1
            bs = ir.shape[0]
            n += bs
            agg["loss"] = agg.get("loss", 0.0) + loss.item() * bs
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v.item() * bs

            bar.update(1, loss=loss.item())
            bar.set_lr(self.optimizer.param_groups[0]["lr"])

            # The periodic log line is kept for the permanent record; when the
            # progress bar is active it is redundant on screen but still written
            # to log.txt, so a tailed log has the same information either way.
            if (i + 1) % 50 == 0:
                self.log.info(
                    f"  epoch {epoch + 1} | step {i + 1}/{total_steps} | "
                    f"loss {loss.item():.4f} | lr "
                    f"{self.optimizer.param_groups[0]['lr']:.2e}"
                )

        bar.close()

        out = {k: v / max(n, 1) for k, v in agg.items()}
        out["epoch_time_s"] = round(timer.reset(), 2)
        return out

    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def validate(self, epoch: int) -> dict[str, Any]:
        self.log.info(f"  [val] epoch {epoch + 1}/{self.total_epochs} ...")
        eval_bar = ProgressBar(
            len(self.val_loader),
            desc=f"  val {epoch + 1}/{self.total_epochs}",
            mode=self.progress_mode,
            unit="img",
            leave=False,
        )
        res = evaluate_model(
            self.model,
            self.val_loader,
            device=self.device,
            amp=self.amp,
            max_images=self.tcfg.get("eval_max_images"),
            # the eval loader runs at batch size 1, so one callback == one image
            progress=lambda n, total: eval_bar.update(1),
        )
        eval_bar.close()
        m = res["metrics"]
        self.log.info(
            f"  [val] epoch {epoch + 1}: "
            + " | ".join(f"{k}={v:.4f}" for k, v in sorted(m.items()))
        )
        return res

    def fit(self, resume_from: str | Path | None = None) -> TrainState:
        epochs = self.total_epochs
        if resume_from:
            self.load(resume_from)
            self.log.info(f"resumed from {resume_from} at epoch {self.state.epoch}")

        ckpt_dir = self.run_dir / "checkpoints"
        self.log.section(f"training for {epochs} epochs")
        sweep_bar = ProgressBar(
            epochs * len(self.train_loader),
            desc="total",
            mode=self.progress_mode,
            leave=True,
        )
        try:
            for epoch in range(self.state.epoch, epochs):
                train_stats = self.train_epoch(epoch)
                row = {"epoch": epoch, "phase": "train", **train_stats}

                is_last = epoch == epochs - 1
                if (epoch + 1) % self.eval_every == 0 or is_last:
                    val = self.validate(epoch)
                    row.update({f"val_{k}": v for k, v in val["metrics"].items()})
                    score = val["metrics"].get(self.best_key)
                    if score is not None and score > self.state.best_metric:
                        self.state.best_metric = float(score)
                        self.state.best_epoch = epoch
                        save_checkpoint(
                            ckpt_dir / "best.pt",
                            self.model,
                            optimizer=self.optimizer,
                            scaler=self.scaler,
                            epoch=epoch,
                            global_step=self.state.global_step,
                            best_metric=self.state.best_metric,
                            config=self.stamped,
                            metrics=val["metrics"],
                        )
                        self.log.info(f"  new best {self.best_key}={score:.4f} -> best.pt")

                self.log.log_metrics(row)
                self.state.epoch = epoch + 1
                sweep_bar.update(len(self.train_loader), loss=train_stats["loss"])
                save_checkpoint(
                    ckpt_dir / "last.pt",
                    self.model,
                    optimizer=self.optimizer,
                    scaler=self.scaler,
                    epoch=self.state.epoch,
                    global_step=self.state.global_step,
                    best_metric=self.state.best_metric,
                    config=self.stamped,
                )

        except KeyboardInterrupt:
            sweep_bar.close()
            self.log.warn("interrupted -- saving state so training can resume")
            save_checkpoint(
                ckpt_dir / "last.pt",
                self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                epoch=self.state.epoch,
                global_step=self.state.global_step,
                best_metric=self.state.best_metric,
                config=self.stamped,
            )
            raise

        finally:
            sweep_bar.close()

        self.log.info(f"done. best {self.best_key}={self.state.best_metric:.4f} "
                      f"at epoch {self.state.best_epoch + 1}")
        self.log.close()
        return self.state

    def load(self, path: str | Path) -> TrainState:
        payload = load_checkpoint(
            path,
            self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
            map_location=self.device,
        )
        self.state = TrainState(
            epoch=int(payload.get("epoch", 0)),
            global_step=int(payload.get("global_step", 0)),
            best_metric=float(payload.get("best_metric", -1e18)),
            best_epoch=-1,
        )
        return self.state