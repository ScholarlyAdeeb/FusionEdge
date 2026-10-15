"""Mixed-precision policy.

Context: an earlier baseline run used float16 autocast and silently failed. The
GradScaler scale collapsed from 32768 to 0 within 50 steps because gradients
overflowed, so nearly every optimizer step was skipped -- and because the loss
curve still looked like plausible noise, nothing raised. Measured from the
identical checkpoint over 250 steps:

    fp16   loss improved  1.4%   (scale collapsed to 0)
    fp32   loss improved 39.9%

These tests pin the policy that prevents a repeat:

  * bfloat16 is the default, and fp32 is the safe fallback,
  * a GradScaler is enabled for fp16 only -- with bf16 it would mask an overflow,
  * a collapsed scaler raises rather than training on almost no gradients,
  * a streak of non-finite values aborts the run instead of being averaged in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from fusionedge.engine.train import AMP_DTYPES, resolve_amp


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #
def test_bfloat16_is_the_default_dtype():
    enabled, name = resolve_amp({}, torch.device("cuda"))
    assert enabled is True
    assert name == "bfloat16"


def test_amp_false_forces_fp32():
    enabled, name = resolve_amp({"amp": False, "amp_dtype": "bfloat16"}, torch.device("cuda"))
    assert enabled is False and name == "float32"


def test_cpu_never_uses_autocast():
    """CPU bf16 autocast is not reliably faster and can lose precision."""
    enabled, name = resolve_amp({}, torch.device("cpu"))
    assert enabled is False and name == "float32"


def test_explicit_float16_is_still_available():
    enabled, name = resolve_amp({"amp_dtype": "float16"}, torch.device("cuda"))
    assert enabled is True and name == "float16"


def test_unknown_dtype_is_rejected():
    with pytest.raises(KeyError, match="unknown amp_dtype"):
        resolve_amp({"amp_dtype": "float8"}, torch.device("cuda"))


def test_grad_scaler_is_needed_only_for_fp16():
    """bfloat16 has fp32's exponent range; a scaler is unnecessary there and,
    worse, would hide an overflow instead of surfacing it."""
    assert AMP_DTYPES["float16"] is True
    assert AMP_DTYPES["bfloat16"] is False
    assert AMP_DTYPES["float32"] is False


def test_bfloat16_has_the_range_fp16_lacks():
    """The physical reason for the choice, asserted rather than asserted-by-doc.

    A head activation of magnitude ~27 is ordinary here; fp16's max finite value
    is 65504 but its *usable* range for gradients is far narrower, while
    bfloat16's exponent range equals fp32's.
    """
    assert torch.finfo(torch.bfloat16).max > 3.0e38
    assert torch.finfo(torch.float16).max == 65504.0


def test_large_activations_are_representable_in_bf16():
    """A value that overflowed the fp16 pipeline during the failed run."""
    x = torch.tensor([27.0, 1e4, 1e5], dtype=torch.float32)
    assert torch.isfinite(x.to(torch.bfloat16)).all()


# --------------------------------------------------------------------------- #
# guards in the training loop
# --------------------------------------------------------------------------- #
@pytest.fixture
def tiny_dataset(tmp_path, monkeypatch):
    from PIL import Image
    import numpy as np

    root = tmp_path / "data" / "Tiny"
    for split in ("train", "val"):
        for mod in ("ir", "vi"):
            (root / split / mod).mkdir(parents=True, exist_ok=True)
            for i in range(4):
                g = torch.Generator().manual_seed(i + hash(mod) % 997)
                arr = (torch.rand(16, 20, generator=g) * 255).to(torch.uint8).numpy()
                if mod == "vi":
                    arr = np.repeat(arr[:, :, None], 3, axis=2)
                Image.fromarray(arr).save(root / split / mod / f"{i}.png")

    reg = tmp_path / "registry"
    reg.mkdir()
    (reg / "Tiny.json").write_text(json.dumps({
        "dataset_id": "Tiny", "status": "ok", "root": str(root),
        "manifest_sha256": "e" * 64, "splits": {"train": 4, "val": 2},
    }))
    monkeypatch.setattr("fusionedge.data.fusion_dataset.REGISTRY_DIR", reg)
    return reg


def _cfg(tmp_path: Path, **train_over) -> dict:
    train = {
        "epochs": 1, "lr": 1e-4, "optimizer": "adam", "scheduler": "none",
        "eval_every": 1, "grad_clip": 1.0, "progress": "never",
        "amp": False, "amp_dtype": "bfloat16",
    }
    train.update(train_over)
    return {
        "run_name": "amp_test", "seed": 1, "deterministic": True, "device": "cpu",
        "dataset": {"id": "Tiny", "train_split": "train", "eval_split": "val"},
        "model": {"name": "fusionedge", "cfg": {"width": 8, "depth": 1}},
        "data": {"batch_size": 2, "num_workers": 0, "crop": 16},
        "train": train,
        "loss": {},
    }


def test_trainer_records_the_precision_policy(tmp_path, tiny_dataset):
    """The precision setting must land in the run's own config, so a metrics.json
    can always be traced back to the numerics it was produced under."""
    from fusionedge.data import load_registry
    from fusionedge.engine.train import Trainer

    tr = Trainer(_cfg(tmp_path), tmp_path / "r1", registry=load_registry("Tiny"),
                 device="cpu")
    logged = (tmp_path / "r1" / "log.txt").read_text(encoding="utf-8")
    assert "amp_dtype" in logged
    assert "float32" in logged


def test_divergence_aborts_instead_of_reporting(tmp_path, tiny_dataset):
    """The core guard: a diverged run must raise, not finish with a metrics.json."""
    from fusionedge.data import load_registry
    from fusionedge.engine.train import Trainer

    tr = Trainer(_cfg(tmp_path, max_nonfinite_streak=3),
                 tmp_path / "r2", registry=load_registry("Tiny"), device="cpu")
    # simulate a divergence without needing a model that genuinely diverges
    for i in range(2):
        tr._register_nonfinite(step=i, epoch=0, stage="loss")
    with pytest.raises(RuntimeError, match="diverged"):
        tr._register_nonfinite(step=2, epoch=0, stage="loss")


def test_single_nonfinite_step_is_tolerated(tmp_path, tiny_dataset):
    """One transient fp16 overflow is not a divergence; aborting on it would make
    fp16 unusable for no benefit."""
    from fusionedge.data import load_registry
    from fusionedge.engine.train import Trainer

    tr = Trainer(_cfg(tmp_path), tmp_path / "r3", registry=load_registry("Tiny"),
                 device="cpu")
    tr._register_nonfinite(step=0, epoch=0, stage="loss")  # must not raise
    assert tr.nonfinite_streak == 1


def test_streak_resets_only_by_aborting(tmp_path, tiny_dataset):
    """Once the streak is high enough the run stops; there is no silent reset that
    would let a diverging run continue."""
    from fusionedge.data import load_registry
    from fusionedge.engine.train import Trainer

    tr = Trainer(_cfg(tmp_path, max_nonfinite_streak=2),
                 tmp_path / "r4", registry=load_registry("Tiny"), device="cpu")
    tr._register_nonfinite(step=0, epoch=0, stage="grad")
    with pytest.raises(RuntimeError, match="non-finite grad"):
        tr._register_nonfinite(step=1, epoch=0, stage="grad")


def test_grad_finite_check_locates_the_bad_parameter(tmp_path, tiny_dataset):
    from fusionedge.data import load_registry
    from fusionedge.engine.train import Trainer

    tr = Trainer(_cfg(tmp_path), tmp_path / "r5", registry=load_registry("Tiny"),
                 device="cpu")
    ok, _ = tr._grads_are_finite()
    assert ok is True  # no grads yet -> nothing non-finite

    out = tr.model(torch.rand(1, 1, 16, 16), torch.rand(1, 3, 16, 16))
    out.mean().backward()
    ok, _ = tr._grads_are_finite()
    assert ok is True

    first_param = next(tr.model.parameters())
    first_param.grad[0, 0, 0, 0] = float("nan")
    ok, name = tr._grads_are_finite()
    assert ok is False and name


def test_training_still_runs_with_guards(tmp_path, tiny_dataset):
    """The guards must not break normal training."""
    from fusionedge.data import load_registry
    from fusionedge.engine.train import Trainer

    tr = Trainer(_cfg(tmp_path, epochs=2), tmp_path / "r6",
                 registry=load_registry("Tiny"), device="cpu")
    tr.fit()
    assert tr.state.epoch == 2
    assert (tmp_path / "r6" / "metrics.csv").exists()


# --------------------------------------------------------------------------- #
# evaluation path uses the same policy
# --------------------------------------------------------------------------- #
def test_fuse_batch_defaults_to_bfloat16():
    import inspect

    from fusionedge.engine.evaluate import fuse_batch

    sig = inspect.signature(fuse_batch)
    assert sig.parameters["amp_dtype"].default == "bfloat16"


def test_fuse_batch_returns_fp32_even_under_autocast():
    """Metrics expect fp32 tensors; returning bf16 would silently reduce metric
    precision."""
    from fusionedge.data.synthetic import make_pair
    from fusionedge.engine.evaluate import fuse_batch
    from fusionedge.utils.registry import build_model

    ir, vi = make_pair(seed=0, height=32, width=32)
    m = build_model("fusionedge", {"width": 8, "depth": 1}).eval()
    out = fuse_batch(m, ir.unsqueeze(0), vi.unsqueeze(0),
                     device=torch.device("cpu"), amp=False)
    assert out.dtype == torch.float32


def test_config_default_is_bfloat16():
    import yaml

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "default.yaml").read_text()
    )
    assert cfg["train"]["amp_dtype"] == "bfloat16"
    assert cfg["train"]["amp"] is True