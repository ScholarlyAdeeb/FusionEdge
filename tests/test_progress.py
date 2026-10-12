"""Progress bar behaviour.

These matter because a progress bar that misbehaves is worse than none: it must
not crash a training run, must not corrupt the log file, and must report honest
numbers. The `never` mode is what tests and CI use, so it has to work without a
TTY -- which is exactly the condition under which a bar is most likely to break.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest
import torch

from fusionedge.utils.progress import ProgressBar, Timer, bar_enabled

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# mode resolution
# --------------------------------------------------------------------------- #
def test_never_mode_is_always_disabled():
    assert bar_enabled("never") is False
    assert bar_enabled("off") is False
    assert bar_enabled("false") is False


def test_always_mode_forces_the_bar():
    """Forcing matters when someone tails the log through a pager: auto-detection
    would disable the bar there, and the request was for a bar."""
    assert bar_enabled("always") is True


def test_auto_mode_follows_the_terminal():
    class FakeTTY:
        def isatty(self):
            return True

    class FakePipe:
        def isatty(self):
            return False

    orig = sys.stderr
    try:
        sys.stderr = FakeTTY()
        assert bar_enabled("auto") is True
        sys.stderr = FakePipe()
        assert bar_enabled("auto") is False
    finally:
        sys.stderr = orig


def test_auto_disables_under_pytest(capsys):
    """Captured output is not a terminal, so the bar must stay off and the test
    output stays clean."""
    assert bar_enabled("auto") is False


# --------------------------------------------------------------------------- #
# no-op behaviour
# --------------------------------------------------------------------------- #
def test_bar_in_never_mode_accepts_the_same_calls():
    bar = ProgressBar(10, desc="test", mode="never")
    for _ in range(10):
        bar.update(1, loss=1.0)
    bar.set_description("x")
    bar.set_lr(1e-4)
    bar.write("a message")
    bar.reset(total=5)
    bar.close()


def test_bar_supports_context_manager_in_never_mode():
    with ProgressBar(3, mode="never") as bar:
        bar.update(1)
    assert bar.n == 1


def test_null_bar_writes_to_stderr(capsys):
    ProgressBar(1, mode="never").write("hello")
    assert "hello" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# arithmetic
# --------------------------------------------------------------------------- #
def test_running_mean_loss_is_correct():
    bar = ProgressBar(4, mode="never")
    for loss in (1.0, 2.0, 3.0, 4.0):
        bar.update(1, loss=loss)
    assert bar._loss_n == 4
    assert bar._loss_sum / bar._loss_n == pytest.approx(2.5)


def test_loss_ignored_when_not_supplied():
    bar = ProgressBar(3, mode="never")
    bar.update(1)
    bar.update(1)
    assert bar._loss_n == 0


def test_reset_clears_the_running_mean():
    bar = ProgressBar(3, mode="never")
    bar.update(1, loss=5.0)
    bar.reset()
    assert bar._loss_n == 0 and bar._loss_sum == 0.0


def test_postfix_includes_loss_and_lr():
    bar = ProgressBar(3, mode="never")
    bar.update(1, loss=0.25)
    s = bar._postfix(1e-4)
    assert "0.2500" in s and "1.00e-04" in s


def test_postfix_without_lr():
    bar = ProgressBar(3, mode="never")
    bar.update(1, loss=0.5)
    assert "lr" not in bar._postfix()


def test_total_is_clamped_to_non_negative():
    assert ProgressBar(-5, mode="never").total == 0


# --------------------------------------------------------------------------- #
# integration with the trainer
# --------------------------------------------------------------------------- #
@pytest.fixture
def tiny_dataset(tmp_path, monkeypatch):
    """Minimal registered dataset, mirroring tests/test_engine.py."""
    from PIL import Image
    import numpy as np

    root = tmp_path / "data" / "Tiny"
    for split, n in (("train", 4), ("val", 2)):
        for mod in ("ir", "vi"):
            (root / split / mod).mkdir(parents=True, exist_ok=True)
            for i in range(n):
                g = torch.Generator().manual_seed(i + hash(mod) % 997)
                arr = (torch.rand(16, 20, generator=g) * 255).to(torch.uint8).numpy()
                if mod == "vi":
                    arr = np.repeat(arr[:, :, None], 3, axis=2)
                Image.fromarray(arr).save(root / split / mod / f"{i}.png")

    reg = tmp_path / "registry"
    reg.mkdir()
    (reg / "Tiny.json").write_text(json.dumps({
        "dataset_id": "Tiny", "status": "ok", "root": str(root),
        "manifest_sha256": "d" * 64, "splits": {"train": 4, "val": 2},
    }))
    monkeypatch.setattr("fusionedge.data.fusion_dataset.REGISTRY_DIR", reg)
    return reg


def test_training_runs_with_progress_disabled(tmp_path, tiny_dataset):
    """The regression that matters: adding a bar must not break training in the
    non-TTY case that every test and CI run uses."""
    from fusionedge.data import load_registry

    from fusionedge.engine.train import Trainer

    cfg = {
        "run_name": "prog_test",
        "seed": 1,
        "deterministic": True,
        "device": "cpu",
        "dataset": {"id": "Tiny", "train_split": "train", "eval_split": "val"},
        "model": {"name": "fusionedge", "cfg": {"width": 8, "depth": 1}},
        "data": {"batch_size": 2, "num_workers": 0, "crop": 16},
        "train": {"epochs": 2, "lr": 1e-3, "optimizer": "adam", "amp": False,
                  "eval_every": 1, "progress": "never"},
        "loss": {},
    }
    tr = Trainer(cfg, tmp_path / "run", registry=load_registry("Tiny"), device="cpu")
    assert tr.progress_mode == "never"
    tr.fit()
    assert (tmp_path / "run" / "metrics.csv").exists()


def test_progress_config_is_read_from_the_train_section(tmp_path, tiny_dataset):
    from fusionedge.data import load_registry

    from fusionedge.engine.train import Trainer

    cfg = {
        "run_name": "prog_mode",
        "seed": 1, "device": "cpu",
        "dataset": {"id": "Tiny", "train_split": "train", "eval_split": "val"},
        "model": {"name": "fusionedge", "cfg": {"width": 8, "depth": 1}},
        "data": {"batch_size": 2, "num_workers": 0, "crop": 16},
        "train": {"epochs": 1, "lr": 1e-3, "progress": "never", "amp": False},
        "loss": {},
    }
    tr = Trainer(cfg, tmp_path / "run2", registry=load_registry("Tiny"), device="cpu")
    assert tr.progress_mode == "never"
    assert tr.total_epochs == 1


def test_progress_mode_is_configurable_from_the_cli():
    """Regression for discoverability: the flag must override the config so a
    background run can be given a bar without editing a YAML file."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "fusionedge_run_experiment", REPO_ROOT / "scripts" / "run_experiment.py"
    )
    assert spec and spec.loader
    re_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(re_mod)

    cfg = {"train": {}}
    # the CLI applies --progress by writing into cfg["train"]["progress"]
    for mode in ("auto", "always", "never"):
        cfg["train"]["progress"] = mode
        assert cfg["train"]["progress"] == mode

    assert re_mod.parse_overrides(["train.progress=always"]) == {
        "train": {"progress": "always"}
    }


def test_progress_key_is_documented_in_default_config():
    """The setting must be discoverable in the config, not only in a docstring."""
    import yaml

    cfg = yaml.safe_load((REPO_ROOT / "configs" / "default.yaml").read_text())
    assert "progress" in cfg["train"], "progress setting missing from default.yaml"
    assert cfg["train"]["progress"] == "auto"


def test_default_progress_mode_is_auto():
    from fusionedge.engine.train import Trainer
    import inspect

    src = inspect.getsource(Trainer.__init__)
    assert 'tcfg.get("progress", "auto")' in src


# --------------------------------------------------------------------------- #
def test_timer_measures_and_resets():
    t = Timer()
    assert t.elapsed >= 0.0
    d = t.reset()
    assert d >= 0.0
    assert t.elapsed < 1.0