"""Reproducibility, config and checkpoint round-trips.

The claim "this project is reproducible" has to be a tested property, not a
README sentence. These tests pin the two things that actually break runs:
seed handling and checkpoint/resume fidelity.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fusionedge.models.baseline import SimpleCNNBaseline
from fusionedge.utils.checkpoint import (
    load_checkpoint,
    load_model_weights,
    load_rng_state,
    rng_state,
    save_checkpoint,
)
from fusionedge.utils.config import Config, dump_config, environment_fingerprint, load_config, stamp
from fusionedge.utils.logger import RunLogger
from fusionedge.utils.seed import seed_everything


# --------------------------------------------------------------------------- #
# seeding
# --------------------------------------------------------------------------- #
def test_seed_everything_makes_torch_and_numpy_reproducible():
    seed_everything(1234)
    a = (torch.rand(4), np.random.rand(4))
    seed_everything(1234)
    b = (torch.rand(4), np.random.rand(4))
    assert torch.equal(a[0], b[0])
    assert np.array_equal(a[1], b[1])


def test_different_seeds_give_different_streams():
    seed_everything(1)
    a = torch.rand(8)
    seed_everything(2)
    b = torch.rand(8)
    assert not torch.equal(a, b)


def test_deterministic_mode_is_reversible():
    """Toggling determinism on then off must not poison later runs."""
    seed_everything(7, deterministic=True)
    seed_everything(7, deterministic=False)
    seed_everything(7, deterministic=False)
    x = torch.rand(4)
    seed_everything(7, deterministic=False)
    assert torch.equal(x, torch.rand(4))


def test_deterministic_mode_disables_tf32():
    """TF32 convs gave ~2e-3 output gaps between memory layouts in "fp32"."""
    seed_everything(7, deterministic=True)
    assert not torch.backends.cudnn.allow_tf32
    assert not torch.backends.cuda.matmul.allow_tf32
    seed_everything(7, deterministic=False)
    assert torch.backends.cudnn.allow_tf32
    assert torch.backends.cuda.matmul.allow_tf32


def test_worker_init_fn_is_picklable():
    """Regression: a closure here breaks every multi-worker DataLoader on Windows.

    The spawn start method pickles worker_init_fn before sending it to the worker,
    and a nested function has no importable identity -> PicklingError. Only Linux
    with fork tolerates it, so this is a platform-dependent failure that CI on
    Linux would not catch.
    """
    import pickle

    from fusionedge.utils.seed import worker_init_fn

    init = worker_init_fn(None)
    restored = pickle.loads(pickle.dumps(init))
    assert callable(restored)
    restored(0)  # must actually run


def test_worker_init_fn_runs_with_and_without_a_callback():
    from fusionedge.utils.seed import worker_init_fn

    seen: list[int] = []
    worker_init_fn(lambda wid: seen.append(wid))(3)
    assert seen == [3]

    worker_init_fn(None)(0)  # no callback must still work


def test_worker_init_fn_seeds_workers_differently():
    """Distinct workers must get distinct numpy/python RNG streams.

    torch's own generator is already per-worker (the DataLoader seeds it from
    `initial_seed() + worker_id`), so the risk this guards is in numpy/random,
    which our augmentation code draws from directly.
    """
    import pickle

    from fusionedge.utils.seed import worker_init_fn

    init = pickle.loads(pickle.dumps(worker_init_fn(None)))

    np.random.seed(0)
    init(0)
    a = np.random.rand(4)

    np.random.seed(0)
    init(1)
    b = np.random.rand(4)

    assert not np.array_equal(a, b)


def test_rng_state_roundtrip_restores_stream():
    seed_everything(99)
    state = rng_state()
    expected = torch.rand(16)
    load_rng_state(state)
    assert torch.equal(torch.rand(16), expected)


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_config_dotted_access():
    cfg = Config({"train": {"batch_size": 4}})
    assert cfg.train.batch_size == 4
    assert cfg.get_path("train.batch_size") == 4
    assert cfg.get_path("train.missing", "fallback") == "fallback"
    assert cfg.get_path("nope.nope.nope", None) is None


def test_config_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        _ = Config({}).does_not_exist


def test_load_config_supports_base_inheritance(tmp_path: Path):
    (tmp_path / "base.yaml").write_text("train:\n  batch_size: 8\n  lr: 0.001\nseed: 1\n")
    (tmp_path / "child.yaml").write_text("_base_: base.yaml\ntrain:\n  lr: 0.01\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.train.lr == 0.01, "child must override the base"
    assert cfg.train.batch_size == 8, "unoverridden base keys must survive"
    assert cfg.seed == 1


def test_load_config_overrides_win(tmp_path: Path):
    (tmp_path / "c.yaml").write_text("train:\n  batch_size: 8\n")
    cfg = load_config(tmp_path / "c.yaml", overrides={"train": {"batch_size": 2}})
    assert cfg.train.batch_size == 2


def test_stamp_records_provenance(tmp_path: Path):
    cfg = stamp(Config({"a": 1}), dataset_registry={
        "dataset_id": "MFNet", "status": "ok", "manifest_sha256": "abc123", "splits": {"train": 10}
    }, argv=["train.py", "--x"])
    s = cfg["_stamped"]
    assert s["dataset_id"] == "MFNet"
    assert s["dataset_manifest_sha256"] == "abc123"
    assert s["argv"] == ["train.py", "--x"]
    assert "torch" in s["environment"] and s["git_commit"]


def test_dump_config_writes_yaml_and_json(tmp_path: Path):
    p = tmp_path / "config.yaml"
    dump_config(Config({"model": {"name": "x"}, "n": 3}), p)
    assert p.exists() and p.with_suffix(".json").exists()
    assert json.loads(p.with_suffix(".json").read_text())["model"]["name"] == "x"


def test_environment_fingerprint_has_no_missing_keys():
    fp = environment_fingerprint()
    for k in ("torch", "cuda_available", "python"):
        assert k in fp


# --------------------------------------------------------------------------- #
# checkpoints
# --------------------------------------------------------------------------- #
def test_checkpoint_roundtrip_restores_weights_exactly(tmp_path: Path):
    model = SimpleCNNBaseline(width=8, depth=1)
    path = save_checkpoint(tmp_path / "ck.pt", model, epoch=3, global_step=30)
    fresh = SimpleCNNBaseline(width=8, depth=1)
    payload = load_checkpoint(path, fresh)
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), fresh.state_dict().items()):
        assert k1 == k2 and torch.equal(v1, v2)
    assert payload["epoch"] == 3 and payload["global_step"] == 30


def test_checkpoint_resume_reproduces_the_next_batch(tmp_path: Path):
    """The real test of resume: replaying from a checkpoint must give the same
    parameter update as an uninterrupted run."""
    ir = torch.rand(2, 1, 32, 32)
    vi = torch.rand(2, 3, 32, 32)
    target_img = torch.rand(2, 3, 32, 32)

    def one_step(model, opt, ir, vi):
        opt.zero_grad()
        torch.nn.functional.l1_loss(model(ir, vi), target_img).backward()
        opt.step()

    seed_everything(5, deterministic=True)
    m1 = SimpleCNNBaseline(width=8, depth=1)
    o1 = torch.optim.Adam(m1.parameters(), lr=1e-2)
    one_step(m1, o1, ir, vi)
    target = [p.detach().clone() for p in m1.parameters()]

    # fresh model, then resume from the checkpoint and take the same step
    seed_everything(5, deterministic=True)
    m2 = SimpleCNNBaseline(width=8, depth=1)
    o2 = torch.optim.Adam(m2.parameters(), lr=1e-2)
    save_checkpoint(tmp_path / "r.pt", m2, optimizer=o2, epoch=0)
    m3 = SimpleCNNBaseline(width=8, depth=1)
    o3 = torch.optim.Adam(m3.parameters(), lr=1e-2)
    load_checkpoint(tmp_path / "r.pt", m3, optimizer=o3)
    one_step(m3, o3, ir, vi)
    for a, b in zip(target, m3.parameters()):
        assert torch.allclose(a, b.detach(), atol=1e-6)


def test_load_model_weights_is_strict(tmp_path: Path):
    model = SimpleCNNBaseline(width=8, depth=1)
    save_checkpoint(tmp_path / "w.pt", model)
    mismatched = SimpleCNNBaseline(width=16, depth=1)
    with pytest.raises(RuntimeError):
        load_model_weights(mismatched, tmp_path / "w.pt")


def test_checkpoint_contains_optimizer_and_scaler_state(tmp_path: Path):
    model = SimpleCNNBaseline(width=8, depth=1)
    opt = torch.optim.Adam(model.parameters())
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    save_checkpoint(tmp_path / "o.pt", model, optimizer=opt, scaler=scaler)
    payload = torch.load(tmp_path / "o.pt", map_location="cpu", weights_only=False)
    assert "optimizer_state" in payload and "scaler_state" in payload


def test_save_checkpoint_is_atomic(tmp_path: Path):
    """No .tmp file should survive a successful save."""
    model = SimpleCNNBaseline(width=8, depth=1)
    save_checkpoint(tmp_path / "last.pt", model)
    assert (tmp_path / "last.pt").exists()
    assert not list(tmp_path.glob("*.tmp"))


# --------------------------------------------------------------------------- #
# logger
# --------------------------------------------------------------------------- #
def test_logger_writes_log_and_appends_csv(tmp_path: Path):
    log = RunLogger(tmp_path / "run1")
    log.info("hello")
    log.log_metrics({"epoch": 1, "loss": 0.5})
    log.log_metrics({"epoch": 2, "loss": 0.4})
    log.close()

    assert "hello" in (tmp_path / "run1" / "log.txt").read_text()
    rows = list(csv_rows(tmp_path / "run1" / "metrics.csv"))
    assert len(rows) == 2
    assert rows[0]["epoch"] == "1" and rows[1]["loss"] == "0.4"


def test_logger_extends_csv_schema_without_rewriting(tmp_path: Path):
    log = RunLogger(tmp_path / "run2")
    log.log_metrics({"epoch": 1, "loss": 0.5})
    log.log_metrics({"epoch": 2, "loss": 0.4, "extra": 1})
    log.close()
    rows = list(csv_rows(tmp_path / "run2" / "metrics.csv"))
    assert len(rows) == 2
    assert rows[0]["extra"] == "", "old rows keep their values"
    assert rows[1]["extra"] == "1"


def csv_rows(path: Path):
    import csv

    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))