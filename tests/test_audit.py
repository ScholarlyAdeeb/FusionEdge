"""Audit tooling: the checks must actually catch the failures they claim to.

An audit that always passes is worse than none, because it creates confidence.
Each test here constructs a deliberately-broken run and asserts the relevant
check fails.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from fusionedge.utils.checkpoint import save_checkpoint
from fusionedge.utils.registry import build_model

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fusionedge_audit_run", REPO_ROOT / "scripts" / "audit_run.py"
)
ar = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(ar)


def _write_image(p: Path, seed: int) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np

    Image.fromarray(
        (np.random.default_rng(seed).random((16, 20)) * 255).astype("uint8")
    ).save(p)


@pytest.fixture
def registry_dir(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "data" / "Tiny"
    for split, n in (("train", 4), ("val", 2)):
        for i in range(n):
            _write_image(root / split / "ir" / f"{split}{i}.png", i)
            _write_image(root / split / "vi" / f"{split}{i}.png", i + 50)

    reg = tmp_path / "registry"
    reg.mkdir()
    (reg / "Tiny.json").write_text(json.dumps({
        "dataset_id": "Tiny", "status": "ok", "root": str(root),
        "manifest_sha256": "c" * 64, "splits": {"train": 4, "val": 2},
    }))
    monkeypatch.setattr("fusionedge.data.fusion_dataset.REGISTRY_DIR", reg)
    return reg


def make_run(tmp_path: Path, registry_dir: Path, **over) -> Path:
    """A complete, healthy run directory. `over` lets tests break one thing."""
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    manifest = "c" * 64

    cfg = {
        "dataset": {"id": "Tiny", "train_split": "train", "eval_split": "val"},
        "model": {"name": "fusionedge", "cfg": {"width": 8, "depth": 1}},
        "train": {"epochs": 2},
        "_stamped": {"dataset_id": "Tiny", "dataset_manifest_sha256": manifest},
    }
    (run / "config.json").write_text(json.dumps(cfg))
    (run / "config.yaml").write_text("run_name: test\n")
    (run / "log.txt").write_text("epoch 0\nepoch 1\n")

    (run / "metrics.csv").write_text(
        "epoch,phase,loss\n0,train,5.0\n1,train,2.0\n"
    )

    metrics = {
        "model": {"name": "fusionedge", "params": 123},
        "dataset": {
            "dataset_id": "Tiny", "manifest_sha256": manifest,
            # the audit prefers this path, falling back to the registry's
            "root": str(Path(registry_dir).parent / "data" / "Tiny"),
            "splits": {"train": 4, "val": 2},
        },
        "evaluation": {"num_images": 2, "elapsed_s": 1.0},
        "metrics": {"EN": 6.5, "AG": 0.004, "MS_SSIM_ir": 0.9},
        "smoke": False,
    }
    metrics.update(over.pop("metrics_over", {}))
    (run / "metrics.json").write_text(json.dumps(metrics))

    (run / "benchmark.json").write_text(json.dumps({
        "params": 123,
        "checkpoint_mb": 0.01,
        "gpu": "Test GPU",
        "latency": [
            {"size": [480, 640], "latency_ms_median": 30.0, "fps": 33.0,
             "peak_vram_mb": 250},
        ],
    }))

    model = build_model("fusionedge", {"width": 8, "depth": 1})
    save_checkpoint(run / "checkpoints" / "best.pt", model, epoch=1,
                    best_metric=0.9, global_step=10)
    save_checkpoint(run / "checkpoints" / "last.pt", model, epoch=2)

    for key, value in over.items():
        target = run / key
        if isinstance(value, dict):
            data = json.loads(target.read_text())
            data.update(value)
            target.write_text(json.dumps(data))
        else:
            target.write_text(str(value))
    return run


def failing_checks(run: Path, allow_smoke: bool = False) -> list[str]:
    """Run the audit, return the names of failing checks.

    The check name is the padded column between the [PASS]/[FAIL] marker and the
    detail text; splitting on ']' alone is ambiguous because the detail may
    itself contain brackets (e.g. "[128, 128]").
    """
    import contextlib
    import io
    import re

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ar.audit(run, allow_smoke=allow_smoke)
    return re.findall(r"\[FAIL\]\s+(.+?)\s\s+\S", buf.getvalue())


# --------------------------------------------------------------------------- #
def test_healthy_run_passes(tmp_path, registry_dir):
    run = make_run(tmp_path, registry_dir)
    assert failing_checks(run) == []


def test_detects_missing_artifacts(tmp_path, registry_dir):
    run = make_run(tmp_path, registry_dir)
    (run / "metrics.json").unlink()
    assert any("artefacts" in n for n in failing_checks(run))


def test_detects_manifest_disagreement(tmp_path, registry_dir):
    run = make_run(tmp_path, registry_dir, metrics_over={
        "dataset": {"manifest_sha256": "d" * 64}
    })
    assert "manifest" in " ".join(failing_checks(run))


def test_detects_smoke_run(tmp_path, registry_dir):
    run = make_run(tmp_path, registry_dir, metrics_over={"smoke": True})
    assert any("smoke" in n for n in failing_checks(run))


def test_allows_smoke_run_when_flagged(tmp_path, registry_dir):
    run = make_run(tmp_path, registry_dir, metrics_over={"smoke": True})
    assert failing_checks(run, allow_smoke=True) == []


def test_detects_split_leakage(tmp_path, registry_dir):
    """If the eval split is the train split, the run is invalid -- the audit must
    notice rather than trusting the config."""
    run = make_run(tmp_path, registry_dir)
    cfg = json.loads((run / "config.json").read_text())
    cfg["dataset"]["eval_split"] = "train"
    (run / "config.json").write_text(json.dumps(cfg))
    names = " ".join(failing_checks(run))
    assert "overlap" in names or "differs" in names


def test_detects_flat_loss_curve(tmp_path, registry_dir):
    run = make_run(tmp_path, registry_dir)
    (run / "metrics.csv").write_text("epoch,phase,loss\n0,train,5.0\n1,train,5.0\n")
    assert "loss" in " ".join(failing_checks(run))


def test_detects_nonfinite_metrics(tmp_path, registry_dir):
    """A NaN in a results table is the classic silent failure."""
    run = make_run(tmp_path, registry_dir, metrics_over={
        "metrics": {"EN": float("nan"), "AG": 0.004}
    })
    assert "finite" in " ".join(failing_checks(run))


def test_detects_corrupt_checkpoint(tmp_path, registry_dir):
    run = make_run(tmp_path, registry_dir)
    (run / "checkpoints" / "best.pt").write_bytes(b"not a checkpoint")
    assert "best.pt" in " ".join(failing_checks(run))


def test_detects_missing_benchmark_latency(tmp_path, registry_dir):
    run = make_run(tmp_path, registry_dir)
    (run / "benchmark.json").write_text(json.dumps({"params": 123, "latency": []}))
    names = " ".join(failing_checks(run))
    assert "latency" in names


def test_detects_tiny_evaluation_coverage(tmp_path, registry_dir):
    run = make_run(tmp_path, registry_dir, metrics_over={
        "evaluation": {"num_images": 1, "elapsed_s": 1.0}
    })
    assert "coverage" in " ".join(failing_checks(run))


def test_audit_exit_code_is_nonzero_on_failure(tmp_path, registry_dir):
    run = make_run(tmp_path, registry_dir)
    (run / "benchmark.json").unlink()
    assert ar.audit(run, allow_smoke=False) == 1