"""The real-image inference path, and its refusal to fake a result.

The failure mode this file guards is specific and dangerous: an untrained
FusionEdge returns the visible image almost exactly, so running it produces a
clean, plausible-looking picture that contains no infrared information. Anyone
skimming a results folder would read that as a working fusion model. These tests
pin the guards against it.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from fusionedge.engine.infer import fuse_pair, is_trained, load_model
from fusionedge.utils.checkpoint import save_checkpoint
from fusionedge.utils.registry import build_model

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def real_pair_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Two directories of genuinely distinct image files on disk.

    These are *files*, not in-memory tensors, so this exercises the real decode
    path. Their content is noise: the point is that the plumbing works on real
    image files, not that fusion quality is good.
    """
    ir_dir, vi_dir = tmp_path / "ir", tmp_path / "vi"
    ir_dir.mkdir()
    vi_dir.mkdir()
    for i in range(3):
        seed = 100 + i
        ir = (np.random.default_rng(seed).random((64, 80)) * 255).astype(np.uint8)
        vi = (np.random.default_rng(seed + 50).random((64, 80, 3)) * 255).astype(np.uint8)
        Image.fromarray(ir).save(ir_dir / f"scene{i}.png")
        Image.fromarray(vi).save(vi_dir / f"scene{i}.png")
    return ir_dir, vi_dir


def _ckpt(path: Path, *, trained: bool) -> Path:
    model = build_model("fusionedge", {"width": 8, "depth": 1})
    if trained:
        # give the zero-init head real weights so the checkpoint looks trained
        torch.nn.init.normal_(model.head.weight, std=0.05)
    return save_checkpoint(path, model, epoch=5 if trained else 0)


# --------------------------------------------------------------------------- #
# trained-detection guard
# --------------------------------------------------------------------------- #
def test_untrained_checkpoint_is_detected(tmp_path: Path):
    ok, reason = is_trained(_ckpt(tmp_path / "u.pt", trained=False))
    assert ok is False
    assert "all-zero" in reason or "untrained" in reason


def test_trained_checkpoint_is_detected(tmp_path: Path):
    ok, reason = is_trained(_ckpt(tmp_path / "t.pt", trained=True))
    assert ok is True, reason


def test_missing_checkpoint_is_reported_not_raised(tmp_path: Path):
    ok, reason = is_trained(tmp_path / "nope.pt")
    assert ok is False
    assert "does not exist" in reason


def test_non_model_file_is_reported(tmp_path: Path):
    p = tmp_path / "junk.pt"
    p.write_bytes(b"definitely not a checkpoint")
    ok, reason = is_trained(p)
    assert ok is False
    assert reason


# --------------------------------------------------------------------------- #
# inference on real image files
# --------------------------------------------------------------------------- #
def test_fuse_pair_accepts_file_paths(real_pair_dirs):
    ir_dir, vi_dir = real_pair_dirs
    model = build_model("fusionedge", {"width": 8, "depth": 1}).eval()
    out = fuse_pair(model, ir_dir / "scene0.png", vi_dir / "scene0.png", device="cpu", amp=False)
    assert out.shape == (64, 80, 3)
    assert out.dtype == np.uint8


def test_fuse_pair_accepts_arrays(real_pair_dirs):
    ir_dir, vi_dir = real_pair_dirs
    ir = np.asarray(Image.open(ir_dir / "scene0.png"))
    vi = np.asarray(Image.open(vi_dir / "scene0.png"))
    model = build_model("fusionedge", {"width": 8, "depth": 1}).eval()
    out = fuse_pair(model, ir, vi, device="cpu", amp=False)
    assert out.shape == (64, 80, 3)


def test_fuse_pair_on_real_files_is_bit_identical_across_runs(real_pair_dirs):
    """Determinism on the real decode path."""
    ir_dir, vi_dir = real_pair_dirs
    model = build_model("fusionedge", {"width": 8, "depth": 1}).eval()
    torch.nn.init.normal_(model.head.weight, std=0.05)
    a = fuse_pair(model, ir_dir / "scene1.png", vi_dir / "scene1.png", device="cpu", amp=False)
    b = fuse_pair(model, ir_dir / "scene1.png", vi_dir / "scene1.png", device="cpu", amp=False)
    assert np.array_equal(a, b)


def test_untrained_model_reproduces_the_visible_image(real_pair_dirs):
    """The exact behaviour that makes unguarded inference misleading: with the
    zero-initialised head, the 'fusion' is just the visible input."""
    ir_dir, vi_dir = real_pair_dirs
    model = build_model("fusionedge", {"width": 8, "depth": 1, "residual_mode": "add"}).eval()
    fused = fuse_pair(model, ir_dir / "scene0.png", vi_dir / "scene0.png",
                      device="cpu", amp=False)
    visible = np.asarray(Image.open(vi_dir / "scene0.png").convert("RGB"))
    mean_abs_diff = np.abs(fused.astype(int) - visible.astype(int)).mean()
    assert mean_abs_diff < 2.0, f"untrained output should equal visible (diff={mean_abs_diff})"


# --------------------------------------------------------------------------- #
# CLI guards
# --------------------------------------------------------------------------- #
def _run(script: str, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *args],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=300,
    )


def test_infer_cli_refuses_untrained_weights(real_pair_dirs, tmp_path: Path):
    ir_dir, vi_dir = real_pair_dirs
    ckpt = _ckpt(tmp_path / "u.pt", trained=False)
    r = _run("infer.py", [
        "--ir", str(ir_dir / "scene0.png"), "--vi", str(vi_dir / "scene0.png"),
        "--weights", str(ckpt), "--model", "fusionedge",
        "--model-cfg", '{"width":8,"depth":1}',
        "--out", str(tmp_path / "out.png"), "--device", "cpu",
    ])
    assert r.returncode == 2
    assert "REFUSING" in r.stderr
    assert "untrained" in r.stderr.lower()


def test_infer_cli_allows_override_for_plumbing(real_pair_dirs, tmp_path: Path):
    ir_dir, vi_dir = real_pair_dirs
    ckpt = _ckpt(tmp_path / "u.pt", trained=False)
    r = _run("infer.py", [
        "--ir", str(ir_dir / "scene0.png"), "--vi", str(vi_dir / "scene0.png"),
        "--weights", str(ckpt), "--model", "fusionedge",
        "--model-cfg", '{"width":8,"depth":1}',
        "--out", str(tmp_path / "out.png"), "--device", "cpu", "--allow-untrained",
    ])
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "out.png").exists()
    # and it must self-report that it contributed nothing
    assert "no infrared contribution" in r.stdout


def test_infer_pairs_requires_real_directories(tmp_path: Path):
    r = _run("infer_pairs.py", [
        "--ir-dir", str(tmp_path / "does_not_exist"),
        "--vi-dir", str(tmp_path / "also_missing"),
        "--weights", str(tmp_path / "x.pt"), "--device", "cpu",
    ])
    assert r.returncode == 2
    assert "will not generate substitute data" in r.stderr


def test_infer_pairs_labels_output_as_non_benchmark(real_pair_dirs, tmp_path: Path):
    ir_dir, vi_dir = real_pair_dirs
    ckpt = _ckpt(tmp_path / "u.pt", trained=False)
    out = tmp_path / "qual"
    r = _run("infer_pairs.py", [
        "--ir-dir", str(ir_dir), "--vi-dir", str(vi_dir), "--weights", str(ckpt),
        "--model", "fusionedge", "--model-cfg", '{"width":8,"depth":1}',
        "--out", str(out), "--device", "cpu", "--allow-untrained", "--grids", "1",
    ])
    assert r.returncode == 0, r.stderr
    meta = json.loads((out / "inference_meta.json").read_text())
    assert meta["dataset_id"].startswith("UNREGISTERED")
    assert meta["weights_trained"] is False
    assert "NOT a benchmark result" in meta["purpose"]
    assert len(list((out / "grids").glob("*.png"))) == 1
    rows = list(csv.DictReader((out / "inference_metrics.csv").open()))
    assert len(rows) == 3
    assert all(r["weights_trained"] == "0" for r in rows)
    assert "not a registered held-out split" in r.stdout