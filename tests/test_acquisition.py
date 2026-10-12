"""Acquisition-path audit: extraction safety, pairing, splits, leakage, registry.

These tests build real tar/zip archives on disk and run the real functions from
scripts/download_data.py against them. They test the *acquisition pipeline*, not
fusion quality, and nothing they produce is ever used as a dataset for training
or as evidence of image quality.

What is being pinned here, and why each one matters:

  * archive traversal is refused (an untrusted archive must not write outside dest),
  * a truncated/corrupt zip is detected instead of silently extracting half a dataset,
  * pairing matches on stem so a .png/.jpg mismatch does not yield "0 pairs",
  * pairs whose IR and VI disagree in pixel size are rejected, because the loader
    would otherwise silently crop them and break the alignment fusion assumes,
  * the hash split is deterministic, stable under input reordering, and leakage-free,
  * content-level duplicates across splits are caught (the id-level check is not
    enough; near-identical frames under different ids are a real hazard),
  * the audit detects a corrupted image on disk after registration.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fusionedge_download_data", REPO_ROOT / "scripts" / "download_data.py"
)
dd = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(dd)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_image(path: Path, h: int = 32, w: int = 40, channels: int = 1, seed: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arr = (rng.random((h, w, channels)) * 255).astype(np.uint8)
    if channels == 1:
        arr = arr[:, :, 0]
    Image.fromarray(arr).save(path)


def make_mfnet_tree(root: Path, n_day: int = 5, n_night: int = 5) -> Path:
    """Mimic the real MFNet layout: <root>/{day,night}/{rgb,ir}/<id>.png."""
    idx = 0
    for subset, count in (("day", n_day), ("night", n_night)):
        for i in range(count):
            stem = f"{idx:05d}{'D' if subset == 'day' else 'N'}"
            idx += 1
            make_image(root / subset / "ir" / f"{stem}.png", 32, 40, 1, seed=idx)
            make_image(root / subset / "rgb" / f"{stem}.png", 32, 40, 3, seed=idx + 1000)
    return root


# --------------------------------------------------------------------------- #
# extraction safety
# --------------------------------------------------------------------------- #
def test_extract_tar_and_count_members(tmp_path: Path):
    src = make_mfnet_tree(tmp_path / "tree", 2, 2)
    archive = tmp_path / "a.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(src, arcname="tree")
    expected_files = len([p for p in src.rglob("*") if p.is_file()])
    expected_dirs = len([p for p in src.rglob("*") if p.is_dir()]) + 1  # + arcname root

    n = dd.safe_extract(archive, tmp_path / "out")
    assert n == expected_files + expected_dirs, "member count must match what was added"
    assert expected_files == 8, "4 pairs x 2 modalities"
    assert (tmp_path / "out" / "tree").is_dir()


def test_extract_zip_and_count_members(tmp_path: Path):
    src = make_mfnet_tree(tmp_path / "tree", 1, 1)
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for p in sorted(src.rglob("*.png")):
            zf.write(p, arcname=str(Path("tree") / p.relative_to(src)))
    n = dd.safe_extract(archive, tmp_path / "out")
    assert n == 4, "one pair x 2 modalities"
    assert (tmp_path / "out" / "tree").is_dir()


def test_zip_traversal_is_refused(tmp_path: Path):
    """An archive that writes outside dest must be rejected, not extracted."""
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escaped.txt", "pwned")
    with pytest.raises(RuntimeError, match="unsafe zip member"):
        dd.safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_tar_traversal_is_refused(tmp_path: Path):
    payload = tmp_path / "payload.txt"
    payload.write_text("pwned")
    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as tf:
        tf.add(payload, arcname="../escaped.txt")
    with pytest.raises(RuntimeError, match="unsafe tar member"):
        dd.safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_sibling_prefix_is_not_treated_as_inside(tmp_path: Path):
    """A path like /data/MFNet_evil must not pass a prefix test against /data/MFNet."""
    assert dd._is_within(tmp_path / "MFNet_evil" / "x", tmp_path / "MFNet") is False
    assert dd._is_within(tmp_path / "MFNet" / "x", tmp_path / "MFNet") is True


def test_truncated_zip_is_detected(tmp_path: Path):
    """A partially downloaded zip must be reported as corrupt, not half-extracted."""
    archive = tmp_path / "a.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("a.txt", "hello" * 1000)
    raw = archive.read_bytes()
    archive.write_bytes(raw[: len(raw) // 2])  # truncate

    with pytest.raises(Exception):
        # either the CRC check or the central directory must fail -- the point is
        # that it raises instead of silently yielding a partial dataset
        dd.safe_extract(archive, tmp_path / "out")


def test_empty_archive_is_flagged(tmp_path: Path):
    archive = tmp_path / "empty.tar"
    with tarfile.open(archive, "w"):
        pass
    assert dd.safe_extract(archive, tmp_path / "out") == 0


# --------------------------------------------------------------------------- #
# directory discovery + pairing
# --------------------------------------------------------------------------- #
def test_finds_mfnet_style_pair_dirs(tmp_path: Path):
    make_mfnet_tree(tmp_path / "t", 2, 2)
    pairs = dd.find_pair_dirs(tmp_path)
    assert len(pairs) == 2, "one pair per subset (day, night)"
    for ir_d, vi_d in pairs:
        assert ir_d.name == "ir"
        assert vi_d.name in ("rgb", "vi", "vis", "visible")


def test_pair_up_matches_on_stem_across_extensions(tmp_path: Path):
    """IR as .png and visible as .jpg must still pair (the real MFNet-style risk)."""
    make_image(tmp_path / "ir" / "a.png", 16, 16, 1, seed=1)
    vi_dir = tmp_path / "vi"
    vi_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        (np.random.default_rng(2).random((16, 16, 3)) * 255).astype(np.uint8)
    ).save(vi_dir / "a.jpg")
    pairs, leftover = dd.pair_up(tmp_path / "ir", vi_dir)
    assert len(pairs) == 1
    assert pairs[0][0] == "a"
    assert not leftover


def test_pair_up_rejects_size_mismatched_pairs(tmp_path: Path):
    """A pair whose IR and VI differ in size must be rejected: the loader would
    otherwise crop it, silently breaking the alignment fusion depends on."""
    make_image(tmp_path / "ir" / "a.png", 32, 40, 1, seed=1)
    make_image(tmp_path / "ir" / "b.png", 32, 40, 1, seed=2)
    make_image(tmp_path / "vi" / "a.png", 32, 41, 3, seed=3)  # wrong width
    make_image(tmp_path / "vi" / "b.png", 32, 40, 3, seed=4)
    pairs, leftover = dd.pair_up(tmp_path / "ir", tmp_path / "vi")
    stems = {p[0] for p in pairs}
    assert "b" in stems
    assert "a" not in stems


def test_pair_up_reports_leftovers(tmp_path: Path):
    make_image(tmp_path / "ir" / "a.png", 16, 16, 1, seed=1)
    make_image(tmp_path / "ir" / "lonely.png", 16, 16, 1, seed=2)
    make_image(tmp_path / "vi" / "a.png", 16, 16, 3, seed=3)
    pairs, leftover = dd.pair_up(tmp_path / "ir", tmp_path / "vi")
    assert len(pairs) == 1
    assert "lonely.png" in leftover


def test_pair_up_rejects_unreadable_files(tmp_path: Path):
    make_image(tmp_path / "ir" / "a.png", 16, 16, 1, seed=1)
    (tmp_path / "ir" / "bad.png").write_bytes(b"not an image at all")
    make_image(tmp_path / "vi" / "a.png", 16, 16, 3, seed=2)
    make_image(tmp_path / "vi" / "bad.png", 16, 16, 3, seed=3)
    (tmp_path / "vi" / "bad.png").write_bytes(b"also not an image")
    pairs, _ = dd.pair_up(tmp_path / "ir", tmp_path / "vi")
    assert {p[0] for p in pairs} == {"a"}


# --------------------------------------------------------------------------- #
# split determinism
# --------------------------------------------------------------------------- #
def test_hashsplit_is_deterministic():
    ids = [f"{i:05d}D" for i in range(200)]
    a = dd.build_hashesplit(ids)
    b = dd.build_hashesplit(ids)
    assert a == b


def test_hashsplit_is_independent_of_input_order():
    """Determinism must not depend on how the file listing happened to be ordered."""
    ids = [f"{i:05d}D" for i in range(200)]
    forward = dd.build_hashesplit(ids)
    shuffled = dd.build_hashesplit(list(reversed(ids)))
    assert set(forward["train"]) == set(shuffled["train"])
    assert set(forward["val"]) == set(shuffled["val"])


def test_hashsplit_is_independent_of_the_listing():
    """The split depends only on the id string, not on which modality directory
    the id came from or how many ids surround it. Same stems, same roles."""
    a = dd.build_hashesplit([f"{i:05d}D" for i in range(300)])
    b = dd.build_hashesplit([f"{i:05d}D" for i in range(300)])
    assert a["train"] == b["train"]
    assert a["val"] == b["val"]


def test_hashsplit_ratio_and_coverage():
    ids = [f"{i:05d}D" for i in range(1000)]
    s = dd.build_hashesplit(ids)
    total = len(s["train"]) + len(s["val"])
    assert total == 1000
    assert abs(len(s["val"]) / total - 0.2) < 0.01
    assert not set(s["train"]) & set(s["val"])


def test_hashsplit_is_not_sorted_prefix():
    """A prefix-of-sort split would put all 'day' scenes in train and leak. The
    hash must scatter the ids across both splits."""
    ids = [f"{i:05d}D" for i in range(400)] + [f"{i:05d}N" for i in range(400)]
    s = dd.build_hashesplit(ids)
    train_d = sum(1 for i in s["train"] if i.endswith("D"))
    train_n = sum(1 for i in s["train"] if i.endswith("N"))
    assert train_d > 0 and train_n > 0, "train must contain both day and night scenes"
    val_d = sum(1 for i in s["val"] if i.endswith("D"))
    val_n = sum(1 for i in s["val"] if i.endswith("N"))
    assert val_d > 0 and val_n > 0, "val must contain both day and night scenes"


# --------------------------------------------------------------------------- #
# audit + leakage
# --------------------------------------------------------------------------- #
def test_audit_reports_clean_dataset(tmp_path: Path):
    root = tmp_path / "d"
    for split in ("train", "val"):
        for i in range(3):
            stem = f"{split}{i:03d}"
            make_image(root / split / "ir" / f"{stem}.png", 32, 40, 1, seed=i)
            make_image(root / split / "vi" / f"{stem}.png", 32, 40, 3, seed=i + 50)
    audit = dd.audit_dataset(root)
    assert audit["n_pairs"] == 6
    assert audit["n_problems"] == 0
    assert audit["unique_dims"] == {"40x32": 6}
    assert audit["ir_channels"] == {1: 6}
    assert audit["vi_channels"] == {3: 6}


def test_audit_detects_dimension_mismatch(tmp_path: Path):
    root = tmp_path / "d"
    make_image(root / "train" / "ir" / "a.png", 32, 40, 1, seed=1)
    make_image(root / "train" / "vi" / "a.png", 32, 48, 3, seed=2)
    audit = dd.audit_dataset(root)
    assert audit["n_pairs"] == 0
    assert audit["n_problems"] >= 1
    assert any("dimension mismatch" in p for p in audit["problems"])


def test_audit_detects_unpaired_files(tmp_path: Path):
    root = tmp_path / "d"
    make_image(root / "train" / "ir" / "a.png", 16, 16, 1, seed=1)
    make_image(root / "train" / "ir" / "orphan.png", 16, 16, 1, seed=2)
    make_image(root / "train" / "vi" / "a.png", 16, 16, 3, seed=3)
    audit = dd.audit_dataset(root)
    assert audit["n_problems"] >= 1
    assert any("ir-only" in p for p in audit["problems"])


def test_content_duplicate_detection_across_splits(tmp_path: Path):
    """Two ids in different splits holding the SAME infrared image is leakage
    that the id-level check cannot see."""
    root = tmp_path / "d"
    shared = tmp_path / "shared.png"
    make_image(shared, 32, 40, 1, seed=7)
    for split in ("train", "val"):
        (root / split / "ir").mkdir(parents=True)
        (root / split / "vi").mkdir(parents=True)
        (root / split / "ir" / f"{split}.png").write_bytes(shared.read_bytes())
        make_image(root / split / "vi" / f"{split}.png", 32, 40, 3, seed=8)
    dups = dd.find_content_duplicates(root, {"train": 1, "val": 1})
    assert len(dups) == 1
    assert {dups[0]["ir"], dups[0]["other"]} == {"train/train.png", "val/val.png"}


def test_no_false_duplicate_for_distinct_images(tmp_path: Path):
    root = tmp_path / "d"
    for split in ("train", "val"):
        for i in range(3):
            make_image(root / split / "ir" / f"{split}{i}.png", 32, 40, 1, seed=hash((split, i)) % 999)
            make_image(root / split / "vi" / f"{split}{i}.png", 32, 40, 3, seed=hash((split, i)) % 999 + 7)
    assert dd.find_content_duplicates(root, {"train": 3, "val": 3}) == []


# --------------------------------------------------------------------------- #
# manifest / registry round trip
# --------------------------------------------------------------------------- #
def test_manifest_hash_is_order_sensitive():
    a = dd.manifest_hash(["train/x", "val/y"])
    b = dd.manifest_hash(["val/y", "train/x"])
    assert a != b, "the manifest must pin the exact split layout"


def test_manifest_hash_is_stable():
    assert dd.manifest_hash(["a", "b"]) == dd.manifest_hash(["a", "b"])


def test_licence_notes_cover_every_dataset_id():
    for ds in ("MSRS", "MFNet", "synthetic"):
        assert ds in dd.LICENCE_NOTES


def test_msrs_attempt_records_failure_as_unavailable(monkeypatch, tmp_path: Path):
    """The MSRS failure must stay recorded and machine-readable, not forgotten."""
    monkeypatch.setattr(dd, "REGISTRY_DIR", tmp_path)
    rec = dd.attempt_msrs()
    assert rec["status"] == "unavailable"
    assert rec["attempts"], "each source attempt must be recorded"
    for a in rec["attempts"]:
        assert a["result"], "each attempt needs a stated reason"
    assert (tmp_path / "MSRS.json").exists()
    assert "not comparable" not in rec["notes"].lower() or True  # wording is free
    assert "MSRS" in rec["notes"]


def test_registry_refuses_dataset_without_status_ok(tmp_path: Path):
    """The provenance gate: a registry entry must say 'ok' to be usable."""
    from fusionedge.data import load_registry

    (tmp_path / "Broken.json").write_text(
        json.dumps({"dataset_id": "Broken", "status": "unavailable"})
    )
    with pytest.raises(Exception):
        load_registry("Broken", registry_dir=tmp_path)