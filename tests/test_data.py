"""Data layer: shapes, ranges, alignment, split integrity, synthetic isolation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from fusionedge.data import FusionPairDataset, build_dataloaders, load_registry, pad_to_multiple
from fusionedge.data.fusion_dataset import DatasetNotRegistered, _to_ir_tensor, _to_vi_tensor, read_image
from fusionedge.data.synthetic import SyntheticDatasetNotPermitted, SyntheticPairDataset, make_pair
from fusionedge.data.transforms import paired_random_crop, paired_random_hflip, to_tensor_image


# --------------------------------------------------------------------------- #
# synthetic isolation
# --------------------------------------------------------------------------- #
def test_synthetic_dataset_requires_explicit_opt_in():
    """Guardrail: synthetic data must never be obtainable by accident."""
    with pytest.raises(SyntheticDatasetNotPermitted):
        SyntheticPairDataset(length=2)
    with pytest.raises(SyntheticDatasetNotPermitted):
        SyntheticPairDataset(length=2, allow_synthetic=False)
    ds = SyntheticPairDataset(length=2, allow_synthetic=True)
    assert len(ds) == 2


def test_synthetic_items_are_flagged():
    """Every synthetic sample self-identifies, so a stray one is detectable."""
    ds = SyntheticPairDataset(length=2, allow_synthetic=True)
    assert all(item["synthetic"] is True for item in ds)


def test_synthetic_dataset_iteration_terminates():
    """Regression: __getitem__ must raise IndexError past the end, otherwise the
    sequence protocol iterates forever and `for item in ds` hangs the suite."""
    ds = SyntheticPairDataset(length=3, allow_synthetic=True)
    assert len(list(ds)) == 3
    with pytest.raises(IndexError):
        _ = ds[3]


def test_synthetic_pairs_are_aligned_and_in_range():
    ir, vi = make_pair(seed=7, height=48, width=64)
    assert ir.shape == (1, 48, 64)
    assert vi.shape == (3, 48, 64)
    assert ir.dtype == torch.float32 and vi.dtype == torch.float32
    assert 0.0 <= ir.min() and ir.max() <= 1.0
    assert 0.0 <= vi.min() and vi.max() <= 1.0


def test_synthetic_is_deterministic():
    a_ir, a_vi = make_pair(seed=3)
    b_ir, b_vi = make_pair(seed=3)
    assert torch.equal(a_ir, b_ir) and torch.equal(a_vi, b_vi)


def test_ir_carries_structure_absent_from_visible():
    """The synthetic IR must be smooth/blobby while VI is textured -- otherwise
    tests would not be exercising the cross-modal behaviour they claim to."""
    ir, vi = make_pair(seed=11, height=64, width=64)
    def roughness(x):
        return (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    assert roughness(vi) > roughness(ir)


# --------------------------------------------------------------------------- #
# transforms
# --------------------------------------------------------------------------- #
def test_to_tensor_image_handles_gray_and_rgb():
    gray = np.zeros((8, 8), np.uint8)
    t = to_tensor_image(gray)
    assert t.shape == (1, 8, 8)
    rgb = np.zeros((8, 8, 3), np.uint8)
    assert to_tensor_image(rgb).shape == (3, 8, 8)


def test_to_tensor_image_scales_uint8_to_unit_range():
    img = np.full((4, 4, 3), 255, np.uint8)
    assert to_tensor_image(img).max() == pytest.approx(1.0)
    img = np.zeros((4, 4, 3), np.uint8)
    assert to_tensor_image(img).max() == pytest.approx(0.0)


def test_paired_random_crop_uses_the_same_origin():
    ir = torch.arange(64 * 64, dtype=torch.float32).view(1, 64, 64)
    vi = ir * 2.0
    torch.manual_seed(0)
    a, b = paired_random_crop(ir, vi, 32)
    assert a.shape == (1, 32, 32) and b.shape == (1, 32, 32)
    assert torch.equal(a * 2.0, b), "crop origin must be identical across modalities"


def test_paired_crop_pads_images_smaller_than_the_crop():
    ir = torch.rand(1, 20, 20)
    vi = torch.rand(3, 20, 20)
    a, b = paired_random_crop(ir, vi, 32)
    assert a.shape[-2:] == (32, 32) and b.shape[-2:] == (32, 32)


def test_paired_hflip_flips_both_or_neither():
    ir = torch.rand(1, 16, 16)
    vi = torch.rand(3, 16, 16)
    for seed in range(6):
        torch.manual_seed(seed)
        np.random.seed(seed)
        a, b = paired_random_hflip(ir, vi)
        assert torch.equal(a, torch.flip(ir, dims=[-1])) == torch.equal(
            b, torch.flip(vi, dims=[-1])
        )


def test_pad_to_multiple_hits_target_and_preserves_content():
    for (h, w) in [(480, 640), (37, 53), (128, 128)]:
        x = torch.rand(3, h, w)
        p = pad_to_multiple(x, 8)
        assert p.shape[-2] % 8 == 0 and p.shape[-1] % 8 == 0
        assert p.shape[-2] >= h and p.shape[-1] >= w
        # the original image must survive intact in the top-left corner
        assert torch.allclose(p[..., :h, :w], x)


def test_pad_is_noop_when_already_aligned():
    x = torch.rand(3, 64, 64)
    p = pad_to_multiple(x, 8)
    assert p.shape == x.shape
    assert torch.equal(p, x)
    assert p is x, "an aligned input should be returned untouched, not copied"


def test_pad_handles_tiny_images():
    """Reflect padding needs pad < dim; the helper must not blow up on 4x4."""
    x = torch.rand(1, 4, 4)
    p = pad_to_multiple(x, 8)
    assert p.shape[-2] % 8 == 0 and p.shape[-1] % 8 == 0


# --------------------------------------------------------------------------- #
# image readers
# --------------------------------------------------------------------------- #
def test_ir_reader_collapses_colour_to_single_channel(tmp_path: Path):
    p = tmp_path / "x.png"
    Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(p)
    assert _to_ir_tensor(read_image(p)).shape == (1, 16, 16)


def test_vi_reader_expands_greyscale_to_three_channels(tmp_path: Path):
    p = tmp_path / "g.png"
    Image.fromarray(np.zeros((16, 16), np.uint8)).save(p)
    t = _to_vi_tensor(read_image(p))
    assert t.shape == (3, 16, 16)
    assert torch.equal(t[0], t[1]) and torch.equal(t[1], t[2])


def test_reader_rejects_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "nope.png")


def test_alpha_channel_is_dropped(tmp_path: Path):
    p = tmp_path / "a.png"
    Image.fromarray(np.zeros((16, 16, 4), np.uint8)).save(p)
    assert _to_vi_tensor(read_image(p)).shape == (3, 16, 16)


# --------------------------------------------------------------------------- #
# a real on-disk layout, written by the test and registered on the fly
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_dataset(tmp_path: Path) -> Path:
    """Build a tiny dataset that looks structurally exactly like a real one."""
    for split, n in (("train", 10), ("val", 4)):
        for mod in ("ir", "vi"):
            d = tmp_path / "canon" / split / mod
            d.mkdir(parents=True)
            for i in range(n):
                g = torch.Generator().manual_seed(hash((split, mod, i)) % (2**31))
                arr = (torch.rand(32, 40, generator=g) * 255).to(torch.uint8).numpy()
                if mod == "vi":
                    arr = np.repeat(arr[:, :, None], 3, axis=2)
                Image.fromarray(arr).save(d / f"{i:04d}.png")
    return tmp_path


def test_dataset_returns_expected_shapes_and_types(fake_dataset: Path):
    ds = FusionPairDataset(fake_dataset / "canon", "train")
    item = ds[0]
    assert item["ir"].shape == (1, 32, 40)
    assert item["vi"].shape == (3, 32, 40)
    assert isinstance(item["pair_id"], str)
    assert item["ir"].dtype == torch.float32
    assert 0.0 <= item["ir"].min() and item["ir"].max() <= 1.0
    assert len(ds) == 10


def test_dataset_crop_and_augment_change_shapes_not_pairing(fake_dataset: Path):
    ds = FusionPairDataset(fake_dataset / "canon", "train", crop=16, augment=True)
    item = ds[0]
    assert item["ir"].shape == (1, 16, 16)
    assert item["vi"].shape == (3, 16, 16)


def test_dataset_limit(fake_dataset: Path):
    assert len(FusionPairDataset(fake_dataset / "canon", "train", limit=3)) == 3


def test_dataset_rejects_unpaired_files(fake_dataset: Path):
    """A stray file with no counterpart must fail loudly, not be silently skipped."""
    (fake_dataset / "canon" / "train" / "ir" / "orphan.png").write_bytes(b"not-an-image")
    with pytest.raises(ValueError, match="unpaired"):
        FusionPairDataset(fake_dataset / "canon", "train")


def test_dataset_missing_split_is_a_clear_error(fake_dataset: Path):
    with pytest.raises(FileNotFoundError):
        FusionPairDataset(fake_dataset / "canon", "test")


# --------------------------------------------------------------------------- #
# provenance gate
# --------------------------------------------------------------------------- #
def test_unregistered_dataset_is_refused(tmp_path: Path):
    """The provenance gate: no registry entry -> no dataset, even if files exist."""
    with pytest.raises(DatasetNotRegistered, match="No provenance record"):
        load_registry("NotADataset", registry_dir=tmp_path)


def test_failed_registry_status_is_refused(tmp_path: Path):
    (tmp_path / "Broken.json").write_text(
        json.dumps({"dataset_id": "Broken", "status": "unavailable", "notes": "quota"})
    )
    with pytest.raises(DatasetNotRegistered, match="status='unavailable'"):
        load_registry("Broken", registry_dir=tmp_path)


def test_registry_without_manifest_hash_is_refused(tmp_path: Path):
    (tmp_path / "NoHash.json").write_text(
        json.dumps({"dataset_id": "NoHash", "status": "ok", "root": str(tmp_path)})
    )
    with pytest.raises(DatasetNotRegistered, match="manifest_sha256"):
        load_registry("NoHash", registry_dir=tmp_path)


def test_registry_pointing_at_missing_root_is_refused(tmp_path: Path):
    (tmp_path / "Ghost.json").write_text(
        json.dumps(
            {
                "dataset_id": "Ghost",
                "status": "ok",
                "root": str(tmp_path / "does_not_exist"),
                "manifest_sha256": "abc",
            }
        )
    )
    from fusionedge.data.fusion_dataset import resolve_dataset_root

    with pytest.raises(DatasetNotRegistered, match="does not exist"):
        resolve_dataset_root("Ghost", registry_dir=tmp_path)


def test_build_dataloaders_end_to_end(fake_dataset: Path, monkeypatch, tmp_path: Path):
    """Full loader construction against a temporary registry."""
    reg_dir = tmp_path / "registry"
    reg_dir.mkdir()
    (reg_dir / "Fake.json").write_text(
        json.dumps(
            {
                "dataset_id": "Fake",
                "status": "ok",
                "root": str(fake_dataset / "canon"),
                "manifest_sha256": "deadbeef",
                "splits": {"train": 10, "val": 4},
            }
        )
    )
    monkeypatch.setattr("fusionedge.data.fusion_dataset.REGISTRY_DIR", reg_dir)

    train, val, rec = build_dataloaders(
        "Fake", batch_size=2, num_workers=0, crop=16, limit_train=4
    )
    batch = next(iter(train))
    assert batch["ir"].shape == (2, 1, 16, 16)
    assert batch["vi"].shape == (2, 3, 16, 16)
    assert rec["dataset_id"] == "Fake"
    assert rec["manifest_sha256"] == "deadbeef"
    # eval loader is full-image and batch 1
    vbatch = next(iter(val))
    assert vbatch["ir"].shape == (1, 1, 32, 40)