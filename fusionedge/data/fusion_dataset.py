"""The real-data dataset: an (ir, vi) pair list read from a registered dataset.

Construction is gated on a provenance registry entry (data/registry/<id>.json)
produced by scripts/download_data.py. That gate is deliberate: it makes it
impossible to train on a folder of images whose provenance nobody wrote down,
and it means every run automatically carries the dataset id and manifest hash.

Split resolution order:
    explicit `split` argument  >  `train` (dataset's own)  >  `val`
The MSRS dataset ships its own train/ and test/ directories; the MFNet fallback
uses our own hash-based train/val split. `val` is used as the held-out split for
MSRS because that is what train/ vs test/ means there in practice for fusion
(no MSRS method is trained on test/), and the registry records the split counts
so the substitution is visible in every result row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.seed import worker_init_fn
from .transforms import paired_random_crop, paired_random_hflip, paired_random_vflip, to_tensor_image

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = REPO_ROOT / "data" / "registry"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class DatasetNotRegistered(RuntimeError):
    """Raised when no provenance record exists for the requested dataset."""


def load_registry(dataset_id: str, registry_dir: Path | None = None) -> dict[str, Any]:
    """Read and sanity-check a dataset registry entry."""
    registry_dir = registry_dir or REGISTRY_DIR
    path = registry_dir / f"{dataset_id}.json"
    if not path.exists():
        available = sorted(p.stem for p in registry_dir.glob("*.json")) if registry_dir.exists() else []
        raise DatasetNotRegistered(
            f"No provenance record for dataset '{dataset_id}' at {path}. "
            f"Registered datasets: {available or 'none'}. "
            f"Run: python scripts/download_data.py --dataset auto"
        )
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("status") != "ok":
        raise DatasetNotRegistered(
            f"Dataset '{dataset_id}' has status={rec.get('status')!r}, not 'ok'. "
            f"Notes: {rec.get('notes')}"
        )
    if not rec.get("manifest_sha256"):
        raise DatasetNotRegistered(
            f"Dataset '{dataset_id}' registry has no manifest_sha256; refusing to "
            f"use a dataset whose contents cannot be pinned."
        )
    return rec


def resolve_dataset_root(dataset_id: str, registry_dir: Path | None = None) -> Path:
    rec = load_registry(dataset_id, registry_dir)
    root = Path(rec["root"])
    if not root.exists():
        raise DatasetNotRegistered(
            f"Dataset '{dataset_id}' registry points at {root}, which does not exist. "
            f"Re-run: python scripts/download_data.py --dataset {dataset_id.lower()} --force"
        )
    return root


def _list_images(d: Path) -> list[str]:
    return sorted(p.name for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def read_image(path: Path) -> np.ndarray:
    """Read an image as HWC uint8 BGR-independent (cv2 gives BGR -> convert).

    Using cv2 for both modalities guarantees both go through one decoder, so a
    grayscale PNG and a 3-channel PNG of the same scene are handled by the same
    arithmetic downstream.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    if img.ndim == 3 and img.shape[2] == 4:  # drop alpha
        img = img[:, :, :3]
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _to_ir_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC/2D image -> (1,H,W) float32 in [0,1] (luminance if 3-channel)."""
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return to_tensor_image(img)


def _to_vi_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC image -> (3,H,W) float32 in [0,1] (grayscale source replicated)."""
    t = to_tensor_image(img)
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    elif t.shape[0] == 4:
        t = t[:3]
    return t


@dataclass(frozen=True)
class PairRef:
    pair_id: str
    ir_path: str
    vi_path: str


class FusionPairDataset(Dataset):
    """(ir, vi) pairs from a registered on-disk dataset.

    Args:
        root: dataset root (as resolved from the registry).
        split: 'train', 'val', or any subdirectory name under root.
        crop: if set, random-crop to this size (train only); None = full image.
        augment: random flips (train only).
        limit: keep only the first N pairs (smoke tests and quick eval).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        *,
        crop: int | None = None,
        augment: bool = False,
        limit: int | None = None,
        pairs: list[str] | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.crop = crop
        self.augment = augment

        ir_dir = self.root / split / "ir"
        vi_dir = self.root / split / "vi"
        if not (ir_dir.is_dir() and vi_dir.is_dir()):
            raise FileNotFoundError(
                f"expected paired dirs {ir_dir} and {vi_dir}. "
                f"Available under {self.root}: "
                f"{sorted(p.name for p in self.root.iterdir() if p.is_dir())}"
            )

        if pairs is None:
            ir_names, vi_names = set(_list_images(ir_dir)), set(_list_images(vi_dir))
            common = sorted(ir_names & vi_names)
            missing = sorted((ir_names ^ vi_names))
            if missing:
                raise ValueError(
                    f"unpaired files in split '{split}': {len(missing)} e.g. {missing[:5]}"
                )
        else:
            common = list(pairs)

        if limit is not None:
            common = common[:limit]

        self.pairs = [
            PairRef(pid, str(ir_dir / pid), str(vi_dir / pid)) for pid in common
        ]
        if not self.pairs:
            raise RuntimeError(f"split '{split}' under {self.root} contains no pairs")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ref = self.pairs[idx]
        ir = _to_ir_tensor(read_image(Path(ref.ir_path)))
        vi = _to_vi_tensor(read_image(Path(ref.vi_path)))

        # A few MSRS/MFNet pairs differ by a pixel in one axis; crop to the
        # common size rather than silently broadcasting into a shape error.
        if ir.shape[-2:] != vi.shape[-2:]:
            h = min(ir.shape[-2], vi.shape[-2])
            w = min(ir.shape[-1], vi.shape[-1])
            ir, vi = ir[:, :h, :w], vi[:, :h, :w]

        if self.crop:
            ir, vi = paired_random_crop(ir, vi, self.crop)
        if self.augment:
            ir, vi = paired_random_hflip(ir, vi)
            ir, vi = paired_random_vflip(ir, vi)

        return {"ir": ir, "vi": vi, "pair_id": ref.pair_id}


def build_dataloaders(
    dataset_id: str,
    *,
    batch_size: int = 8,
    num_workers: int = 4,
    crop: int | None = 128,
    augment: bool = True,
    train_split: str = "train",
    eval_split: str = "val",
    pin_memory: bool = True,
    limit_train: int | None = None,
    limit_eval: int | None = None,
    persistent_workers: bool | None = None,
) -> tuple[DataLoader, DataLoader, dict[str, Any]]:
    """Build train/eval loaders plus the registry record describing the data.

    batch_size is a *config* knob rather than a constant: the RTX 4070 Laptop in
    this project's environment has 8 GB of VRAM, so crop size and batch size
    interact and both are exposed.
    """
    rec = load_registry(dataset_id)
    root = Path(rec["root"])

    train_set = FusionPairDataset(
        root, train_split, crop=crop, augment=augment, limit=limit_train
    )
    eval_set = FusionPairDataset(root, eval_split, crop=None, augment=False, limit=limit_eval)

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=len(train_set) > batch_size,
        worker_init_fn=worker_init_fn(None),
        generator=torch.Generator().manual_seed(0),
        persistent_workers=persistent_workers,
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=1,  # full 480x640 images; batching here buys nothing and costs VRAM
        shuffle=False,
        num_workers=max(0, min(2, num_workers)),
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn(None),
        persistent_workers=False,
    )
    return train_loader, eval_loader, rec