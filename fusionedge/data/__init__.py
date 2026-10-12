"""Dataset loaders.

Tensor convention (used everywhere in this project):

    ir : float32 tensor (B, 1, H, W) in [0, 1]
    vi : float32 tensor (B, 3, H, W) in [0, 1]
    fu : float32 tensor (B, 3, H, W) in [0, 1]   -- fused RGB output
    pair_id : str

Infrared is kept as a single channel rather than replicated to three. Replication
is the convention in much of the fusion literature, but it triples the stem's
input width for a channel that carries no extra information, and it makes the
"what does the model actually see" question harder to answer in an interview.

Loading policy:
  * Files are read lazily per item and converted with a single code path, so a
    `.png` and a `.jpg` of the same scene produce identical pixels.
  * IR images that are stored as 3-channel files are collapsed to luminance; IR
    images stored as 1-channel files are expanded on read only where a 3-channel
    source is required (never in the network).
  * Training uses random crops + flips (geometric only -- no photometric jitter,
    which would fight the intensity loss); evaluation uses the full image with
    reflection padding to a multiple of 8 and crops back afterwards.
"""

from .fusion_dataset import (
    FusionPairDataset,
    build_dataloaders,
    load_registry,
    resolve_dataset_root,
)
from .synthetic import SyntheticPairDataset
from .transforms import paired_random_crop, pad_to_multiple, to_tensor_image

__all__ = [
    "FusionPairDataset",
    "SyntheticPairDataset",
    "build_dataloaders",
    "load_registry",
    "resolve_dataset_root",
    "paired_random_crop",
    "pad_to_multiple",
    "to_tensor_image",
]