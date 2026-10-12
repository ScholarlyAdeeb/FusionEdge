# FAILED RUN — do not use these numbers

**Status: INVALID. Superseded by `experiments/baseline_simplecnn_v2_sigmoid/`.**

Preserved deliberately. A project that quietly deletes its failed runs is a
project whose remaining artefacts cannot be trusted, because there is no way to
tell whether a number came from a working configuration or a broken one.

---

## What happened

`configs/baseline.yaml` → SimpleCNN baseline, trained on MSRS train (1083 pairs),
60 epochs planned, stopped by hand after epoch 4.

## Symptom

Training loss was flat and then rising:

| epoch | loss | intensity | gradient | ssim | edge | color | epoch time |
|---|---|---|---|---|---|---|---|
| 0 | 16.451 | 0.244 | 1.228 | 0.912 | 0.024 | 0.372 | 18.8 s |
| 1 | 16.431 | 0.245 | 1.226 | 0.911 | 0.024 | 0.372 | 11.4 s |
| 2 | 16.462 | 0.245 | 1.229 | 0.911 | 0.025 | 0.370 | 9.6 s |

(epoch times decrease, so this is not a throughput problem)

Validation after epoch 1: EN 6.53, MS-SSIM_ir 0.4824 — not meaningful, from a
model whose loss never moved.

## Root cause

`SimpleCNNBaseline.forward` ended in `torch.clamp(x, 0.0, 1.0)`.

`clamp` has **zero gradient** outside `[0,1]`. Measured on real MSRS crops from
this run's own checkpoint:

```
output range              [0.0000, 1.0000]      <- pinned at both ends
pixels at lower clamp     47.1%
pixels at upper clamp      4.9%
pixels saturated total    52.0%                 <- contributing no gradient
```

Confirmed with 60 Adam steps on a single real batch:

| output head | loss | saturated pixels |
|---|---|---|
| `clamp(0,1)` | 13.296 → **13.296** | 58.3% |
| `sigmoid` | 8.350 → **8.181** | 0.1% |

The clamp was the entire difference.

## Why this matters for the comparison

**FusionEdge never had this bug.** Its output is `clamp(vi + delta)`, which starts
at exactly the visible image, so ~0% of pixels sit on a boundary. Leaving the
baseline bug in place would have produced a comparison rigged in FusionEdge's
favour — FusionEdge would have beaten a baseline that could not learn at all, for
reasons having nothing to do with the architectural contributions under study.

## Correction

`fusionedge/models/baseline.py` now ends in `torch.sigmoid(x)`. Nothing else
changed: same stem, same 4 residual separable blocks, same width 32, same loss,
same optimizer, same LR schedule, same seed, same data pipeline, same epochs.

The fix makes the **baseline stronger**. It does not make FusionEdge stronger.

## Files preserved here

| File | Contents |
|---|---|
| `metrics.csv` | 4 epochs of the flat loss curve (the evidence) |
| `log.txt` | full run transcript |
| `config.yaml` / `config.json` | the configuration that produced the failure |
| `checkpoints/last.pt` | weights from the broken run, for reproducing the measurement |

## Reproducing the measurement

```python
import torch
from fusionedge.utils.registry import build_model
from fusionedge.utils.checkpoint import load_checkpoint

m = build_model("baseline", {"width": 32, "depth": 4})
load_checkpoint("last.pt", m, map_location="cpu")   # the broken checkpoint
m.eval()
# saturation measurement and the 60-step comparison are in
# tests/test_output_head.py
```