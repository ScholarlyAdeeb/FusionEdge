# FAILED RUN — do not use these numbers

**Status: INVALID. Superseded by `experiments/baseline_simplecnn_v3_bf16/`.**

Second failure of the same baseline, from a different cause. Preserved for the
same reason as the first: the evidence is the point.

---

## Symptom

* Training loss flat at ~7.49 for 5 epochs (7.4899, 7.4693, 7.5001, 7.4928, 7.4945)
* `loss nan` from epoch 6 step 50 onward
* The run then **hung in validation** (GPU at 0%, no progress for 5+ minutes) and
  was killed by hand
* `RuntimeWarning: invalid value encountered in cast` in
  `fusionedge/engine/evaluate.py:38` — the NaN propagated into the uint8
  conversion of fused images
* `metrics.json` was never written

## Root cause: float16 autocast, not the model

Continued from this run's own epoch-5 checkpoint for 200 steps under each
precision:

| precision | loss change | GradScaler scale | time |
|---|---|---|---|
| **fp16** | **+3.62%** (got worse) | **0 — collapsed** | 37.5s |
| **bf16** | **−21.39%** | 1 (no scaler needed) | 35.1s |
| **fp32** | **−43.06%** | 1 | 35.8s |

`float16` has a maximum finite value of 65504 and a narrow usable range for
gradients. Activations leaving the network head measured |x| ≈ 27, and gradients
overflowed often enough that the `GradScaler` drove its scale to **zero**. At
scale 0 the scaler skips any step containing an inf, so *nearly every optimizer
step was silently discarded*.

The critical detail: **nothing raised.** The loss curve looked like ordinary
noise, the run appeared to progress, and a metrics table would have been written
at the end containing numbers from a model that had barely trained.

## Why the earlier sigmoid fix looked ineffective

The first correction (`clamp` → `sigmoid`) halved the loss in the first epoch
and then appeared to do nothing. It was working — fp32 from the identical
checkpoint improves 43% over 200 steps. AMP was blocking the entire effect.

## Correction

`fusionedge/engine/train.py` now uses **bfloat16** by default:

* bfloat16 has fp32's 8-bit exponent, so it does not overflow at these
  magnitudes, and **no GradScaler is needed or used** (enabling one there would
  mask an overflow rather than surface it)
* two guards now make silent failure impossible:
  1. a `GradScaler` whose scale collapses below 1e-8 raises immediately,
  2. a streak of ≥25 consecutive non-finite loss/gradient values raises,
  3. per-step finiteness checks on both loss and every parameter gradient
* `evaluate.py` uses the same dtype policy and returns fp32 tensors so metric
  precision is not silently reduced

`configs/default.yaml` documents the choice and records this measurement.

Cost: bf16 is essentially as fast as fp16 here (35.1s vs 37.5s), so no
throughput penalty. fp32 is the same speed in this measurement and remains
available via `amp: false`.

## Files preserved

| File | Contents |
|---|---|
| `metrics.csv` | 5 epochs of the flat curve |
| `log.txt` | including the `loss nan` lines at epoch 6 |
| `config.yaml` / `config.json` | the fp16 configuration |
| `checkpoints/last.pt` | weights at epoch 5, used for the comparison above |