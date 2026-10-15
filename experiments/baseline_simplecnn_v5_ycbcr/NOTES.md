# SUPERSEDED: trained on a corrupted (bf16) loss

**Status:** completed, **failed the acceptance gate** (5 of 14 checks). It is
excluded from every table. Superseded by `experiments/baseline_simplecnn_v6/`.

## Symptom

* Near-binary, inverted output: sky and road white, people black. 29% of pixels
  were saturated, and correlation with the visible luminance was −0.61.
* Training loss 7.19 -> 6.85 (−4.8%) over 60 epochs. The intensity term rose
  from 0.38 to 0.43, worse than returning the visible image unchanged.
* `results/checks/baseline_simplecnn_v5_ycbcr_samples.png`.

## Root cause

`Trainer.train_epoch` called the loss inside `torch.autocast(bfloat16)`. Casting
the inputs with `.float()` did not prevent autocast from re-casting the loss's
conv2d ops (Sobel, SSIM Gaussian) to bf16, which corrupted their gradients.

The model in eval mode and train mode was equally inverted, so this was not a
BatchNorm issue.

CPU A/B test, same model, data and seed, 300 steps:

| loss computed | intensity L1 | corr(f, vi) | corr(f, ir) | saturated |
|---|---|---|---|---|
| fp32 | 0.032 | +0.93 | +0.50 | 0.7% |
| inside bf16 autocast | NaN | – | – | – |
| bf16 forward + fp32 loss (fix) | 0.032 | +0.93 | +0.50 | 0.7% |

CPU and CUDA bf16 kernels differ, so CUDA gave finite-but-wrong gradients rather
than NaN. v6 on the GPU is the confirmation.

The same bug affected v3 and v4 and is the likely contributor to v4's false
colour.
