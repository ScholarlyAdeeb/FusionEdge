# SUPERSEDED: failed visual inspection (false colour)

**Status:** completed and mechanically valid, but **not a usable baseline**. Excluded
from every table (see the `SUPERSEDED` marker). Superseded by
`experiments/baseline_simplecnn_v5_ycbcr/`.

## What passed

`audit_run.py` passed all 17 checks. `check_run.py` passed all 12 checks it had
at the time: convergence (7.16 -> 4.75), strict reload, determinism, saturation
0.01%, contrast.

## What failed

The sample grid (`results/checks/baseline_simplecnn_v4_samples.png`) shows heavy
**false colour**: yellow foliage, green pedestrians and wheels, a grey-lavender
wash. Thermal targets did reach the output, but as *colour* rather than
luminance. The luminance correlation with IR was 0.056, versus 0.80 with visible.
Test EN was 3.96, versus 7.32 for simply returning the visible image on the
holdout.

## Cause

The model predicts RGB from scratch. The loss constrains luminance strongly
(intensity weight 8, gradient 10, where the gradient term takes the max over
channels) but chroma only weakly (colour weight 1). FusionEdge's residual head
`vi + delta` changes luminance only and inherits chroma exactly. Comparing the
two would therefore have measured output parameterisation, not fusion.

## Correction (v5)

Every model now fuses luminance only and takes Cb/Cr from the visible image,
the standard approach on MSRS (`fusionedge/models/colour.py`). The acceptance
gate gained two checks that v4 would have failed: IR correlation must exceed the
visible image's own IR correlation by 0.05, and chroma must stay within 0.03 of
the visible image.

## Addendum (2026-10-14, later)

v4 also trained with the loss inside bf16 autocast, which corrupted its gradients
(see `experiments/baseline_simplecnn_v5_ycbcr/NOTES.md`). That bug is a likely
contributor to the false colour. The switch to luminance-only output stands
regardless, because it removes the output-format confound between the models.
