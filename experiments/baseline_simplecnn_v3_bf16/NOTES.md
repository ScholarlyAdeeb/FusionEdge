# SUPERSEDED RUN — not part of any comparison

**Status: stopped deliberately on 2026-10-13 (~00:50 IST) during recovery, at
epoch ~3-4 of 60. Superseded by `experiments/baseline_simplecnn_v4/`.**
The exact stop time and PID are in `results/recovery/launch_*.txt`.

Artefacts are left exactly as the run wrote them. They are not deleted because
they show that the bf16 fix trained (loss 6.72 at epoch 1 versus a flat 7.49
under fp16).

## Why it was stopped rather than allowed to finish

1. **Selection on the test split.** `eval_split: val` is the MSRS `test/`
   directory, and `best.pt` was selected on it every 2 epochs. Any number
   reported from this run would be optimistically biased.
2. **Baseline handicap.** `SimpleCNNBaseline` applied `LeakyReLU(0.2)` *before*
   the sigmoid. A dark pixel (MSRS is mostly night scenes) then needs a
   pre-activation of about -20 to reach 0.02. FusionEdge has no such
   constraint, so the comparison would have favoured FusionEdge.
3. **Non-standard selection metric.** `MS_SSIM` returned the mean of per-level
   products instead of the standard product of per-scale contrast terms.
4. **LR scheduler** was stepped before `optimizer.step()` and was not saved in
   checkpoints, so this run could not be resumed faithfully either.

All four are fixed in v4: a train-holdout for selection, the test split
evaluated once on `last.pt`, `use_relu_head: false`, standard MS-SSIM, and the
scheduler state checkpointed. Regression tests are in
`tests/test_recovery_fixes.py`.
