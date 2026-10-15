# Comparison protocol (fixed 2026-10-13, before any v4/FusionEdge result existed)

This protocol is written down *before* the runs it governs, so it cannot be
tuned to their outcome. Every model below — the baseline, FusionEdge, and each
ablation — is trained and evaluated with it unchanged.

## Data and splits

| Role | Source | Pairs | Used for |
|---|---|---|---|
| fit | MSRS `train/` minus holdout | 975 | gradient updates |
| holdout | MSRS `train/`, 10% by sha1(`0:<id>`) rank | 108 | per-epoch monitoring, `best.pt` (informational) |
| test | MSRS `test/` (called `val` in code) | 361 | **evaluated once, after training** |

* Dataset manifest: `e5f03de8da4a87cb…` (`data/registry/MSRS.json`), upstream commit `5fa66e33c5ba`.
* The exact holdout ids and SHA-256 are written to `<run>/holdout_manifest.json`; every run must show the same hash.
* The MSRS test split influences no choice: no early stopping, no checkpoint selection, no hyperparameter change.
* Known limitation: holdout and fit come from the same recording pass, so near-duplicate frames can occur across them. This makes holdout numbers optimistic. That's acceptable because holdout numbers are never reported as results.

## Training (identical for every model; `configs/default.yaml`)

Adam, lr 1e-4, cosine schedule to 1e-6 stepped per optimizer step, 60 epochs, batch 8, 128×128 random crops + flips, grad-clip 1.0, bf16 autocast (no GradScaler), seed 1234, deterministic kernels, loss weights `intensity 8 / gradient 10 / ssim 2 / edge 1 / color 1`.

## Output space (added 2026-10-14, before any FusionEdge run)

Every model fuses **luminance only**: it produces Y, and Cb/Cr are taken unchanged from the visible image (BT.601, `fusionedge/models/colour.py`). This is standard on MSRS.

It was introduced after baseline v4, which predicted RGB, passed every automated check and still visibly failed with false colour. FusionEdge's residual head `vi + delta` was already luminance-only (proved by `test_residual_head_is_already_luminance_only_fusion`). Without this change, the comparison would have measured output parameterisation, not fusion. The direct-output ablation (`ablation_direct_y_head`) also predicts Y.

The acceptance gate (`scripts/check_run.py`) gained two checks at the same time:
- luminance must correlate with IR at least 0.05 more than the visible image does;
- chroma must stay within 0.03 of the visible image.

## FusionEdge output head (fixed 2026-10-15, before any full FusionEdge run)

The original residual head, `clamp(vi + delta)` with a zero-initialised `delta`, starts **exactly** at the visible image and the pre-flight showed it never leaves. On the 256-image smoke subset (40 epochs, 1,280 steps):
- with that zero-init head, mean |dY| = 0.0005 and corr(fused, IR) = +0.020, against +0.014 for the visible image itself;
- with a 1e-3 random head init, mean |dY| = 0.0011 and corr = +0.028.

The cause is the loss, not the optimiser. At the identity, the intensity, gradient and edge terms (L1/ReLU against `max(IR, vi)` targets) all sit on their kinks at most pixels. Any delta that isn't spatially selective is pushed back, and the head's gradient flips sign from step to step.

The loss does reward IR, though. On the same batches it scores 0.879 for the identity, 0.996 for baseline v6, and **0.750** for "visible Y raised to IR where IR is brighter".

The head is therefore now `residual_mode: inject`:

    g = sigmoid(head(x));   Y = Y_vi + g * relu(IR - Y_vi);   out = recolour(Y, vi)

It has the same parameters as before (the same 3×3 conv) and still keeps the visible chroma. It can only *add* luminance where IR is brighter than the visible image, and is smooth in the parameters at its start (g = 0.5). Smoke check with the same 40 epochs:
- corr(fused, IR) = +0.154, mean |dY| = 0.013;
- the gate moved from a flat 0.5 to mean 0.76, SD 0.26;
- holdout loss fell from 1.035 at initialisation to 0.954.

`inject` is the model default, so every single-component ablation inherits it. `add` remains selectable for reproducibility only. The loss is unchanged, so baseline v6 stays valid. No test-split number informed this change.

Known consequence: FusionEdge can only brighten, never darken, the visible luminance. That suits MSRS, where thermal targets are bright, but it is a modelling assumption.

## Attribution control: baseline backbone + inject head (recorded 2026-10-15, before it was run)

**Why:** `fusionedge_main` reached its best holdout MS-SSIM_ir at epoch 4, and its holdout metrics barely moved afterwards. The untrained inject head (the fixed rule g = 0.5) already gets most of the way: MS-SSIM_ir 0.446 against 0.475 trained, on the 108-image holdout. FusionEdge also runs 2.8× slower than baseline v6 (47.0 ms vs 16.7 ms at 480×640) with 22,339 parameters against 13,089. So it is open whether the FusionEdge backbone (cross-modal exchange, SE, edge-guided gate) adds anything beyond the head.

**Run:** `baseline_simplecnn_inject` (`configs/baseline_inject.yaml`).
- It uses the baseline-v6 backbone unchanged, with `output_space: inject`. That head is `colour.inject_ir`, the same function FusionEdge's head now calls, and it is zero-initialised the same way.
- At initialisation it therefore produces exactly FusionEdge's starting output (unit test `test_baseline_inject_starts_at_the_same_rule_as_fusionedge`).
- It has 13,089 parameters, the same as v6.
- Everything else matches baseline v6 and FusionEdge: data, holdout, loss, schedule, seed, gates, and `last.pt` as the reported checkpoint.

**How it will be read (fixed now):**
- *v6 → baseline+inject*: the effect of the **head**, with the backbone fixed.
- *baseline+inject → FusionEdge*: the effect of the **FusionEdge backbone**, with the head fixed. This is the comparison that decides whether that backbone earns its cost.
- If baseline+inject matches or beats FusionEdge on most of the 8 holdout metrics, the FusionEdge backbone is **not supported**. FusionEdge's advantages over v6 would then be credited to the head.
- These are single-seed results and are labelled so. A difference smaller than the seed spread is reported as "no clear difference" once the 3-seed repeat exists.
- The comparison uses holdout metrics only. The test split is still evaluated exactly once, after training, and informs nothing.

## Seed repeats (recorded 2026-10-15, before any repeat was run)

**Why:** under the attribution rule above, `baseline_simplecnn_inject` and `fusionedge_main` split the 8 holdout metrics 4–4 on seed 1234. The rule never defined "matches", so it is undecided on one seed.

**Runs:** two more seeds each, launched one at a time with `scripts/win/protocol_run.ps1`, through the same tests, pre-flight, audit and gate. All other settings are unchanged.

| Run | Config | `seed` | `data.loader_seed` |
|---|---|---|---|
| `baseline_simplecnn_inject` (done) | `baseline_inject.yaml` | 1234 | 0 (legacy) |
| `baseline_simplecnn_inject_s1235` | `baseline_inject_s1235.yaml` | 1235 | 1235 |
| `baseline_simplecnn_inject_s1236` | `baseline_inject_s1236.yaml` | 1236 | 1236 |
| `fusionedge_main` (done) | `fusionedge.yaml` | 1234 | 0 (legacy) |
| `fusionedge_main_s1235` | `fusionedge_s1235.yaml` | 1235 | 1235 |
| `fusionedge_main_s1236` | `fusionedge_s1236.yaml` | 1236 | 1236 |

Baseline v6 is **not** repeated. The head effect (v6 → baseline+inject) therefore stays a single-seed result and is labelled so.

**Data-loader seed fix (found while preparing these runs):** the train loader's generator was hard-coded to 0. It sets the shuffle order and also every worker's augmentation seed. With `num_workers > 0`, as in training, seeds 1234 and 1235 produced *identical* first batches: same order, crops and flips. A seed repeat would then vary only the weight init and understate run-to-run spread. `data.loader_seed` now sets that generator. Its default of 0 reproduces every earlier run exactly. The repeats set it equal to their seed, and with 2 workers their batches now differ.

**Decision rule (fixed now; the seed-1234 values are already known, so the rule is defined by seed spread rather than by those values):**
- **Per metric M** (the 8 holdout metrics; `last.pt`; same 108 holdout images; `scripts/analyse_fusion.py`):
  - compute each model's mean and sample SD over its 3 seeds;
  - compute the paired differences d_s = (baseline+inject − FusionEdge) for seeds 1234, 1235 and 1236.
- **Clear difference on M:** |mean difference| > max(SD_baseline+inject, SD_FusionEdge) **and** all three d_s have the same sign. Anything else is "no clear difference".
- **Backbone not supported:** baseline+inject *matches or beats* FusionEdge (that is, FusionEdge is **not** clearly better) on ≥ 5 of the 8 metrics.
- **Backbone supported:** FusionEdge is **clearly better** on ≥ 5 of the 8 metrics.
- **Otherwise mixed:** reported metric by metric, with no overall claim.
- The cost difference (22,339 vs 13,089 parameters; 47.0 vs 16.9 ms at 480×640) is reported next to the verdict in every case.
- The test split is still evaluated once per run and informs nothing.

## Loss precision (fixed 2026-10-14, before any FusionEdge run)

The loss is computed in genuine fp32, **outside** the bf16 autocast block (`fp32_loss` in `fusionedge/losses/fusion_loss.py`).

Until this date it ran inside the block, where autocast re-cast its conv2d ops (Sobel, SSIM) to bf16 and corrupted their gradients. Baselines v3–v5 trained on that corrupted loss; v5 converged to an inverted, near-binary image. In a CPU A/B test with the same model, data and 300 steps:
- loss inside autocast: NaN;
- loss in fp32: intensity L1 0.032, correlation +0.93 with the visible image and +0.50 with IR.

## fp32 means fp32 (fixed 2026-10-14, before any v6 run)

Deterministic mode (`seed_everything`, used for training and evaluation) now disables TF32 for cuDNN convolutions and cuBLAS matmuls. PyTorch enables TF32 by default on this GPU, so every "fp32" conv, including the Sobel/SSIM convs in `fp32_loss`, actually ran with a 10-bit mantissa. The v6 pre-flight caught this: the same checkpoint gave a 1.9e-3 max output gap between contiguous and channels_last layouts with TF32, and 2.0e-6 without. Latency (`benchmark.py`) is still measured on the fast path with TF32 on, the same as before.

## Reported checkpoint

`last.pt` for every model, meaning the same fixed budget for all. `best.pt` (holdout MS-SSIM_ir) is saved for inspection only.

## Metrics (test split, full 480×640, one pass)

EN, AG, SF, MI, SSIM_ir/vi, MS-SSIM_ir/vi as implemented in `fusionedge/metrics/metrics.py`.

* **MS-SSIM** was corrected to the standard product-of-scales definition on 2026-10-13. Numbers from earlier runs are not comparable.
* **AG and SF** are computed on [0,1] luma. AG has no 1/√2 factor, and SF is √2× the usual definition. They are therefore **not comparable to published MSRS tables**; they are only for comparing models within this project. Unit tests pin these definitions.
* No metric is a ground-truth accuracy: MSRS has no fused reference. A model "wins" on a metric, not overall, and every table reports all metrics.

## Claims

A claim that FusionEdge beats the baseline on metric M requires a higher mean on the test split under this protocol. Single-seed results are labelled as such. If time allows, a 3-seed repeat (seeds 1234, 1235, 1236) is used to report mean ± std, and differences smaller than the seed spread are reported as "no clear difference".

## Gate before FusionEdge

The baseline run must pass `scripts/check_run.py` (convergence, strict reload of `last.pt`, determinism, output-quality checks) and `scripts/audit_run.py`. Its sample grid is inspected visually.

Every model (baseline, FusionEdge, each ablation) passes through the same `check_run.py` and `audit_run.py`, and its sample grid is inspected.

### Convergence rule changed (2026-10-15, before any full FusionEdge run; applies to every model)

The old rule was "mean train loss of the last 5 epochs ≥ 10% below epoch 0" (pre-flight: "epoch 3 < 0.8 × epoch 1"). It is replaced by `fusionedge/engine/learning_check.py`:

- **Measurement:** the training loss (fp32, eval mode) on fixed, un-augmented holdout images, for the trained weights and for a fresh model of the same architecture seeded with the run's seed.
- **Pass condition:** the loss must fall by at least **1%** and by more than 10× the run-to-run evaluation noise.
- **Images:** the gate uses the first 16 holdout ids; the pre-flight uses its 8 smoke holdout images.

Why the old rule went: a relative drop from epoch 0 measures the starting point as much as the learning. FusionEdge's zero-initialised head starts close to a good solution. A 20% smoke drop would need a loss of 0.693, below even the hand-built "visible Y raised to IR" output (0.750). Per-epoch train loss also moves about ±0.03 with the random crops. The old ratio is still reported as information.

Calibration, using only finished runs and smoke runs, before any full FusionEdge run (drop vs fresh init, 16 holdout images):

| Run | Drop vs fresh init | Verdict |
|---|---|---|
| baseline v6 | +76.4% | pass |
| baseline v4 | +39.4% | pass |
| FusionEdge `inject`, 40-epoch smoke | +6.0% | pass |
| FusionEdge `inject`, 3-epoch pre-flight smoke | +4.1% | pass |
| FusionEdge `add`, stuck at identity (3 smoke runs) | −0.17% to +0.25% | fail |

Evaluation noise was exactly 0. Baseline v6 was re-checked under the new rule: it passes all 14 checks, and its holdout metrics are unchanged. The previous check outputs are kept in `results/checks/_archive/pre_gate_change_20261015/`.

### Pre-flight precision check A (2026-10-15)

"Smooth-term gradient matches fp64" now excludes pixels where the fused luminance equals the intensity target to within 1e-5. Those pixels sit on the L1 kink, where rounding flips the subgradient sign. A random-init model has almost none; the inject head has them wherever IR ≤ visible (58–78% of pixels). On the other pixels the fp32/fp64 cosine is 1.0000000.

A new check compares the gradient on the **parameters** (what the optimiser uses) from the fp32 and fp64 losses. Its cosine is 1.0000000, because the tied pixels do not depend on the parameters.

## Superseded runs

`_failed/baseline_clamp_bug`, `_failed/baseline_fp16_amp` and `baseline_simplecnn_v3_bf16` used test-split selection and/or the handicapped baseline head. They are preserved with notes and excluded from every table.
