# Overnight report — 2026-10-15 (deadline 10:00 IST)

Final update: 08:10 IST. Everything planned for tonight is complete, except the items listed in §7. Nothing has been committed or pushed.

## 1. Completed and verified

| item | artifact | verification |
|---|---|---|
| Baseline + inject experiment (protocol written before the run) | `experiments/baseline_simplecnn_inject/`, `configs/baseline_inject.yaml`, `docs/PROTOCOL.md` "Attribution control" | audit 17/17, gate 14/14, 60/60 epochs |
| Seed repeats ×4 (protocol and decision rule written before the runs) | `experiments/{baseline_simplecnn_inject,fusionedge_main}_s{1235,1236}/`, launcher log `results/seed_repeats/` | each: tests 378 passed, pre-flight 35/35, 60/60 epochs, audit 17/17, gate 14/14 |
| Pre-registered seed verdict | `results/analysis/20261015_075430/seed_rule.{json,md}` (`scripts/seed_rule.py`) | rule applied mechanically, exactly as written in `docs/PROTOCOL.md` "Seed repeats" |
| Holdout analyses | `results/analysis/20261015_031806/` (3 runs, 8 panels); `results/analysis/20261015_075430/` (7 runs, 4 panels) | 108 holdout images, `last.pt`, fp32 |
| Inference demo | `scripts/infer.py --run <experiment> --fp32` | output pixel-identical (max diff 0 levels) to the evaluation forward pass for all 3 seed-1234 runs on holdout pair `00096D` |
| README | `README.md` | documented commands checked (§5) |
| Results report | `docs/RESULTS.md` (generated: `scripts/make_results.py --analysis results/analysis/20261015_075430`) | every number read from artifacts |
| Architecture | `docs/ARCHITECTURE.md` | shapes and parameter counts from forward hooks at 480×640 |
| Experiment inventory | `docs/EXPERIMENTS.md`, `results/inventory/inventory_*.json` (`scripts/inventory.py`) | read from run directories and check files |
| Representative visuals | `docs/assets/showcase_holdout.jpg`, `docs/assets/gate_holdout.jpg` (`scripts/make_showcase.py`) | holdout images only; inspected |
| Data-loader seed bug fixed before the repeats | `data.loader_seed` in `fusionedge/data/fusion_dataset.py` and `fusionedge/engine/train.py`; test `test_loader_seed_drives_the_train_generator` | before the fix, seeds 1234 and 1235 produced identical batches with workers; after it they differ. Default 0 reproduces all earlier runs exactly |

## 2. Experiment IDs and reported checkpoints (all accepted: audit 17/17, gate 14/14)

| run id | model / head | seed (loader seed) | reported checkpoint |
|---|---|---|---|
| `baseline_simplecnn_v6` | baseline / Y | 1234 (0) | `experiments/baseline_simplecnn_v6/checkpoints/last.pt` |
| `baseline_simplecnn_inject` | baseline / inject | 1234 (0) | `experiments/baseline_simplecnn_inject/checkpoints/last.pt` |
| `baseline_simplecnn_inject_s1235` | baseline / inject | 1235 (1235) | `experiments/baseline_simplecnn_inject_s1235/checkpoints/last.pt` |
| `baseline_simplecnn_inject_s1236` | baseline / inject | 1236 (1236) | `experiments/baseline_simplecnn_inject_s1236/checkpoints/last.pt` |
| `fusionedge_main` | FusionEdge / inject | 1234 (0) | `experiments/fusionedge_main/checkpoints/last.pt` |
| `fusionedge_main_s1235` | FusionEdge / inject | 1235 (1235) | `experiments/fusionedge_main_s1235/checkpoints/last.pt` |
| `fusionedge_main_s1236` | FusionEdge / inject | 1236 (1236) | `experiments/fusionedge_main_s1236/checkpoints/last.pt` |

Checkpoints are git-ignored. Superseded and failed runs (v3, v4, v5, `_failed/baseline_clamp_bug`, `_failed/baseline_fp16_amp`) are kept with their notes and listed in `docs/EXPERIMENTS.md`. `experiments/baseline_simplecnn_v2_sigmoid`, which git shows as deleted, was **moved** to `experiments/_failed/baseline_fp16_amp`: same metrics.csv and config, differing only in line endings. It was moved before this session.

## 3. Measured results

**Seed verdict (holdout, 3 seeds, pre-registered rule): FusionEdge backbone NOT supported.**

| metric | baseline + inject, mean ± SD | FusionEdge, mean ± SD | paired diff (1234, 1235, 1236) | result |
|---|---|---|---|---|
| EN | 6.7212 ± 0.0005 | 6.7198 ± 0.0021 | +0.0004, +0.0032, +0.0006 | no clear difference |
| AG | 0.02371 ± 0.00002 | 0.02374 ± 0.00004 | −0.00006, −0.00001, −0.00002 | no clear difference |
| SF | 0.06415 ± 0.00002 | 0.06426 ± 0.00006 | −0.00016, −0.00006, −0.00011 | FusionEdge |
| MI | 6.2652 ± 0.0087 | 6.1346 ± 0.0488 | +0.0947, +0.1803, +0.1170 | baseline + inject |
| SSIM_ir | 0.4924 ± 0.0004 | 0.4818 ± 0.0023 | +0.0083, +0.0128, +0.0109 | baseline + inject |
| SSIM_vi | 0.9063 ± 0.0005 | 0.9177 ± 0.0038 | −0.0079, −0.0152, −0.0113 | FusionEdge |
| MS-SSIM_ir | 0.4857 ± 0.0004 | 0.4701 ± 0.0042 | +0.0111, +0.0192, +0.0165 | baseline + inject |
| MS-SSIM_vi | 0.9256 ± 0.0003 | 0.9350 ± 0.0031 | −0.0065, −0.0122, −0.0094 | FusionEdge |

- **Count:** FusionEdge is clearly better on 3 of 8 metrics and baseline + inject on 3 of 8. Baseline + inject therefore matches or beats FusionEdge on 5 of 8, which meets the rule's threshold of 5. This is a **narrow** verdict.
- **The split is a trade-off:** FusionEdge wins the visible-similarity metrics and baseline + inject the infrared-similarity metrics.
- **Stability:** FusionEdge varies several times more across seeds. Its mean gate where IR is brighter is 0.67 / 0.59 / 0.59, against 0.81 / 0.80 / 0.79 for baseline + inject.

**Cost** (from each run's `benchmark.json`; RTX 4070 Laptop GPU, batch 1, fp32, median of 200 iterations):

| | params | latency 480×640 | peak VRAM 480×640 | optimiser wall time per run |
|---|---|---|---|---|
| baseline v6 | 13,089 | 16.7 ms | 199 MB | 43.9 min |
| baseline + inject (3 seeds) | 13,089 | 16.9 ms | 199 MB | 34.3 / 36.3 / 41.1 min |
| FusionEdge (3 seeds) | 22,339 | 47.0 / 47.0 / 46.9 ms | 454 MB | 45.2 / 47.0 / 47.5 min |

- `best.pt` (informational only) landed on epoch index 4 for all six inject-head runs, and 54 for v6.
- The seed-1234 holdout table for all three models, the test-split numbers, and the pixel-level behaviour (brightening and darkening fractions, chroma change, gate) are in `docs/RESULTS.md`.
- The test split (361 images) was evaluated once per run, automatically, and was **not** used for any decision.

## 4. Tests and audits

- **Tests:** the full suite (`python -m pytest -q`) ran 11 times since 04:07 IST, counting the 4 seed pipelines.
  - 10 runs: **378 passed, 0 failed, 0 skipped.**
  - 1 run (07:55 IST): **377 passed, 1 failed.** The failing test's name was not captured, because only the summary line was printed.
  - It did not recur in 6 subsequent full runs, or in 6 repeated runs of the six GPU-touching test files (118 tests each). It is recorded as an **unresolved intermittent failure** (§7), not hidden.
- **Audits:** `scripts/audit_run.py` gives 17/17 for all 7 accepted runs.
- **Acceptance gate:** `scripts/check_run.py` gives 14/14 for all 7 accepted runs, including the learning check (loss on 16 fixed holdout images ≥ 1% below a fresh init):
  - v6: +76.4%
  - baseline + inject: +8.8% / +8.8% / +8.9%
  - FusionEdge: +7.3% / +6.2% / +6.1%
- **Pre-flight:** 35/35 for every run launched tonight.

## 5. Commands (verified tonight unless noted)

```bash
python -m pytest -q
python scripts/infer.py --run experiments/fusionedge_main --fp32 --ir data/MSRS/canonical/train/ir/00096D.png --vi data/MSRS/canonical/train/vi/00096D.png --out results/demo/fusionedge_00096D.png
python scripts/audit_run.py experiments/fusionedge_main
python scripts/download_data.py --verify-only
python scripts/analyse_fusion.py experiments/baseline_simplecnn_inject experiments/baseline_simplecnn_inject_s1235 experiments/baseline_simplecnn_inject_s1236 experiments/fusionedge_main experiments/fusionedge_main_s1235 experiments/fusionedge_main_s1236 experiments/baseline_simplecnn_v6
python scripts/seed_rule.py results/analysis/<stamp>/summary.json
python scripts/make_results.py --analysis results/analysis/<stamp>
python scripts/make_showcase.py
python scripts/inventory.py
```

- **Training:** `scripts/win/protocol_run.ps1 -Run <id> -Cfg <config>` ran all five of tonight's training runs (via `seed_repeats.ps1` for four). `python scripts/run_experiment.py --config <config>` is the core it calls.
- **Not re-run tonight:** `scripts/check_run.py`, because it rewrites that run's results files; it ran inside every pipeline.

## 6. Decisions made autonomously

- **Seed repeats kept running** after the sprint brief arrived, because they finished by 07:46, well inside the deadline.
- **No ablations.** Seven configs at about 50 minutes each could not finish by 10:00. After the seed verdict, ablating FusionEdge components is also of limited value.
- **`--run` and `--fp32` added** to the inference CLI, so the demo always uses the reported `last.pt` at evaluation precision.
- **`results/**/pytest_*.txt` added to `.gitignore`.** Those logs contain the local Windows username in pytest temp paths. The files were not edited; every pipeline log keeps the pass/fail line.
- **Stale references fixed:** a dated note at the top of `docs/BLOCKERS.md`, and two docstring references to a missing `docs/EXPERIMENTS.md` section.
- **The near-tie in the seed verdict is reported as is.** No tolerance or threshold was changed after seeing results.

## 7. Unresolved, skipped, and why

- **Intermittent test failure:** 1 of 11 full-suite runs, test unidentified, not reproduced since. Next time, run with `-rf` so the name is printed. The launchers currently save the full pytest log, but those logs are now git-ignored, not deleted.
- **Ablations not run** (7 configs), as above.
- **Baseline v6 has one seed,** so the head effect (v6 → inject head) is single-seed.
- **Weak spread estimates:** three seeds give weak SD estimates, and the verdict sits exactly on the threshold (5 of 8).
- **No comparison with published methods.** AG and SF use project-specific definitions and are not comparable to published MSRS tables.
- **FusionEdge latency not profiled per module.**
- **Dashboard windows still open:** seven `scripts/monitor.py` dashboards (`cmd /k`), one per run, are still running. They are read-only, and the GPU is idle. Close them by hand.
- **Not fixed in artifacts:** the absolute repo path `D:\Projects\Sam` appears in run logs, config stamps and `data/registry/MSRS.json`, which was already committed before tonight.
- **Nothing committed or pushed.** The working tree has about 314 files to add (dry run): no checkpoints, datasets, credentials or pytest logs, and the largest file is 361 KB.

## 8. Recommended next step

Commit the current state after a quick review of `git status`.

Scientifically, the most informative next experiment is to **profile FusionEdge latency per module** and run **`ablation_avg_fusion`** (FusionEdge with its adaptive gate replaced by averaging). This is one run of about 50 minutes, and the gate is the component most directly implicated in FusionEdge's higher visible-similarity scores and its seed instability.

If the paper's claim is "lightweight fusion", the evidence tonight favours **baseline + inject head** as the model to present: 13,089 parameters, 16.9 ms, stable across seeds. Use FusionEdge as the documented negative result.

## Addendum — release review (2026-10-15, after the sprint)

- **Intermittent test failure: still unidentified.** The full suite was re-run 9 more times with `-rf`, 8 of them with JUnit output and 3 with all 20 CPU cores saturated. All 9 passed 378/378. That makes **1 failure in 20 full runs**, with the test name never captured. The launcher (`scripts/win/protocol_run.ps1`) now runs pytest with `-rf`, so a recurrence will be named in the saved log.
- **Recommended model made explicit** in README.md and docs/RESULTS.md: `baseline_simplecnn_inject`. FusionEdge is documented as a negative result. The README demo now uses the recommended model, and its output was verified pixel-identical to the evaluation path.
- **README setup corrected.** The pinned `torch==2.12.1+cu126` needs `--extra-index-url https://download.pytorch.org/whl/cu126`. A plain `pip install -r requirements.txt` would fail on PyPI. All 12 pinned packages match the installed environment. A fresh-clone install and download were **not** re-run, because that needs network access and a new environment.
- **Release hygiene:** three groups were excluded from git via `.gitignore`. They are kept on disk; nothing was deleted.
  - `experiments/_preflight/` (smoke-run scratch; verdicts are in `results/preflight/*.json`)
  - `results/*/preflight_2*.txt` (console copies; 5 contained the local username)
  - `results/tables.md` (stale table showing only the superseded v5 run)
- **Still disclosed, not rewritten (provenance):**
  - The absolute path `D:\Projects\Sam` appears in 47 to-be-added files (run logs, check and pre-flight records) and in 3 already-tracked files, including `data/registry/MSRS.json` (`root`).
  - The already-tracked `data/_mfnet_retry.log` contains the local username, in the path of a gdown cookie file; the path only, not cookie contents.
  - No remote is configured, so nothing has been pushed yet.
