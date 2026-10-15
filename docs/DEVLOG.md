# Dev log

Notes on what I ran, what broke, and what I decided, with the evidence for each. The dates are the real run dates. Rules for each experiment were written into [PROTOCOL.md](PROTOCOL.md) before the experiment ran.

## Where things stand

The two inject-head models were each trained with three seeds, and every run passed the automated audit (17/17) and acceptance gate (14/14).

| run | model / head | seed | checkpoint (not committed) |
|---|---|---|---|
| `baseline_simplecnn_v6` | baseline / Y head | 1234 | `experiments/baseline_simplecnn_v6/checkpoints/last.pt` |
| `baseline_simplecnn_inject` | baseline / inject | 1234 | `experiments/baseline_simplecnn_inject/checkpoints/last.pt` |
| `baseline_simplecnn_inject_s1235` | baseline / inject | 1235 | `…/baseline_simplecnn_inject_s1235/checkpoints/last.pt` |
| `baseline_simplecnn_inject_s1236` | baseline / inject | 1236 | `…/baseline_simplecnn_inject_s1236/checkpoints/last.pt` |
| `fusionedge_main` | FusionEdge / inject | 1234 | `experiments/fusionedge_main/checkpoints/last.pt` |
| `fusionedge_main_s1235` | FusionEdge / inject | 1235 | `…/fusionedge_main_s1235/checkpoints/last.pt` |
| `fusionedge_main_s1236` | FusionEdge / inject | 1236 | `…/fusionedge_main_s1236/checkpoints/last.pt` |

Older and failed runs (v3, v4, v5, `_failed/baseline_clamp_bug`, `_failed/baseline_fp16_amp`) are kept with notes in [EXPERIMENTS.md](EXPERIMENTS.md). The run originally named `baseline_simplecnn_v2_sigmoid` is the one now in `_failed/baseline_fp16_amp`; it was moved, not deleted.

## 15 Oct: does the FusionEdge backbone earn its cost?

Over three seeds, comparing baseline + inject against FusionEdge on the 108 holdout images:

| metric | baseline + inject, mean ± SD | FusionEdge, mean ± SD | difference per seed (1234, 1235, 1236) | result |
|---|---|---|---|---|
| EN | 6.7212 ± 0.0005 | 6.7198 ± 0.0021 | +0.0004, +0.0032, +0.0006 | no clear difference |
| AG | 0.02371 ± 0.00002 | 0.02374 ± 0.00004 | −0.00006, −0.00001, −0.00002 | no clear difference |
| SF | 0.06415 ± 0.00002 | 0.06426 ± 0.00006 | −0.00016, −0.00006, −0.00011 | FusionEdge |
| MI | 6.2652 ± 0.0087 | 6.1346 ± 0.0488 | +0.0947, +0.1803, +0.1170 | baseline + inject |
| SSIM_ir | 0.4924 ± 0.0004 | 0.4818 ± 0.0023 | +0.0083, +0.0128, +0.0109 | baseline + inject |
| SSIM_vi | 0.9063 ± 0.0005 | 0.9177 ± 0.0038 | −0.0079, −0.0152, −0.0113 | FusionEdge |
| MS-SSIM_ir | 0.4857 ± 0.0004 | 0.4701 ± 0.0042 | +0.0111, +0.0192, +0.0165 | baseline + inject |
| MS-SSIM_vi | 0.9256 ± 0.0003 | 0.9350 ± 0.0031 | −0.0065, −0.0122, −0.0094 | FusionEdge |

- **The rule:** before running the seeds I set it as follows. A difference counts only if the gap in means is bigger than the seed spread **and** all three seeds agree on the direction. The backbone is "not supported" if FusionEdge fails to win clearly on at least 5 of the 8 metrics.
- **The verdict:** FusionEdge won 3, baseline + inject won 3, and 2 were ties. So the verdict is "not supported", but only just, since 5 was the threshold.
- **It's a trade-off, not a loss:** FusionEdge keeps closer to the visible image, and the baseline + inject brings in more infrared. FusionEdge's gate settled at a different opening on each seed (0.67 / 0.59 / 0.59 where IR is brighter), against a steady 0.81 / 0.80 / 0.79 for the baseline. That's where its larger spread comes from.
- **Cost:** 22,339 parameters and 47.0 ms per frame, against 13,089 and 16.9 ms. Measured as the median of 200 batch-1 fp32 runs on an RTX 4070 Laptop GPU, with peak VRAM 454 MB vs 199 MB. One training run took 45–48 min for FusionEdge and 34–41 min for the baseline.

So the model I'd recommend is the baseline + inject head. The full tables are in [RESULTS.md](RESULTS.md).

## 15 Oct: bugs found while setting up the seed runs

- **Seeds weren't changing the data.** The training data loader's random generator was hard-coded to 0. It decides both the batch order and every random crop and flip, so with loader workers, seeds 1234 and 1235 saw identical batches. The repeats would only have varied the weight initialisation and understated the spread. A new `data.loader_seed` setting fixes it, with a test. The default of 0 keeps every earlier run reproducible.
- **"Loss fell 10% from epoch 0" was the wrong convergence check.** It mostly measures where a model *starts*. The inject head starts close to a good answer, so it couldn't drop 10% even while learning. I replaced it with "loss on fixed holdout images at least 1% below an untrained copy", and checked it against runs I already knew the answer for:
  - three stuck runs scored −0.17% to +0.25%;
  - every model that learned scored at least +4.1%.
- **TF32 was on by default.** PyTorch was running "fp32" convolutions at reduced precision, so the same checkpoint gave outputs about 2e-3 apart depending on memory layout. Deterministic mode now turns TF32 off; latency is still measured with it on.

## 14–15 Oct: from the add head to the inject head

The first FusionEdge head was `clamp(vi + delta)`, starting at delta = 0, and it never trained. After 1,280 steps the output had moved by at most 0.001 on average. The model starts exactly on the visible image, where the L1 and ReLU loss terms all sit on their kinks, so any change that isn't selective gets pushed straight back.

The loss *did* reward adding infrared. Raising the visible brightness to the IR level wherever IR is brighter scored 0.750, against 0.879 for doing nothing. So I changed the parameterisation to `Y = Y_vi + sigmoid(g) · relu(IR − Y_vi)`. It learned within the first smoke run.

Then the obvious question: was it the head or the FusionEdge backbone doing the work? I added the same head to the plain baseline, which led to the comparison above.

## Tests

- The suite has 378 tests.
- I ran the full suite 22 times on 15 Oct, the last on a fresh clone from GitHub. 21 runs passed cleanly. One (07:55) had a single failure that I never managed to name or reproduce, even across 3 runs with all 20 CPU cores saturated. It's still open.
- The Windows launcher now runs pytest with `-rf`, so a recurrence will print the test's name.

## Commands

```bash
python -m pytest -q
python scripts/infer.py --run experiments/baseline_simplecnn_inject --fp32 --ir data/MSRS/canonical/train/ir/00096D.png --vi data/MSRS/canonical/train/vi/00096D.png --out results/demo/baseline_inject_00096D.png
python scripts/audit_run.py experiments/fusionedge_main
python scripts/analyse_fusion.py experiments/baseline_simplecnn_inject experiments/baseline_simplecnn_inject_s1235 experiments/baseline_simplecnn_inject_s1236 experiments/fusionedge_main experiments/fusionedge_main_s1235 experiments/fusionedge_main_s1236 experiments/baseline_simplecnn_v6
python scripts/seed_rule.py results/analysis/<folder>/summary.json
python scripts/make_results.py --analysis results/analysis/<folder>
```

The inference command's output is pixel-identical to the evaluation code's forward pass, checked on the example pair for all three models.

## Open items

- **Intermittent test failure:** see above.
- **Ablations:** the seven FusionEdge ablations haven't been run. Given the seed result, the most useful one is `ablation_avg_fusion`, which replaces FusionEdge's gate with a plain average, plus a per-module latency profile.
- **Seeds:** the plain baseline (Y head) has only one seed, and three seeds per inject model is the minimum.
- **No external comparison:** I haven't compared against published fusion methods, and my AG/SF definitions differ from the published MSRS tables.
- **Hardware:** latency has only been measured on a laptop GPU, not on a phone or NPU.
- **Local paths:** run logs and `data/registry/MSRS.json` still contain this machine's project path (`D:\Projects\Sam`). I left them as recorded.
