# Status: dataset acquired; remaining gates before any reported number

> **Note (2026-10-15):** this page records the project's state up to 2026-10-13 and is kept for history.
> - The gates in §3–§5 have since been passed; the "six ablations" in §4 Step 5 have not been run.
> - Current state: [RESULTS.md](RESULTS.md) for the results, [EXPERIMENTS.md](EXPERIMENTS.md) for every run, [PROTOCOL.md](PROTOCOL.md) for the rules.

**MSRS is acquired and validated. Real-data training is no longer blocked on
acquisition.** The gates below now concern *results*, not *data*.

---

## 0. Correction

An earlier revision of this document reported MSRS as unavailable. That was wrong:
the check had inspected only the repository README. The dataset is committed as
tracked files in `github.com/Linfeng-Tang/MSRS` (1444 pairs + a detection subset
with YOLO labels). The mistake and its cause are recorded in `docs/DATASET.md` §0
rather than quietly overwritten.

---

## 1. Dataset status

| | |
|---|---|
| Dataset | **MSRS** (acquired) |
| Method | git sparse checkout, commit `5fa66e33c5ba` |
| Pairs | train 1083, val(=test) 361, detection 80 |
| Split | the **published** partition, unmodified |
| Audit | 1524 pairs, all 640×480, IR 1-channel, VI 3-channel, **0 problems** |
| Leakage | 0 shared ids, 0 byte-identical IR images across splits |
| Registry | `data/registry/MSRS.json`, `status: "ok"` |
| Verify | `python scripts/download_data.py --verify-only` → all checks pass |

Because the published split is used unaltered, results here **are** comparable to
published MSRS numbers. This applies to MSRS only.

## 2. What is complete

| Component | State |
|---|---|
| Repository, pinned deps, `.gitignore` | done |
| Data layer, losses, metrics, classical baselines | done, cross-validated against scikit-image |
| Training / evaluation / benchmark engine | done |
| FusionEdge + 6 ablation switches | done |
| Real-image inference path | done |
| Acquisition + audit + split + leakage verification | done |
| GPU capability and latency profile | done |
| YOLOv5x6 mAP evaluator (ultralytics + ONNX fallback) | done |

## 3. Remaining gates, in order

| Gate | What must happen | Blocks |
|---|---|---|
| **G1 smoke run** | `run_experiment.py --config configs/baseline.yaml --smoke` exits 0 with finite metrics and `"smoke": true` | everything downstream |
| **G2 baseline run** | `baseline.yaml` completes; `metrics.json` written | any comparison |
| **G3 FusionEdge run** | `fusionedge.yaml` completes | the comparison itself |
| **G4 ablations** | 6 ablation configs complete | the ablation table |
| **G5 figures** | `scripts/make_figures.py` generates tables **from** `metrics.json` | README results |
| **G6 mAP** | YOLOv5x6 evaluation on `detection/` with its YOLO labels | the downstream-task claim |

## 4. Rules that remain in force

* **No quality claim before G2–G5.** Nothing in README or docs may state that
  FusionEdge outperforms the baseline until the held-out numbers say so on the
  MSRS test split. If the ablations do not support a component, the component is
  removed rather than defended.
* **No hand-typed numbers.** Every table is generated from `experiments/*/metrics.json`
  by `scripts/make_figures.py`, which refuses to mix datasets or inference
  backends in one table.
* **No synthetic training.** `SyntheticPairDataset` is fenced behind
  `allow_synthetic=True`, marks every sample, and is unreachable from
  `build_dataloaders`.
* **Untrained weights are refused** by `infer.py` / `infer_pairs.py`, because an
  untrained FusionEdge returns the visible image unchanged — a plausible-looking
  picture carrying zero infrared information.

---

## 4. Unblock checklist

### Step 1 — obtain one archive

**Recommended: MFNet detection release** (`ir_det_dataset.tar.gz`). It is the
approved fallback, the parent of MSRS, and it ships the detection annotations
needed for the YOLO mAP metric.

Preferred method (authenticated, not quota-blocked):

1. Open the U-Tokyo project page:
   `https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/`
2. Follow the "Multi-spectral Object Detection Dataset" Google Drive link.
3. Download in a browser where you are signed in.
4. Place the archive at:

```
data/_archives/ir_det_dataset.tar.gz
```

No code change needed. `scripts/download_data.py` prefers a manual archive over
Drive and makes no network request when one is present.

Alternative: `python scripts/retry_mfnet.ps1` (exponential backoff, 6h → 24h,
stops on a quota signal rather than polling Drive aggressively).

### Step 2 — verify the archive is complete

```powershell
# expected: ~1569 pairs across day/ and night/, 480x640
# structure after extraction:
#   ir_det_dataset/<day|night>/ir/*.png
#   ir_det_dataset/<day|night>/rgb/*.png
#   ir_det_dataset/anno_json/*.json      (detection labels, for YOLO mAP)
Get-ChildItem -Recurse data\_archives | Select-Object Name,Length
```

**Required contents checklist:**

- [ ] `ir_det_dataset/day/ir/*.png` — non-empty
- [ ] `ir_det_dataset/day/rgb/*.png` — non-empty
- [ ] `ir_det_dataset/night/ir/*.png` — non-empty
- [ ] `ir_det_dataset/night/rgb/*.png` — non-empty
- [ ] `ir_det_dataset/anno_json/*.json` — present (needed for YOLO mAP)
- [ ] archive size is plausible for 1,569 pairs at 480x640 (order of GB, not MB)
- [ ] `python -c "import tarfile;print(len(tarfile.open('data/_archives/ir_det_dataset.tar.gz').getmembers()))"` returns a large member count, not 0

### Step 3 — register and validate

```powershell
python scripts/download_data.py --dataset mfnet
```

**This must print, in order:**

- [ ] `extracted N members` with N large (not 0)
- [ ] `N unique pairs collected` — expect ~1569 (or ~1400+ after rejected duplicates)
- [ ] `audit: N pairs, dims={'640x480': N}, ir_channels={1: N}, vi_channels={3: N}`
- [ ] `audit ... n_problems: 0` (no dimension mismatches, no unpaired files)
- [ ] `split val: ...` / `train: ...` totalling the pair count, ~20% in val
- [ ] `MFNet: status=ok`

Then re-verify independently:

```powershell
python scripts/download_data.py --verify-only
```

**Validation results required:**

- [ ] `ok train: N pairs` and `ok val: N pairs`
- [ ] `ok no train/val leakage` — zero shared pair ids
- [ ] `ok no cross-split IR content duplicates` — zero byte-identical IR images across splits
- [ ] `ok manifest hash matches registry`
- [ ] `ok content audit clean (N loadable pairs)`
- [ ] `data/registry/MFNet.json` has `status: "ok"` and a non-null `manifest_sha256`

### Step 4 — confirm the gate opens

```powershell
python scripts/run_experiment.py --config configs/baseline.yaml --smoke
```

- [ ] exits 0
- [ ] `experiments/baseline_simplecnn_smoke/` contains `config.yaml`, `metrics.json`, `benchmark.json`, `checkpoints/best.pt`
- [ ] `metrics.json` shows `"smoke": true` and a `dataset.dataset_id` of `MFNet` with the manifest hash
- [ ] printed metrics are all finite

### Step 5 — real runs (only after the above)

```powershell
python scripts/run_experiment.py --config configs/baseline.yaml
python scripts/run_experiment.py --config configs/fusionedge.yaml
# then the six ablations, sequentially
```

- [ ] `fusionedge_main` completes before any comparison is written
- [ ] the ablation table is generated **by** `scripts/make_figures.py` from `metrics.json` files, never typed by hand
- [ ] every table is labelled `MFNet`, and states that it is not comparable to published MSRS numbers
- [ ] no statement that FusionEdge beats the baseline appears anywhere before the numbers support it

---

## 5. If MSRS becomes obtainable later

MSRS is the benchmark most fusion papers report on, so it is worth revisiting.

1. Place the archive in `data/_archives/`.
2. `python scripts/download_data.py --dataset msrs`
3. It will use the **published** `train/` + `test/` split rather than a re-split.
4. Train a **fresh** set of runs. MSRS rows and MFNet rows must never share a
   table — `scripts/make_figures.py` refuses to build one, by design.