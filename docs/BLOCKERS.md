# Blocked: what is waiting on what, and exactly what is needed to unblock

**Status: no real-data training, no benchmark table, no quality claim is possible yet.**

This file exists so the blockage is precise rather than vague. It states what is
done, what is blocked, and the exact checklist required to release the block.

---

## 1. What is complete and verified

| Component | State | Evidence |
|---|---|---|
| Repository, pinned deps, `.gitignore` | done | commit `ff58bfa` |
| Data layer, losses, metrics, classical baselines | done | unit tests, cross-validated against scikit-image |
| Training / evaluation / benchmark engine | done | integration tests on a generated 8-pair fixture |
| FusionEdge architecture + 6 ablation switches | done | 33 architecture tests incl. learnability check |
| Real-image inference path | done | `scripts/infer_pairs.py`, tested on real image files |
| Acquisition pipeline (extract, pair, split, audit, verify) | done | 27 tests in `tests/test_acquisition.py` |
| GPU capability and latency profile | done | `scripts/gpu_check.py`, `scripts/profile_latency.py` |
| **Test suite** | **244 passing** | `python -m pytest tests -q` |

## 2. What is blocked

| Blocked item | Blocked by |
|---|---|
| Baseline training run | dataset acquisition |
| FusionEdge training run | dataset acquisition |
| Ablation sweep (6 configs) | dataset acquisition |
| EN / AG / SF / MS-SSIM tables | dataset acquisition |
| YOLOv5x6 mAP | dataset acquisition + detection annotations |
| Any claim that FusionEdge > baseline | dataset acquisition + completed ablation |
| README results tables | the above |

**Both approved datasets are unavailable:**

* **MSRS** — the official repository publishes no archive URL. The only GitHub
  copy that could be verified (`Linfeng-Tang/SeAFusion`) contains 1,445
  segmentation *labels* and no IR/VI pairs. No HuggingFace mirror exists.
* **MFNet** — both Google Drive transports are quota-blocked. The detection
  release's folder listing succeeds but the file returns *"have had many
  accesses"* and direct requests redirect to a Google account sign-in page; the
  segmentation release returns HTTP 500.

Recorded in `data/registry/MSRS.json` and `data/registry/MFNet.json`, both with
`status: "unavailable"` and a verbatim reason per attempt.

## 3. Why nothing was substituted

Per the project specification and the approved plan:

* **No synthetic training.** `SyntheticPairDataset` requires an explicit
  `allow_synthetic=True` flag, marks every sample `"synthetic": true`, and is
  unreachable from `build_dataloaders`. Training entry points never see it.
* **No unlabelled substitute.** An arbitrary IR/VI folder would produce numbers
  that cannot be attributed to any dataset.
* **No blocked-but-noted numbers.** `run_experiment.py` exits with code 2 and an
  explicit message when the dataset registry is unusable, rather than training on
  something else.
* **No quality claims.** `infer.py` and `infer_pairs.py` *refuse* to run on
  untrained weights, because an untrained FusionEdge returns the visible image
  unchanged — a plausible-looking picture with zero infrared content. Outputs are
  labelled `"dataset_id": "UNREGISTERED"` and `"NOT a benchmark result"`.

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