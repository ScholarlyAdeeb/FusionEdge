# Dataset provenance, splits and acquisition log

Human-readable companion to `data/registry/*.json`, which is the machine-checkable
version. Every run records the `dataset_id` and `manifest_sha256` it was computed
on, and `scripts/make_figures.py` refuses to emit a table spanning two datasets.

---

## 0. Correction: MSRS *is* obtainable

An earlier revision of this project recorded MSRS as **unavailable**. That was
wrong, and the error is worth recording rather than quietly overwriting: the
check that produced it read the repository's **README only**, and concluded "no
archive URL published". A later, proper inspection of the GitHub trees API found
that the dataset is committed **as tracked files** in the repository:

| Path | Files | Content |
|---|---|---|
| `train/ir`, `train/vi` | 1083 each | training pairs |
| `train/Segmentation_labels/` | 1083 | segmentation masks |
| `test/ir`, `test/vi` | 361 each | test pairs |
| `test/Segmentation_labels/` | 361 | segmentation masks |
| `detection/ir`, `detection/vi` | 80 each | detection pairs |
| `detection/labels/` | 81 | YOLO-format `.txt` |

**1083 + 361 = 1444 pairs**, exactly the count the MSRS paper reports. The
`detection/` subset with YOLO labels is what the YOLOv5x6 mAP metric needs.

The lesson generalises: "the README has no download link" is not the same claim as
"the dataset is unavailable", and only the second one would have justified the
MFNet fallback.

## 1. Acquisition policy (as approved)

1. **MSRS first** — the benchmark most fusion papers report on. It is available,
   so it is the dataset used.
2. **MFNet fallback** — retained, and still attempted automatically if MSRS ever
   fails. It remains Drive-quota-blocked (see §3).
3. If both fail, stop and report. No substitute data, no synthetic training.

---

## 2. MSRS — ACQUIRED

```powershell
python scripts/download_data.py --dataset msrs
```

**Method: git sparse checkout**, not the repository tarball.

```
git init && git remote add origin https://github.com/Linfeng-Tang/MSRS.git
git config core.sparseCheckout true
git fetch --depth 1 --filter=blob:none origin main
git checkout FETCH_HEAD
```

with `.git/info/sparse-checkout` listing `train/{ir,vi}`, `test/{ir,vi}` and
`detection/{ir,vi,labels}`.

The full repository is ~656 MB, dominated by segmentation labels this project does
not use. The sparse checkout transfers only the image directories and leaves a
**verifiable commit SHA** in `data/registry/MSRS.json` under `source_commit`, so a
result can be tied to an exact upstream state rather than to "whatever was
downloaded that day".

### Split: the published partition, reproduced verbatim

MSRS ships its own `train/` and `test/`. **No re-splitting is performed.** `test/`
is reported under the name `val` throughout this codebase, because no MSRS method
trains on it; `docs/` and every result table state this explicitly.

| Split (this codebase) | MSRS directory | Pairs | Expected | Paper |
|---|---|---|---|---|
| `train` | `train/` | 1083 | 1083 | ✓ |
| `val` (held-out) | `test/` | 361 | 361 | ✓ |
| `detection` (mAP only) | `detection/` | 80 | 80 | ✓ |

The expected counts are asserted at acquisition time and a mismatch is logged as
`MISMATCH` rather than silently accepted.

**This is the important consequence:** because the published split is used
unaltered, results from this project **are** comparable to published MSRS numbers.
That is only true for MSRS — not for MFNet.

---

## 3. MFNet — retained fallback, still quota-blocked

Both known Google Drive transports remain blocked:

| Transport | Result |
|---|---|
| Detection release (`ir_det_dataset.tar.gz`, id `0B7Loi-7ye3pPY1dRalliazlQMDg`) | Folder lists; file returns *"have had many accesses"*; direct requests redirect to account sign-in |
| Segmentation release | Inner `images/` folder returns HTTP 500 |

`scripts/retry_mfnet.ps1` retries with exponential backoff (6h → 24h) and stops on
a quota signal rather than polling. It is **not running**: MSRS succeeded, so the
fallback is not needed.

---

## 4. Validation performed at acquisition

`scripts/download_data.py` runs these before writing `status: "ok"`:

- **Extraction integrity** — member count reported; zip CRCs checked; absolute,
  parent-traversing, link and device members refused.
- **Pairing** — matched on filename stem, so a `.png`/`.jpg` mismatch cannot be
  misreported as a corrupt dataset.
- **Dimension agreement** — a pair whose IR and VI differ in pixel size is
  **rejected**, because the loader would otherwise silently crop it and break the
  alignment fusion depends on.
- **Content audit** — every pair is decoded; loadability, dimension agreement and
  channel counts are recorded in the registry as `audit`.
- **Split integrity** — zero shared pair ids between `train` and `val`; zero
  byte-identical infrared images across splits.
- **Manifest** — SHA-256 over the ordered `split/pair_id` list.

Re-check any time with:

```powershell
python scripts/download_data.py --verify-only
```

### Known limitation, stated rather than hidden

The cross-split duplicate check is **exact-match only** (SHA-256 of file bytes).
Near-duplicate frames under different ids would not be caught. A perceptual hash
would catch more but requires an arbitrary similarity threshold, which is harder
to defend than an unambiguous exact check. MSRS is a curated dataset with a single
recording pass per scene, so near-duplicates across the published split are not
expected — but this is an assumption, not a verified fact.

---

## 5. Preprocessing

Applied at load time only; source files on disk are never modified.

| Step | Treatment | Why |
|---|---|---|
| Decode | single path via `cv2` for both modalities | a grayscale and a colour file of the same scene go through identical arithmetic |
| Infrared | kept as **1 channel**; 3-channel IR collapsed to BT.601 luma | replicating to 3 channels triples stem input width for no information |
| Visible | 3 channels; grayscale sources replicated | the colour loss term needs a chroma definition |
| Scale | `/255` to float32 in `[0,1]` | matches loss and metric definitions |
| Alpha | dropped | not present |
| Size mismatch | cropped to the common size | some pairs differ by a pixel in one axis |
| Train | random crop 128×128 (reflect-pad if smaller) + h/v flips | **geometric only** — no photometric jitter, which would fight the intensity loss |
| Eval | full image, reflect-padded to a multiple of 8, cropped back | padding avoids artificial border edges the gradient/edge losses would learn to reproduce |

---

## 6. Licence

MSRS images are **not redistributed** by this repository. Only the download script
and the provenance record are committed; `.gitignore` excludes `data/MSRS/`.
Obtain the dataset from the authors' own repository and cite:

> Tang, L., Yuan, J., & Ma, J. *MSRS: Multi-Spectral Road Scenarios for Practical
> Infrared and Visible Image Fusion.* IEEE Transactions on Multimedia, 2022.