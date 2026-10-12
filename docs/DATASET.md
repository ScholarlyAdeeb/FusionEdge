# Dataset provenance, splits and acquisition log

This file is the human-readable companion to `data/registry/*.json`, which is the
machine-checkable version. Every run in `experiments/` records the `dataset_id`
and `manifest_sha256` it was computed on, and `scripts/make_figures.py` refuses to
emit a comparison table that spans two datasets. **MSRS results and MFNet results
are never placed in the same table and are never described interchangeably.**

---

## 1. Acquisition policy (as approved)

1. **MSRS first.** It is the benchmark most infrared/visible fusion papers report
   on, so it is tried before anything else.
2. **MFNet as the approved fallback.** It is the parent dataset MSRS was curated
   from (1,569 aligned day/night pairs at 480x640).
3. If both fail, stop and report. No substitute data, and no synthetic training.

The synthetic dataset in `fusionedge/data/synthetic.py` is used by unit tests
only and is refused by the training and evaluation entry points.

---

## 2. MSRS — ATTEMPTED, UNAVAILABLE

MSRS (*Multi-Spectral Road Scenarios for Practical Infrared and Visible Image
Fusion*, Tang et al.) contains 1,444 aligned IR/VI pairs at 480x640, and is the
standard benchmark for this task. It could **not** be obtained:

| Source | Result |
|---|---|
| `github.com/Linfeng-Tang/MSRS` (official) | Repository documents the dataset but publishes **no archive URL**. Distribution is via Baidu Pan / author channels. |
| `github.com/Linfeng-Tang/SeAFusion/datasets/MSRS` | Verified via the GitHub trees API: contains **only `Label/*.png` — 1,445 segmentation masks**. No `ir/` or `vi/` image pairs. |
| HuggingFace search (`MSRS`, `image fusion`, `infrared visible fusion`) | No MSRS mirror found. |

Recorded in `data/registry/MSRS.json` with `status: "unavailable"`.

**Consequence: no MSRS number appears anywhere in this repository.** If MSRS is
obtained later, run `python scripts/download_data.py --dataset msrs`, point the
configs at `dataset.id: MSRS`, and train a fresh set of runs — do not mix those
rows with MFNet rows.

---

## 3. MFNet — ATTEMPTED, CURRENTLY BLOCKED BY GOOGLE DRIVE QUOTA

MFNet (*Multispectral image dataset for RGBT detection*, Takdir et al., U-Tokyo /
RIKEN AIP) is publicly linked from the U-Tokyo project page. Both known Drive
transports are currently quota-blocked:

| Transport | Source | Result |
|---|---|---|
| Object-detection release (preferred) | `drive.google.com/drive/folders/1YtEMiUC8sC0iL9rONNv96n5jWuIsWrVY` (`ir_det_dataset.tar.gz`, id `0B7Loi-7ye3pPY1dRalliazlQMDg`) | Folder listing succeeds; the file itself returns *"Cannot retrieve the public link… have had many accesses"*; direct `drive.usercontent.google.com` requests redirect to a Google **account sign-in page**. |
| Semantic-segmentation release | `drive.google.com/drive/folders/18BQFWRfhXzSuMloUmtiBRFrr6NSrf8Fw` | Inner `images/` folder listing returns **HTTP 500**. |

Google Drive applies a per-file download quota to widely-mirrored files; this
archive is widely mirrored. The quota is transient, so
`scripts/retry_mfnet.ps1` retries on a bounded schedule (default 8 attempts,
25 min apart) and stops as soon as the registry reports `status: ok`.

### Manual fallback (recommended if the retries are exhausted)

Download from a browser (where you are authenticated) and drop the archive in
`data/_archives/`:

```
data/_archives/ir_det_dataset.tar.gz
```

Then:

```powershell
python scripts/download_data.py --dataset mfnet
```

The script prefers a manually-provided archive over Drive, records its SHA-256,
and proceeds with the normal verification. No code change is needed.

---

## 4. Split definition (MFNet)

Because the published MSRS split is unavailable, we define our own split and it
is recorded in the registry:

```
sha1(pair_id) -> first 8 hex digits -> uint32
ascending order over all pair ids
lowest 20%  -> val
remainder   -> train
```

* **Content-hash based, not RNG-seed based.** Identical on every machine and
  across reruns, independent of library RNG changes.
* **Pairs are grouped before splitting.** The hash is computed on the *pair id*,
  so the IR and VI files of a scene can never straddle the boundary. This is
  verified, not assumed: `scripts/download_data.py --verify-only` asserts zero
  `pair_id` intersection between splits and checks the manifest hash against
  disk.
* Split counts and the `manifest_sha256` are written to
  `data/registry/MFNet.json` and copied into every run's stamped config.

**Deviation from published benchmarks, stated plainly:** MFNet results here are
*not* comparable to published MSRS numbers. MSRS was curated from MFNet (misaligned
pairs removed, infrared contrast enhanced with dark-channel prior), so the two
distributions differ. Any table in this repository is labelled with its dataset.

---

## 5. Preprocessing

Applied at load time only; source files on disk are never modified.

| Step | Treatment | Why |
|---|---|---|
| Decode | single path via `cv2` for both modalities | a grayscale and a colour file of the same scene go through identical arithmetic |
| Infrared | kept as **1 channel**; 3-channel IR collapsed to BT.601 luma | replicating to 3 channels triples stem input width for no information, and makes "what does the model see" harder to answer |
| Visible | 3 channels; grayscale sources replicated | the network's colour term needs a chroma definition |
| Scale | `/255` to float32 in `[0,1]` | matches the loss and metric definitions |
| Alpha channel | dropped | not present in either dataset |
| Size mismatch | cropped to the common size | some pairs differ by a pixel in one axis; cropping beats a shape error |
| Train | random crop 128x128 (reflect-pad if smaller) + h/v flips | **geometric only** — no photometric jitter, which would fight the intensity loss |
| Eval | full image, reflect-padded to a multiple of 8, cropped back | padding avoids artificial border edges that the gradient/edge losses would learn to reproduce |

---

## 6. Reproducing the acquisition

```powershell
python scripts/download_data.py --dataset auto     # MSRS, then MFNet
python scripts/download_data.py --dataset mfnet    # fallback only
python scripts/download_data.py --verify-only      # re-check splits + manifest
```