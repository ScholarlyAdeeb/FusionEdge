"""Dataset acquisition for FusionEdge.

Strategy (see docs/DATASET.md for the rationale):

  1. MSRS first.  MSRS is the published benchmark most IR/VL fusion papers evaluate
     on, so we try to obtain it before anything else.  Its official repository
     (Linfeng-Tang/MSRS) ships no data, and the copies we could verify on GitHub
     (e.g. Linfeng-Tang/SeAFusion) contain only the 1445 segmentation *labels*,
     not the IR/VI image pairs.  We still keep the attempt scripted and logged so
     the failure is a recorded, reproducible fact rather than an anecdote.

  2. MFNet fallback.  MFNet is the parent dataset MSRS was curated from
     (1569 aligned day/night pairs, 480x640) and is publicly linked from
     Linfeng-Tang/SeAFusion's README (a Google Drive folder).  We download it and
     build our own reproducible 80/20 split.

Every acquisition writes `data/registry/<dataset_id>.json` describing provenance:
source URL, retrieval timestamp, SHA-256 of the downloaded archive, licence note,
pair counts per split, and a SHA-256 manifest hash over the ordered list of
`split/pair_id` entries.  Downstream, every training run copies that
`dataset_id` + `manifest_sha256` into its own config, so a result row can always
be traced back to the exact bytes it was computed on -- and MSRS numbers can
never be silently presented as MFNet numbers or vice versa.

Usage
-----
    python scripts/download_data.py --dataset mfnet
    python scripts/download_data.py --dataset auto      # MSRS, then MFNet
    python scripts/download_data.py --verify-only
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Iterable

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
REGISTRY_DIR = DATA_DIR / "registry"

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# --- MSRS acquisition ------------------------------------------------------
# CORRECTION (see docs/DATASET.md): an earlier revision of this script recorded
# MSRS as unavailable after inspecting only the README. The dataset IS present in
# the repository as tracked files:
#
#   train/ir, train/vi            1083 pairs   (plus train/Segmentation_labels)
#   test/ir,  test/vi             361 pairs   (plus test/Segmentation_labels)
#   detection/ir, detection/vi     80 pairs   (+ detection/labels, YOLO format)
#                                                          -> 1083 + 361 = 1444
#
# which is exactly the 1,444 pairs the MSRS paper reports.
#
# We use a git sparse checkout rather than the repository tarball: the full repo
# is ~656 MB (dominated by segmentation labels we do not need), while a sparse
# checkout of only the image directories transfers far less and leaves a
# verifiable commit SHA in the registry.
MSRS_SOURCES: list[dict] = [
    {
        "kind": "git_sparse",
        "url": "https://github.com/Linfeng-Tang/MSRS.git",
        "ref": "main",
        "sparse_paths": [
            "train/ir",
            "train/vi",
            "test/ir",
            "test/vi",
            "detection/ir",
            "detection/vi",
            "detection/labels",
        ],
        "note": (
            "Official MSRS repository, dataset committed as tracked files. "
            "Sparse checkout of the image directories plus the detection labels."
        ),
    },
]

#: The published MSRS split, as documented by the dataset authors. We reproduce
#: it verbatim and never re-split: train/ for fitting, test/ for reporting.
MSRS_SPLITS = {"train": ("train/ir", "train/vi"), "val": ("test/ir", "test/vi")}

#: Expected counts from the MSRS paper (Tang et al.), used as a sanity gate.
#: A mismatch is reported loudly rather than silently accepted.
MSRS_EXPECTED = {"train": 1083, "val": 361, "detection": 80}

# --- MFNet fallback --------------------------------------------------------
# Two distinct Google Drive releases of MFNet exist; both are listed so the
# attempt order is explicit and the failure is recorded per-source.
#
#   (a) 1YtEMiUC8sC0iL9rONNv96n5jWuIsWrVY  "Multi-spectral Object Detection
#       Dataset" -- the canonical MFNet release linked from the U-Tokyo project
#       page (https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/).
#       Contains ir_det_dataset.tar.gz (day+night, 1569 pairs, 480x640).
#   (b) 18BQFWRfhXzSuMloUmtiBRFrr6NSrf8Fw  "Multi-spectral Semantic
#       Segmentation Dataset", linked from Linfeng-Tang/SeAFusion README.
#       MSRS was curated from the MFNet *detection* split, so (a) is preferred.
MFNET_SOURCES: list[dict] = [
    {
        "kind": "gdrive_folder",
        "url": "https://drive.google.com/drive/folders/1YtEMiUC8sC0iL9rONNv96n5jWuIsWrVY",
        "note": (
            "U-Tokyo official MFNet object-detection release (ir_det_dataset.tar.gz, "
            "file id 0B7Loi-7ye3pPY1dRalliazlQMDg). Preferred transport."
        ),
    },
    {
        "kind": "gdrive_folder",
        "url": "https://drive.google.com/drive/folders/18BQFWRfhXzSuMloUmtiBRFrr6NSrf8Fw",
        "note": (
            "MFNet semantic-segmentation release linked from Linfeng-Tang/SeAFusion "
            "README. Secondary transport."
        ),
    },
]

LICENCE_NOTES = {
    "MSRS": (
        "MSRS: Multi-Spectral Road Scenarios for Practical Infrared and Visible "
        "Image Fusion (Tang et al.). Obtained from the authors' own repository "
        "github.com/Linfeng-Tang/MSRS, which commits the dataset as tracked "
        "files. Research use; cite the original paper. The images are NOT "
        "redistributed by this project -- only the download script and the "
        "provenance record are committed."
    ),
    "MFNet": (
        "MFNet: Multispectral image dataset for RGBT detection "
        "(Takdir et al., U-Tokyo, RIKEN AIP). Research use; cite the original "
        "paper and the U-Tokyo project page. Redistribution is not granted by "
        "this repository."
    ),
    "synthetic": "Procedurally generated in-repo. No external licence applies.",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    print(f"[{_dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def _is_within(child: Path, parent: Path) -> bool:
    """True if `child` is inside `parent`.

    Uses Path.relative_to rather than a string prefix test: the naive
    `str(child).startswith(str(parent))` form wrongly accepts sibling paths that
    merely share a prefix (e.g. /data/MFNet_evil vs /data/MFNet).
    """
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _members_of(archive: Path) -> list[str]:
    if archive.suffix.lower() in (".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".tar.gz"):
        if archive.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive) as zf:
                return zf.namelist()
        with tarfile.open(archive) as tf:
            return tf.getnames()
    raise ValueError(f"unsupported archive type: {archive.name}")


def safe_extract(archive: Path, dest: Path) -> int:
    """Extract an archive, refusing any member that escapes `dest`.

    Returns the number of members extracted, which the caller uses to detect a
    truncated or partially-downloaded archive: gdown can exit 0 after fetching
    only part of a folder, and an extraction that yields 0 members must not be
    mistaken for a successful acquisition.

    Traversal is checked with Path.relative_to, and absolute members, links and
    devices are refused outright (Python's tar filter="data" handles this too,
    but the check happens first so the failure message names the member).
    """
    dest.mkdir(parents=True, exist_ok=True)

    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise RuntimeError(
                    f"zip archive is corrupt: first bad member is {bad!r} "
                    f"(the download was likely truncated)"
                )
            for m in zf.infolist():
                if not _is_within(dest / m.filename, dest):
                    raise RuntimeError(f"refusing unsafe zip member: {m.filename}")
            count = len(zf.infolist())
            zf.extractall(dest)
        return count

    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        for m in members:
            if not _is_within(dest / m.name, dest):
                raise RuntimeError(f"refusing unsafe tar member: {m.name}")
            if m.issym() or m.islnk() or m.isdev():
                raise RuntimeError(f"refusing link/device tar member: {m.name}")
        # filter="data" additionally strips absolute paths, setuid bits and
        # non-regular files; on Python < 3.12 we do our own checks above.
        try:
            tf.extractall(dest, filter="data")  # type: ignore[call-arg]
        except TypeError:  # Python < 3.12
            tf.extractall(dest)
    return len(members)


def archive_member_count(archive: Path) -> int:
    """How many members the archive claims to contain (no extraction)."""
    return len(_members_of(archive))


def find_pair_dirs(root: Path) -> list[tuple[Path, Path]]:
    """Locate `ir`/`vi` directory pairs anywhere under `root`.

    Returns [(ir_dir, vi_dir), ...] sorted for determinism. Accepts the common
    naming variants: vi/vis/visible/rgb/v and ir/infrared/thermal.
    """
    vi_names = {"vi", "vis", "visible", "rgb", "v", "infrared_visible"}
    ir_names = {"ir", "infrared", "thermal", "i"}
    found: list[tuple[Path, Path]] = []
    for dirpath, dirnames, _ in os.walk(root):
        here = Path(dirpath)
        lowered = {d.lower(): d for d in dirnames}
        vi = next((lowered[n] for n in sorted(lowered) if n in vi_names), None)
        ir = next((lowered[n] for n in sorted(lowered) if n in ir_names), None)
        if vi and ir:
            found.append((here / ir, here / vi))
            # don't descend further: nested ir/vi would be a subset
            dirnames[:] = []
    return sorted(found, key=lambda t: str(t[0]))


def list_images(d: Path) -> list[str]:
    return sorted(
        p.name for p in d.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def pair_up(ir_dir: Path, vi_dir: Path) -> tuple[list[str], list[str]]:
    """Return (paired_ids, unpaired_left).

    Pairs are matched on filename *stem* rather than full filename. Both
    directories must be equally strict about extensions, but a dataset that
    stores infrared as .png and visible as .jpg (or vice versa) is common enough
    that matching on the exact name would silently yield zero pairs and then be
    reported as "corrupt" -- a misleading diagnosis. Stem matching pairs what is
    actually a pair and reports the extension mismatch as unpaired leftovers,
    which is the honest outcome.
    """
    ir = list_images(ir_dir)
    vi = list_images(vi_dir)

    def by_stem(names: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for n in names:
            out.setdefault(Path(n).stem, n)
        return out

    ir_map, vi_map = by_stem(ir), by_stem(vi)
    pairs = sorted((k, ir_map[k], vi_map[k]) for k in ir_map.keys() & vi_map.keys())

    # a pair is only usable if the two files have the same pixel dimensions;
    # a mis-dimensioned pair would be silently cropped at load time, which
    # would corrupt the alignment the fusion task depends on
    validated: list[tuple[str, str, str]] = []
    mismatched: list[str] = []
    for stem, ir_name, vi_name in pairs:
        try:
            ir_img = cv2.imread(str(ir_dir / ir_name), cv2.IMREAD_UNCHANGED)
            vi_img = cv2.imread(str(vi_dir / vi_name), cv2.IMREAD_UNCHANGED)
        except Exception:  # noqa: BLE001
            mismatched.append(stem)
            continue
        if ir_img is None or vi_img is None:
            mismatched.append(stem)
            continue
        if ir_img.shape[:2] != vi_img.shape[:2]:
            mismatched.append(stem)
            continue
        validated.append((stem, ir_name, vi_name))

    leftover = sorted(
        (set(ir) | set(vi))
        - {ir_name for _, ir_name, _ in validated}
        - {vi_name for _, _, vi_name in validated}
    )
    if mismatched:
        log(
            f"  warning: {len(mismatched)} pairs rejected for unreadable or "
            f"size-mismatched IR/VI, e.g. {mismatched[:5]}"
        )
    return validated, leftover


def file_digest(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a single file's bytes."""
    return sha256_file(path, chunk)


def manifest_hash(entries: Iterable[str]) -> str:
    h = hashlib.sha256()
    for e in entries:
        h.update(e.encode())
        h.update(b"\n")
    return h.hexdigest()


def audit_dataset(root: Path) -> dict:
    """Audit a laid-out dataset: loadability, dimension agreement, channel counts.

    This is the gate that stands between "files were downloaded" and "these files
    are a usable, aligned IR/VI corpus". It is deterministic given the same tree,
    so its output is stored in the registry and re-checkable via --verify-only.

    Returns a dict; `problems` is non-empty if anything is inconsistent.
    """
    problems: list[str] = []
    dims: dict[str, int] = {}
    ir_ch: dict[int, int] = {}
    vi_ch: dict[int, int] = {}
    n_pairs = 0

    for split in sorted(p.name for p in root.iterdir() if p.is_dir()):
        ir_d, vi_d = root / split / "ir", root / split / "vi"
        if not (ir_d.is_dir() and vi_d.is_dir()):
            problems.append(f"{split}: missing ir/ or vi/ directory")
            continue
        ir_names = set(list_images(ir_d))
        vi_names = set(list_images(vi_d))
        if ir_names != vi_names:
            problems.append(
                f"{split}: {len(ir_names - vi_names)} ir-only, "
                f"{len(vi_names - ir_names)} vi-only files"
            )
        for name in sorted(ir_names & vi_names):
            ir_img = cv2.imread(str(ir_d / name), cv2.IMREAD_UNCHANGED)
            vi_img = cv2.imread(str(vi_d / name), cv2.IMREAD_UNCHANGED)
            if ir_img is None or vi_img is None:
                problems.append(f"{split}/{name}: unreadable")
                continue
            if ir_img.shape[:2] != vi_img.shape[:2]:
                problems.append(
                    f"{split}/{name}: dimension mismatch "
                    f"ir={ir_img.shape[:2]} vi={vi_img.shape[:2]}"
                )
                continue
            n_pairs += 1
            dims[f"{ir_img.shape[1]}x{ir_img.shape[0]}"] = dims.get(
                f"{ir_img.shape[1]}x{ir_img.shape[0]}", 0
            ) + 1
            c_ir = 1 if ir_img.ndim == 2 else ir_img.shape[2]
            c_vi = 1 if vi_img.ndim == 2 else vi_img.shape[2]
            ir_ch[c_ir] = ir_ch.get(c_ir, 0) + 1
            vi_ch[c_vi] = vi_ch.get(c_vi, 0) + 1

    return {
        "n_pairs": n_pairs,
        "unique_dims": dims,
        "ir_channels": ir_ch,
        "vi_channels": vi_ch,
        "problems": problems[:50],
        "n_problems": len(problems),
    }


# --------------------------------------------------------------------------- #
# MSRS attempt
# --------------------------------------------------------------------------- #
def git_sparse_checkout(
    url: str,
    ref: str,
    sparse_paths: list[str],
    dest: Path,
    timeout_min: float = 25.0,
) -> str:
    """Sparse-checkout `sparse_paths` from `url` into `dest`. Returns the commit SHA.

    Uses `git clone --filter=blob:none --sparse`, which avoids downloading the
    repository's full history and its unrequested blobs. The commit SHA is
    returned and stored in the registry so a result can be tied to an exact
    upstream state.
    """
    import subprocess

    dest.mkdir(parents=True, exist_ok=True)
    work = dest / "_repo"
    _force_rmtree(work)
    # `git init` requires the directory to exist; creating it first avoids the
    # cwd-based NotADirectoryError on Windows.
    work.mkdir(parents=True, exist_ok=True)

    def run(args: list[str], cwd: Path | None = None, timeout: float | None = None) -> None:
        log("  $ git " + " ".join(args))
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout if timeout else timeout_min * 60,
            check=False,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-4:]
            raise RuntimeError(f"git {' '.join(args[:2])} failed: {' | '.join(tail)}")

    run(["init", "--quiet"], cwd=work)
    run(["remote", "add", "origin", url], cwd=work)
    run(["config", "core.sparseCheckout", "true"], cwd=work)
    run(["config", "extensions.partialClone", "origin"], cwd=work)
    (work / ".git" / "info").mkdir(parents=True, exist_ok=True)
    (work / ".git" / "info" / "sparse-checkout").write_text(
        "\n".join(sparse_paths) + "\n", encoding="utf-8"
    )
    run(
        ["fetch", "--depth", "1", "--filter=blob:none", "origin", ref],
        cwd=work,
    )
    run(["checkout", "--quiet", "FETCH_HEAD"], cwd=work)

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(work),
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return sha


def _force_rmtree(path: Path) -> None:
    """Remove a tree, clearing read-only bits first.

    Git writes pack files read-only on Windows, so a plain shutil.rmtree raises
    PermissionError on the second run. Handled here rather than at each call
    site, since re-running the script after a partial download is the normal
    case, not an edge case.
    """
    if not path.exists():
        return

    def onerror(func, p, _exc):
        try:
            os.chmod(p, 0o700)
            func(p)
        except Exception:  # noqa: BLE001 -- best effort; rmtree will report if it fails
            pass

    shutil.rmtree(path, onerror=onerror)


def attempt_msrs(timeout_min: float = 25.0, force: bool = False) -> dict:
    """Acquire MSRS from the official repository and reproduce its published split.

    Unlike MFNet, MSRS ships an explicit train/ and test/ partition, so we use it
    as-is: `train/` for fitting, `test/` as the held-out split reported as `val`
    (no MSRS method trains on test/). The `detection/` subset is registered
    separately for the YOLOv5x6 mAP metric.
    """
    log("MSRS: acquiring from the official repository (sparse checkout)")
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    registry_file = REGISTRY_DIR / "MSRS.json"
    root = DATA_DIR / "MSRS"

    if registry_file.exists() and not force:
        rec = json.loads(registry_file.read_text())
        if rec.get("status") == "ok" and Path(rec.get("root", "")).exists():
            log("MSRS: already present and registered; skipping download")
            return rec

    attempts: list[dict] = []
    entry = dict(MSRS_SOURCES[0])
    entry["attempted_at"] = utc_now()
    commit = None
    try:
        _force_rmtree(root)
        commit = git_sparse_checkout(
            entry["url"], entry["ref"], entry["sparse_paths"], root, timeout_min
        )
        entry["downloaded"] = True
        entry["commit"] = commit
        entry["result"] = f"sparse checkout ok at commit {commit[:12]}"
    except Exception as exc:  # noqa: BLE001
        entry["downloaded"] = False
        entry["result"] = f"{type(exc).__name__}: {exc}"
        log(f"  - MSRS: {entry['result']}")

    attempts.append(entry)

    def unavailable_record(note: str) -> dict:
        rec = {
            "dataset_id": "MSRS",
            "status": "unavailable",
            "retrieved_at": utc_now(),
            "licence": LICENCE_NOTES["MSRS"],
            "attempts": attempts,
            "root": None,
            "splits": {},
            "manifest_sha256": None,
            "notes": note,
        }
        registry_file.write_text(json.dumps(rec, indent=2))
        return rec

    if not entry.get("downloaded"):
        return unavailable_record("MSRS checkout failed; see attempts.")

    repo = root / "_repo"
    canonical = root / "canonical"
    splits: dict[str, list[str]] = {}

    for split, (ir_rel, vi_rel) in MSRS_SPLITS.items():
        ir_src, vi_src = repo / ir_rel, repo / vi_rel
        if not (ir_src.is_dir() and vi_src.is_dir()):
            return unavailable_record(f"missing {ir_rel}/{vi_rel} in the checkout")
        pairs, leftover = pair_up(ir_src, vi_src)
        if leftover:
            log(f"  warning: {split}: {len(leftover)} unpaired files skipped")
        if not pairs:
            return unavailable_record(f"no validated pairs in {split}")
        for mod in ("ir", "vi"):
            (canonical / split / mod).mkdir(parents=True, exist_ok=True)
        for stem, ir_name, vi_name in pairs:
            shutil.copy2(ir_src / ir_name, canonical / split / "ir" / f"{stem}.png")
            shutil.copy2(vi_src / vi_name, canonical / split / "vi" / f"{stem}.png")
        splits[split] = sorted(stem for stem, _, _ in pairs)
        expected = MSRS_EXPECTED[split]
        got = len(splits[split])
        flag = "ok" if got == expected else f"MISMATCH (paper reports {expected})"
        log(f"  {split}: {got} pairs [{flag}]")

    # detection subset, used for the YOLOv5x6 mAP metric
    det_ir, det_vi = repo / "detection" / "ir", repo / "detection" / "vi"
    det_pairs: list[str] = []
    if det_ir.is_dir() and det_vi.is_dir():
        det_pairs_list, _ = pair_up(det_ir, det_vi)
        for mod in ("ir", "vi"):
            (canonical / "detection" / mod).mkdir(parents=True, exist_ok=True)
        for stem, ir_name, vi_name in det_pairs_list:
            shutil.copy2(det_ir / ir_name, canonical / "detection" / "ir" / f"{stem}.png")
            shutil.copy2(det_vi / vi_name, canonical / "detection" / "vi" / f"{stem}.png")
            det_pairs.append(stem)
        det_pairs.sort()
        log(f"  detection: {len(det_pairs)} pairs "
            f"[{'ok' if len(det_pairs) == MSRS_EXPECTED['detection'] else 'MISMATCH'}]")

    det_labels_src = repo / "detection" / "labels"
    det_labels_dst = DATA_DIR / "MSRS" / "detection_labels"
    if det_labels_src.is_dir():
        _force_rmtree(det_labels_dst)
        shutil.copytree(det_labels_src, det_labels_dst)
        n_lab = len([p for p in det_labels_dst.iterdir() if p.is_file()])
        log(f"  detection labels: {n_lab} files -> {det_labels_dst}")

    audit = audit_dataset(canonical)
    if audit["n_pairs"] == 0:
        return unavailable_record(f"content audit found no loadable pairs: {audit}")
    log(f"  audit: {audit['n_pairs']} pairs, dims={audit['unique_dims']}, "
        f"ir_ch={audit['ir_channels']}, vi_ch={audit['vi_channels']}, "
        f"problems={audit['n_problems']}")

    # The manifest covers the train/val partition only. `detection/` is an
    # auxiliary subset for the mAP metric; its count is recorded separately in
    # `splits` so a change in either is still visible, but it is deliberately
    # excluded so that this manifest means "the benchmark split", which is what
    # a results row is attributable to.
    entries = [f"{s}/{pid}" for s, ids in splits.items() for pid in ids]

    total = sum(len(v) for v in splits.values())
    record = {
        "dataset_id": "MSRS",
        "status": "ok",
        "retrieved_at": utc_now(),
        "licence": LICENCE_NOTES["MSRS"],
        "attempts": attempts,
        "source_commit": commit,
        "source_url": MSRS_SOURCES[0]["url"],
        "root": str(canonical),
        "split_definition": (
            "The dataset's own published partition, reproduced verbatim: "
            "train/ is used for fitting and test/ is the held-out split (reported "
            "as 'val'). No re-splitting is performed, so these numbers are "
            "comparable to published MSRS results."
        ),
        "split_note": (
            "MSRS ships train/ and test/; test/ is named 'val' throughout this "
            "codebase because no MSRS method trains on it."
        ),
        "split_val_frac": round(len(splits["val"]) / max(total, 1), 4),
        "splits": {k: len(v) for k, v in splits.items()} | {"detection": len(det_pairs)},
        "expected_splits": MSRS_EXPECTED,
        "audit": audit,
        "manifest_covers": sorted(splits.keys()),
        "manifest_sha256": manifest_hash(entries),
        "detection_labels": str(det_labels_dst) if det_labels_dst.exists() else None,
        "notes": (
            f"Primary benchmark dataset. {total} pairs across the published "
            f"train/test partition, plus {len(det_pairs)} detection pairs for "
            f"YOLOv5x6 mAP."
        ),
    }
    registry_file.write_text(json.dumps(record, indent=2))
    log(f"MSRS: status=ok, splits={record['splits']} -> registry written")
    return record


# --------------------------------------------------------------------------- #
# MFNet acquisition
# --------------------------------------------------------------------------- #
def download_gdrive_folder(url: str, dest: Path, timeout_min: float) -> None:
    """Download a Google Drive folder with gdown (may require cookie auth on
    quota-limited large folders; a clear error is raised on failure)."""
    import gdown  # imported lazily: only this path needs the optional dep

    dest.parent.mkdir(parents=True, exist_ok=True)
    log(f"  gdown folder download -> {dest} (timeout {timeout_min:.0f} min)")
    # gdown has no native timeout; enforce one by running the transfer as a
    # subprocess with a wall-clock cap, so a hung quota page cannot stall us.
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "gdown",
        "--folder",
        "--no-cookies" if os.environ.get("FUSIONEDGE_GDOWN_NOCOOKIE") else "--continue",
        url,
        "-O",
        str(dest),
    ]
    log("  $ " + " ".join(cmd))
    try:
        proc = subprocess.run(cmd, timeout=timeout_min * 60, check=False)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"gdown timed out after {timeout_min:.0f} min (Drive quota or rate limit)"
        )
    if proc.returncode != 0:
        raise RuntimeError(f"gdown exited with code {proc.returncode}")


def normalise_mfnet(root: Path) -> Path:
    """MFNet ships as <root>/<subset>/{rgb,ir}/*.png or with a nested archive.
    Return a root that contains at least one ir/vi directory pair."""
    if find_pair_dirs(root):
        return root
    # sometimes everything is one level deeper
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if find_pair_dirs(child):
            return child
    return root


def build_hashesplit(pairs: list[str], val_frac: float = 0.2) -> dict[str, list[str]]:
    """Deterministic content-independent split by SHA-1 of the pair id.

    Chosen over a seeded RNG so the split is identical on every machine and
    across reruns regardless of library RNG changes. 20% of ids go to val;
    because the hash is computed per *pair id* and pairs are grouped before
    splitting, IR and VI files of the same scene can never straddle the split.
    """
    scored = []
    for pid in pairs:
        h = hashlib.sha1(pid.encode()).hexdigest()
        scored.append((int(h[:8], 16) / 0xFFFFFFFF, pid))
    scored.sort()
    n_val = int(round(len(scored) * val_frac))
    val = sorted(pid for _, pid in scored[:n_val])
    train = sorted(pid for _, pid in scored[n_val:])
    return {"train": train, "val": val}


def attempt_mfnet(timeout_min: float = 10.0, force: bool = False) -> dict:
    log("MFNet: attempting acquisition (approved fallback)")
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    out_root = DATA_DIR / "MFNet"
    registry_file = REGISTRY_DIR / "MFNet.json"

    if registry_file.exists() and not force:
        rec = json.loads(registry_file.read_text())
        if rec.get("status") == "ok":
            log("MFNet: already present and registered; skipping download")
            return rec

    archive_dir = DATA_DIR / "_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    attempts = []
    downloaded_archive: Path | None = None

    # --- manual drop: an archive already present in data/_archives wins. ---
    # This is the documented escape hatch when Drive is quota-blocked: download
    # ir_det_dataset.tar.gz from a browser (which is authenticated) and drop it
    # in data/_archives/. The SHA-256 is recorded either way.
    local = sorted(archive_dir.glob("*.tar.gz")) + sorted(archive_dir.glob("*.tgz")) \
        + sorted(archive_dir.glob("*.zip"))
    if local:
        src = local[0]
        log(f"MFNet: using manually-provided archive {src}")
        try:
            n_members = safe_extract(src, out_root)
            log(f"  extracted {n_members} members ({src.stat().st_size / 2**20:.1f} MB)")
            if n_members == 0:
                raise RuntimeError(
                    "archive contained no members -- it is empty or was not fully "
                    "downloaded"
                )
            downloaded_archive = src
            attempts.append(
                {
                    "kind": "local_archive",
                    "url": str(src.relative_to(REPO_ROOT)),
                    "sha256": sha256_file(src),
                    "downloaded": True,
                    "result": "extracted from data/_archives",
                    "attempted_at": utc_now(),
                }
            )
        except Exception as exc:  # noqa: BLE001
            attempts.append(
                {
                    "kind": "local_archive",
                    "url": str(src.relative_to(REPO_ROOT)),
                    "downloaded": False,
                    "result": f"{type(exc).__name__}: {exc}",
                    "attempted_at": utc_now(),
                }
            )
            log(f"  extraction of {src.name} failed: {exc}")

    if not any(a.get("downloaded") for a in attempts):
        for src in MFNET_SOURCES:
            entry = dict(src)
            entry["attempted_at"] = utc_now()
            try:
                download_gdrive_folder(src["url"], out_root, timeout_min)
                downloaded_archive = None
                entry["downloaded"] = True
                entry["result"] = "folder downloaded"
                attempts.append(entry)
                break
            except Exception as exc:  # noqa: BLE001 -- we record the failure verbatim
                entry["downloaded"] = False
                entry["result"] = f"{type(exc).__name__}: {exc}"
                attempts.append(entry)
                log(f"  - {src['url']}: {entry['result']}")

    if not any(a.get("downloaded") for a in attempts):
        record = {
            "dataset_id": "MFNet",
            "status": "unavailable",
            "retrieved_at": utc_now(),
            "licence": LICENCE_NOTES["MFNet"],
            "attempts": attempts,
            "root": None,
            "splits": {},
            "manifest_sha256": None,
            "notes": "Both approved sources failed. Real-data training is BLOCKED.",
        }
        registry_file.write_text(json.dumps(record, indent=2))
        return record

    # ---- organise into a canonical layout ------------------------------- #
    log("MFNet: normalising layout")
    pair_dirs = find_pair_dirs(out_root)
    if not pair_dirs:
        # maybe a zip landed inside
        for z in sorted(out_root.rglob("*.zip")):
            try:
                safe_extract(z, out_root / "extracted")
            except Exception as exc:  # noqa: BLE001
                log(f"  extraction failed for {z.name}: {exc}")
        pair_dirs = find_pair_dirs(out_root)

    if not pair_dirs:
        record = {
            "dataset_id": "MFNet",
            "status": "corrupt",
            "retrieved_at": utc_now(),
            "licence": LICENCE_NOTES["MFNet"],
            "attempts": attempts,
            "root": str(out_root),
            "splits": {},
            "manifest_sha256": None,
            "notes": "Download completed but no ir/vi directory pair could be found.",
        }
        registry_file.write_text(json.dumps(record, indent=2))
        return record

    canonical = out_root / "canonical"
    for split_name in ("train", "val"):
        for mod in ("ir", "vi"):
            (canonical / split_name / mod).mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    n_rejected = 0
    for ir_dir, vi_dir in pair_dirs:
        pairs, leftover = pair_up(ir_dir, vi_dir)
        if leftover:
            log(f"  warning: {ir_dir.name}/{vi_dir.name}: {len(leftover)} unpaired files")
        for stem, ir_name, vi_name in pairs:
            if stem in seen:
                # same scene id appearing in two subset dirs (e.g. day/ and
                # night/) -- keep the first occurrence so the pair set is stable
                n_rejected += 1
                continue
            seen.add(stem)
            shutil.copy2(ir_dir / ir_name, canonical / "ir" / f"{stem}.png")
            shutil.copy2(vi_dir / vi_name, canonical / "vi" / f"{stem}.png")
    log(f"  {len(seen)} unique pairs collected"
        + (f" ({n_rejected} duplicate ids skipped)" if n_rejected else ""))
    if not seen:
        raise RuntimeError(
            "no validated IR/VI pairs found after dimension checking; "
            "refusing to register an empty dataset"
        )

    pairs = sorted(seen)
    splits = build_hashesplit(pairs)
    for split_name, ids in splits.items():
        d = canonical / split_name
        for pid in ids:
            name = f"{pid}.png"
            shutil.move(str(canonical / "ir" / name), str(d / "ir" / name))
            shutil.move(str(canonical / "vi" / name), str(d / "vi" / name))
    for mod in ("ir", "vi"):
        stale = sorted(p for p in (canonical / mod).iterdir() if p.is_file())
        if stale:
            raise RuntimeError(
                f"{len(stale)} files left unsplit in {mod}/ -- split logic and "
                f"pair set disagree; refusing to register"
            )

    entries = [f"{s}/{pid}" for s, ids in splits.items() for pid in ids]

    # --- image-content audit ------------------------------------------- #
    # Every registered pair must be loadable and share pixel dimensions. This is
    # re-checkable later via `download_data.py --verify-only`, which recomputes it.
    audit = audit_dataset(canonical)
    if audit["n_pairs"] == 0:
        raise RuntimeError(f"registered dataset has no loadable pairs: {audit}")
    log(
        f"  audit: {audit['n_pairs']} pairs, dims="
        f"{audit['unique_dims']}, ir_channels={audit['ir_channels']}, "
        f"vi_channels={audit['vi_channels']}"
    )

    record = {
        "dataset_id": "MFNet",
        "status": "ok",
        "retrieved_at": utc_now(),
        "licence": LICENCE_NOTES["MFNet"],
        "attempts": attempts,
        "archive_sha256": (
            sha256_file(downloaded_archive) if downloaded_archive else None
        ),
        "archive_bytes": (
            downloaded_archive.stat().st_size if downloaded_archive else None
        ),
        "audit": audit,
        "root": str(canonical),
        "split_definition": (
            "sha1(pair_id) -> uint32/first 8 hex digits, ascending; the lowest 20% "
            "of pair ids form 'val', the remainder 'train'. Pairs are grouped before "
            "splitting so IR and VI of a scene never straddle train/val."
        ),
        "split_val_frac": 0.2,
        "splits": {k: len(v) for k, v in splits.items()},
        "manifest_sha256": manifest_hash(entries),
        "notes": (
            "Fallback dataset. MSRS was attempted first and is unavailable; these "
            "numbers must be reported as MFNet results and are not comparable to "
            "published MSRS numbers."
        ),
    }
    registry_file.write_text(json.dumps(record, indent=2))
    log(f"MFNet: status=ok, splits={record['splits']} -> registry written")
    return record


# --------------------------------------------------------------------------- #
def verify(record_path: Path) -> int:
    """Re-check a registry entry against the files on disk. Exit code = health."""
    rec = json.loads(record_path.read_text())
    ds = rec["dataset_id"]
    log(f"verify {ds}: status={rec.get('status')}")
    if rec.get("status") != "ok":
        log("  not acquired -- see 'notes' and 'attempts' in the registry")
        return 1
    root = Path(rec["root"])
    if not root.exists():
        log(f"  FAIL: root {root} missing")
        return 1
    splits = rec["splits"]
    for split, expected in splits.items():
        ir_d = root / split / "ir"
        vi_d = root / split / "vi"
        if not (ir_d.exists() and vi_d.exists()):
            log(f"  FAIL: {split}: missing ir/vi dirs")
            return 1
        n_ir = len(list_images(ir_d))
        n_vi = len(list_images(vi_d))
        if n_ir != n_vi or n_ir != expected:
            log(f"  FAIL: {split}: ir={n_ir} vi={n_vi} expected={expected}")
            return 1
        log(f"  ok {split}: {n_ir} pairs")

    # Split-partition leakage: no pair id may appear in two splits.
    # Only the train/val splits take part in the manifest. Auxiliary subsets
    # (e.g. MSRS's `detection/`, used only for the mAP metric) are covered by
    # their own counts above; including them would make the manifest depend on
    # which auxiliary subsets happen to be present, and the write-side and
    # verify-side would disagree about whether they belong.
    manifest_splits = {
        k: v for k, v in splits.items() if k in ("train", "val", "test", "valid")
    }
    all_ids = {split: set(list_images(root / split / "ir")) for split in manifest_splits}
    names = sorted(all_ids)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            inter = all_ids[names[i]] & all_ids[names[j]]
            if inter:
                log(f"  FAIL: leakage {len(inter)} ids shared by {names[i]}/{names[j]}")
                return 1
    log("  ok no train/val leakage")

    # Content-level leakage. Two different pair ids can still be the *same
    # picture* -- MFNet has day/night subsets and near-duplicate frames, so an
    # identical (or near-identical) infrared image can end up on both sides.
    # That is the failure that silently inflates every metric, so it is checked
    # directly on pixel content rather than assumed absent.
    dup = find_content_duplicates(root, splits)
    if dup:
        log(f"  FAIL: {len(dup)} pairs share identical IR content across splits")
        for d in dup[:5]:
            log(f"        {d['ir']} == {d['other']}")
        return 1
    log("  ok no cross-split IR content duplicates")

    entries = [f"{s}/{pid}" for s, ids in all_ids.items() for pid in sorted(ids)]
    recomputed = manifest_hash(entries)
    if recomputed != rec.get("manifest_sha256"):
        log(f"  FAIL: manifest hash mismatch (disk={recomputed[:12]} "
            f"registry={str(rec.get('manifest_sha256'))[:12]})")
        return 1
    log("  ok manifest hash matches registry")

    # Re-run the content audit and compare with what was recorded at
    # acquisition. This catches on-disk corruption or later edits.
    audit = audit_dataset(root)
    recorded = rec.get("audit")
    if recorded:
        if audit["n_pairs"] != recorded.get("n_pairs"):
            log(f"  FAIL: audit pair count changed "
                f"(registry={recorded.get('n_pairs')} disk={audit['n_pairs']})")
            return 1
        if audit["n_problems"]:
            log(f"  FAIL: audit found {audit['n_problems']} problem(s): "
                f"{audit['problems'][:3]}")
            return 1
        log(f"  ok content audit clean ({audit['n_pairs']} loadable pairs)")
    return 0


def find_content_duplicates(root: Path, splits: dict[str, int]) -> list[dict[str, str]]:
    """Pairs in different splits whose infrared image is byte-identical.

    Uses a per-file SHA-256 rather than a perceptual hash: byte equality is
    unambiguous, cheap, and cannot produce a false accusation. Near-duplicates
    are *not* reported here, because a perceptual hash threshold would itself be
    an arbitrary, hard-to-defend choice; docs/DATASET.md records this limitation
    instead of hiding it.
    """
    by_split: dict[str, dict[str, str]] = {}
    for split in splits:
        digests: dict[str, str] = {}
        for name in sorted(list_images(root / split / "ir")):
            try:
                digests[file_digest(root / split / "ir" / name)] = f"{split}/{name}"
            except OSError:
                continue
        by_split[split] = digests

    out: list[dict[str, str]] = []
    keys = sorted(by_split)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = by_split[keys[i]], by_split[keys[j]]
            shared = set(a) & set(b)
            for h in sorted(shared):
                out.append({"ir": a[h], "other": b[h]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["auto", "msrs", "mfnet"], default="auto")
    ap.add_argument("--timeout-min", type=float, default=10.0)
    ap.add_argument("--force", action="store_true", help="re-download even if registered")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "README.md").write_text(
        "# data/\n\nThis directory is populated by `scripts/download_data.py`.\n"
        "Image files are git-ignored; the provenance registry in `data/registry/`\n"
        "is committed. See docs/DATASET.md.\n"
    )

    if args.verify_only:
        rc = 0
        for f in sorted(REGISTRY_DIR.glob("*.json")):
            rc |= verify(f)
        return 1 if rc else 0

    if args.dataset in ("auto", "msrs"):
        msrs = attempt_msrs(timeout_min=args.timeout_min, force=args.force)
        if msrs.get("status") == "ok":
            log("")
            log("MSRS acquired -- no fallback needed.")
            return 0 if not args.verify_only else 0
        log("MSRS unavailable; falling back to MFNet")
        log("")

    if args.dataset == "mfnet":
        mf = attempt_mfnet(timeout_min=args.timeout_min, force=args.force)
        if mf.get("status") != "ok":
            log("")
            log("BLOCKED: no real dataset is available.")
            log("Training on synthetic data is not permitted by the project spec.")
            return 2

    # final verification of whatever we have
    rc = 0
    for f in sorted(REGISTRY_DIR.glob("*.json")):
        rc |= verify(f)
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())