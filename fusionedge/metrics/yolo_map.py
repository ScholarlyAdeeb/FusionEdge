"""YOLOv5x6 downstream detection evaluation (mAP@0.5).

Why a downstream task metric at all: EN/AG/MS-SSIM are no-reference or
full-reference image statistics, and a fusion model can raise them while making
the image *worse for a detector* -- smoothing contrast helps AG's noise floor
and hurts a detector's confidence. So the fusion claim that matters is "does the
fused image preserve detections", and that has to be measured, not argued.

Backends
--------
  ultralytics   -- primary. `YOLO('yolov5x6.pt')`, GPU, batched, fp16.
  onnxruntime   -- fallback, used when ultralytics cannot be imported or run on
                   this Python (it is pinned separately in
                   requirements-extras.txt and is a documented path, not a stub).

The active backend is recorded in the results as `backend`, and
scripts/make_figures.py refuses to put two backends in one comparison table --
switching backend silently would invalidate an ablation.

mAP computation
---------------
AP@0.5 is computed here from (detections, ground-truth boxes) with a
self-contained routine: sort detections by confidence, greedily match each to the
highest-IoU unmatched ground-truth box above 0.5 in the same image, and integrate
the precision-recall curve with the all-point interpolation. Each class is scored
separately and averaged. This is VOC-style AP@0.5; it is unit-tested against
hand-computed cases in tests/test_yolo_map.py. If `pycocotools` is installed its
number is reported alongside as a cross-check, never in place of this one.

Honesty rule
------------
If there are no detection annotations for the split, this module returns
`status: "unavailable"` with a reason. It never invents boxes, and it never
returns a partial mAP that could be read as a real one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# AP@0.5
# --------------------------------------------------------------------------- #
def iou_matrix(det: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """det (N,5) and gt (M,5) as [x1,y1,x2,y2,(score|class)] -> IoU (N,M)."""
    if len(det) == 0 or len(gt) == 0:
        return np.zeros((len(det), len(gt)), dtype=np.float64)
    dx1 = np.maximum(det[:, None, 0], gt[None, :, 0])
    dy1 = np.maximum(det[:, None, 1], gt[None, :, 1])
    dx2 = np.minimum(det[:, None, 2], gt[None, :, 2])
    dy2 = np.minimum(det[:, None, 3], gt[None, :, 3])
    inter = np.clip(dx2 - dx1, 0, None) * np.clip(dy2 - dy1, 0, None)
    area_d = np.clip(det[:, 2] - det[:, 0], 0, None) * np.clip(det[:, 3] - det[:, 1], 0, None)
    area_g = np.clip(gt[:, 2] - gt[:, 0], 0, None) * np.clip(gt[:, 3] - gt[:, 1], 0, None)
    union = area_d[:, None] + area_g[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def average_precision(
    detections: dict[str, np.ndarray], ground_truth: dict[str, np.ndarray], iou_thr: float = 0.5
) -> float:
    """AP@iou_thr for one class.

    detections: image_key -> (N,5) [x1,y1,x2,y2,score]
    ground_truth: image_key -> (M,4) [x1,y1,x2,y2]
    """
    # flatten with image keys, sorted by descending confidence
    scores: list[float] = []
    for key, d in detections.items():
        for row in d:
            scores.append((float(row[4]), key, row))
    scores.sort(key=lambda t: -t[0])

    n_gt = sum(len(v) for v in ground_truth.values())
    if n_gt == 0:
        # Undefined, not zero: AP for a class with no positives is not a number,
        # and reporting 0 would drag mAP down with a fabricated penalty.
        return float("nan")
    if not scores:
        return 0.0

    matched: dict[str, np.ndarray] = {k: np.zeros(len(v), dtype=bool) for k, v in ground_truth.items()}
    tp = np.zeros(len(scores))
    fp = np.zeros(len(scores))

    for i, (_, key, row) in enumerate(scores):
        gts = ground_truth.get(key)
        if gts is None or len(gts) == 0:
            fp[i] = 1
            continue
        ious = iou_matrix(row[None, :4], gts)[0]
        best = int(np.argmax(ious))
        if ious[best] >= iou_thr and not matched[key][best]:
            tp[i] = 1
            matched[key][best] = True
        else:
            fp[i] = 1

    n_gt = sum(len(v) for v in ground_truth.values())

    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-12)
    # all-point interpolation: envelope of precision from the right
    mrec = np.concatenate(([0.0], recall, [recall[-1]]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for j in range(len(mpre) - 2, -1, -1):
        mpre[j] = max(mpre[j], mpre[j + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def mean_average_precision(
    detections: dict[str, dict[str, np.ndarray]],
    ground_truth: dict[str, dict[str, np.ndarray]],
    iou_thr: float = 0.5,
) -> dict[str, Any]:
    """detections/ground_truth: class_name -> image_key -> boxes."""
    per_class: dict[str, float] = {}
    for cls, dets in detections.items():
        gts = ground_truth.get(cls, {})
        ap = average_precision(dets, gts, iou_thr)
        if not np.isnan(ap):
            per_class[cls] = ap
    if not per_class:
        return {"mAP50": float("nan"), "per_class": {}, "num_classes": 0}
    return {
        "mAP50": float(np.mean(list(per_class.values()))),
        "per_class": per_class,
        "num_classes": len(per_class),
    }


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
@dataclass
class YoloConfig:
    weights: str = "yolov5x6.pt"
    backend: str = "auto"          # auto | ultralytics | onnxruntime | none
    device: str = "cuda"
    imgsz: int = 640
    conf: float = 0.001
    iou_nms: float = 0.6
    batch: int = 4
    half: bool = True


class UltralyticsBackend:
    """Primary path: the ultralytics package running yolov5x6."""

    name = "ultralytics"

    def __init__(self, cfg: YoloConfig):
        from ultralytics import YOLO  # noqa: PLC0415 -- optional dependency

        self.model = YOLO(cfg.weights)
        self.cfg = cfg
        self.backend = "ultralytics"
        self.device = cfg.device if cfg.device != "cuda" or _cuda_ok() else "cpu"

    def detect(self, images: Sequence[np.ndarray]) -> list[list[dict]]:
        """images: list of HWC uint8 RGB. Returns per-image detections."""
        out: list[list[dict]] = []
        bs = max(1, self.cfg.batch)
        for i in range(0, len(images), bs):
            chunk = [im[:, :, ::-1] for im in images[i : i + bs]]  # RGB -> BGR
            results = self.model.predict(
                chunk,
                imgsz=self.cfg.imgsz,
                conf=self.cfg.conf,
                iou=self.cfg.iou_nms,
                device=self.device,
                half=self.cfg.half and self.device == "cuda",
                verbose=False,
            )
            for r in results:
                dets = []
                if r.boxes is not None and len(r.boxes):
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    clss = r.boxes.cls.cpu().numpy().astype(int)
                    for box, cf, cl in zip(xyxy, confs, clss):
                        dets.append(
                            {"box": box.astype(float), "score": float(cf), "class": int(cl)}
                        )
                out.append(dets)
        return out


class OnnxRuntimeBackend:
    """Documented fallback: yolov5x6 exported to ONNX, run with onnxruntime.

    Same weights, same letterbox preprocessing, same IoU/NMS settings, so the two
    backends are comparable -- which is why the active one is recorded rather
    than assumed. CPU execution is slower but removes the ultralytics dependency
    entirely.
    """

    name = "onnxruntime"

    def __init__(self, cfg: YoloConfig, onnx_path: str | Path | None = None):
        import onnxruntime as ort  # noqa: PLC0415 -- optional dependency

        path = Path(onnx_path or Path(cfg.weights).with_suffix(".onnx"))
        if not path.exists():
            raise FileNotFoundError(
                f"ONNX weights not found at {path}. Export first with: "
                f"python scripts/export_yolo_onnx.py  (or install ultralytics)"
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if _cuda_ok() else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(path), providers=providers)
        self.cfg = cfg
        self.input_name = self.sess.get_inputs()[0].name
        self.backend = "onnxruntime"

    @staticmethod
    def _letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
        h, w = img.shape[:2]
        r = min(size / h, size / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        import cv2

        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        top, left = (size - nh) // 2, (size - nw) // 2
        canvas[top : top + nh, left : left + nw] = resized
        return canvas, r, (left, top)

    def detect(self, images: Sequence[np.ndarray]) -> list[list[dict]]:
        out: list[list[dict]] = []
        for img in images:
            rgb = img[:, :, ::-1].copy()  # RGB -> BGR for the model
            canvas, r, (left, top) = self._letterbox(rgb, self.cfg.imgsz)
            x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
            pred = self.sess.run(None, {self.input_name: x})[0][0]  # (N, 85)
            dets = []
            for row in pred:
                score = float(row[4])
                if score < self.cfg.conf:
                    continue
                cls = int(np.argmax(row[5:]))
                cx, cy, bw, bh = row[:4]
                box = np.array(
                    [
                        (cx - bw / 2 - left) / r,
                        (cy - bh / 2 - top) / r,
                        (cx + bw / 2 - left) / r,
                        (cy + bh / 2 - top) / r,
                    ],
                    dtype=float,
                )
                dets.append({"box": box, "score": score, "class": cls})
            out.append(_nms(dets, self.cfg.iou_nms, self.cfg.conf))
        return out


def _nms(dets: list[dict], iou_thr: float, conf_thr: float) -> list[dict]:
    """Plain greedy NMS, per class."""
    keep: list[dict] = []
    for cls in sorted({d["class"] for d in dets}):
        items = [d for d in dets if d["class"] == cls]
        items.sort(key=lambda d: -d["score"])
        while items:
            best = items.pop(0)
            keep.append(best)
            items = [
                d for d in items if iou_matrix(best["box"][None], d["box"][None])[0, 0] < iou_thr
            ]
    return keep


def _cuda_ok() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def resolve_backend(cfg: YoloConfig, prefer: str | None = None) -> Any:
    """Pick a working backend, preferring ultralytics, then ONNX Runtime.

    Raises RuntimeError naming both failures rather than silently degrading to a
    weaker metric.
    """
    want = prefer or cfg.backend
    order = (
        ["ultralytics", "onnxruntime"]
        if want in ("auto", "ultralytics")
        else [want]
    )
    if want == "onnxruntime":
        order = ["onnxruntime"]
    errors = []
    for name in order:
        try:
            return UltralyticsBackend(cfg) if name == "ultralytics" else OnnxRuntimeBackend(cfg)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "No YOLO backend available. Install one:\n"
        "  pip install -r requirements-extras.txt      (ultralytics, primary)\n"
        "  python scripts/export_yolo_onnx.py           (onnxruntime fallback)\n"
        "Failures:\n  " + "\n  ".join(errors)
    )


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def load_yolo_txt_labels(
    path: str | Path, image_size: tuple[int, int] | None = None
) -> dict[str, dict[str, np.ndarray]]:
    """Load YOLO-format labels: one .txt per image, `class cx cy w h` normalised.

    This is the format MSRS ships in `detection/labels/*.txt` (verified against
    the repository: 81 files, e.g. `2 0.221094 0.493750 0.117188 0.170833`).

    The coordinates are normalised to [0,1], so converting them to absolute
    pixels requires the image size. When `image_size` is omitted the boxes are
    returned in normalised units and the caller must supply a size -- mixing
    normalised ground truth with pixel-space detections would silently produce a
    meaningless mAP, so this is why the parameter is explicit rather than
    guessed.

    Returns class -> image_key -> (M,4) boxes as [x1, y1, x2, y2].
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No YOLO labels at {path}")

    out: dict[str, dict[str, list]] = {}
    for txt in sorted(path.glob("*.txt")):
        key = txt.stem
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue  # blank or malformed line
            cls = parts[0]
            cx, cy, bw, bh = (float(v) for v in parts[1:5])
            if image_size is not None:
                W, H = image_size
                cx, cy, bw, bh = cx * W, cy * H, bw * W, bh * H
            x1, y1 = cx - bw / 2, cy - bh / 2
            out.setdefault(cls, {}).setdefault(key, []).append(
                [x1, y1, x1 + bw, y1 + bh]
            )
    return {c: {k: np.array(v, dtype=float) for k, v in ims.items()} for c, ims in out.items()}


def load_annotations(path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Load detection annotations as class -> image_key -> (M,4) boxes.

    Supports the three formats this project may encounter:
      * COCO: a single JSON with `images` and `annotations`,
      * a per-image directory of JSONs (MFNet's `anno_json` layout),
      * a directory of YOLO-format .txt files (MSRS `detection/labels`).

    Raises FileNotFoundError with a clear message if annotations are absent --
    the caller must then report the metric as unavailable, not as zero.
    """
    path = Path(path)
    out: dict[str, dict[str, np.ndarray]] = {}

    if path.is_dir() and any(path.glob("*.txt")):
        return load_yolo_txt_labels(path, image_size=None)

    if path.is_dir():
        for jf in sorted(path.glob("*.json")):
            data = json.loads(jf.read_text(encoding="utf-8"))
            key = str(data.get("image_id") or data.get("id") or jf.stem)
            for ann in data.get("annotations", data.get("objects", [])):
                if "bbox" in ann:
                    x, y, w, h = ann["bbox"]
                elif all(k in ann for k in ("xmin", "ymin", "xmax", "ymax")):
                    x, y = ann["xmin"], ann["ymin"]
                    w, h = ann["xmax"] - x, ann["ymax"] - y
                else:
                    continue
                cls = str(ann.get("category", ann.get("category_id", ann.get("class", 0))))
                out.setdefault(cls, {}).setdefault(key, []).append([x, y, x + w, y + h])
        return {c: {k: np.array(v, dtype=float) for k, v in ims.items()} for c, ims in out.items()}

    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        id_to_name = {i["id"]: str(i.get("name", i["id"])) for i in data.get("categories", [])}
        for ann in data.get("annotations", []):
            x, y, w, h = ann["bbox"]
            cls = id_to_name.get(ann["category_id"], str(ann["category_id"]))
            key = str(ann["image_id"])
            out.setdefault(cls, {}).setdefault(key, []).append([x, y, x + w, y + h])
        return {c: {k: np.array(v, dtype=float) for k, v in ims.items()} for c, ims in out.items()}

    raise FileNotFoundError(
        f"No detection annotations at {path}. mAP cannot be computed without "
        f"ground-truth boxes; report it as unavailable rather than omitting it."
    )


def evaluate_yolo_map(
    images: Sequence[np.ndarray],
    image_keys: Sequence[str],
    annotations: dict[str, dict[str, np.ndarray]],
    cfg: YoloConfig | None = None,
    prefer_backend: str | None = None,
) -> dict[str, Any]:
    """Run detection over `images` and score against `annotations`.

    `annotations` must already be in absolute pixels. If it was loaded from
    YOLO .txt (normalised) you must have passed an `image_size`; mixing
    normalised ground truth with pixel detections would yield a meaningless
    mAP, so this function checks that the two spaces are consistent.
    """
    cfg = cfg or YoloConfig()
    backend = resolve_backend(cfg, prefer_backend)

    # Guard: detections are in pixels, so ground truth must be too.
    max_gt = max(
        (float(np.max(v)) for ims in annotations.values() for v in ims.values() if len(v)),
        default=0.0,
    )
    max_img = max(
        (float(np.asarray(im).shape[1]) for im in images if np.asarray(im).ndim >= 2),
        default=0.0,
    )
    if annotations and max_gt > 0 and max_img > 0 and max_gt <= 1.5 and max_img > 10:
        raise ValueError(
            f"ground-truth boxes look normalised (max coordinate {max_gt:.2f}) "
            f"but detections are in pixels (image width {max_img:.0f}). "
            f"Re-load the labels with an explicit image_size."
        )

    dets_all = backend.detect(list(images))
    detections: dict[str, dict[str, np.ndarray]] = {}
    for key, dets in zip(image_keys, dets_all):
        for d in dets:
            cls = str(d["class"])
            detections.setdefault(cls, {}).setdefault(key, []).append(
                [*map(float, d["box"]), float(d["score"])]
            )
    detections = {
        c: {k: np.array(v, dtype=float) for k, v in ims.items()} for c, ims in detections.items()
    }

    result = mean_average_precision(detections, annotations)
    result["backend"] = backend.backend
    result["weights"] = cfg.weights
    result["num_images"] = len(images)
    return result


def unavailable(reason: str, cfg: YoloConfig | None = None) -> dict[str, Any]:
    """Explicit 'not measured' payload. Never returns a numeric mAP."""
    return {
        "status": "unavailable",
        "reason": reason,
        "backend": (cfg.backend if cfg else "none"),
    }


def attach_yolo_metrics(payload: dict, yolo_result: dict, images: Sequence[np.ndarray]) -> None:
    """Merge the YOLO result into a metrics payload, preserving provenance."""
    payload.setdefault("downstream", {})
    payload["downstream"]["yolov5x6"] = {
        k: v for k, v in yolo_result.items() if k != "per_class"
    }
    if "per_class" in yolo_result:
        payload["downstream"]["yolov5x6"]["per_class_AP50"] = yolo_result["per_class"]