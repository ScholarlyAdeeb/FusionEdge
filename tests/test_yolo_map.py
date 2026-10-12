"""mAP@0.5 correctness.

mAP is the metric most likely to be quietly wrong: it depends on matching,
thresholding and interpolation choices that nobody states. These tests pin the
routine against hand-computed cases, and pin the honesty rule -- that a missing
annotation set yields `status: unavailable` rather than a number.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from fusionedge.metrics.yolo_map import (
    YoloConfig,
    average_precision,
    iou_matrix,
    load_annotations,
    mean_average_precision,
    unavailable,
)


# --------------------------------------------------------------------------- #
def test_iou_matrix_basics():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    b = np.array([[20, 20, 30, 30]], dtype=float)  # disjoint
    assert iou_matrix(a, b)[0, 0] == pytest.approx(0.0)
    c = np.array([[0, 0, 10, 10]], dtype=float)
    d = np.array([[5, 0, 15, 10]], dtype=float)  # half overlap: 50/150
    assert iou_matrix(c, d)[0, 0] == pytest.approx(50 / 150, rel=1e-9)


def test_iou_matrix_handles_empty():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((1, 4))).shape == (0, 1)
    assert iou_matrix(np.zeros((1, 4)), np.zeros((0, 4))).shape == (1, 0)


# --------------------------------------------------------------------------- #
def test_ap_perfect_detection_is_one():
    det = {"img": np.array([[0, 0, 10, 10, 0.9]], dtype=float)}
    gt = {"img": np.array([[0, 0, 10, 10]], dtype=float)}
    assert average_precision(det, gt) == pytest.approx(1.0)


def test_ap_missed_detection_is_zero():
    det = {"img": np.array([[50, 50, 60, 60, 0.9]], dtype=float)}
    gt = {"img": np.array([[0, 0, 10, 10]], dtype=float)}
    assert average_precision(det, gt) == pytest.approx(0.0)


def test_ap_respects_the_iou_threshold():
    """A box overlapping just under 0.5 IoU must not count as a hit."""
    det = {"img": np.array([[0, 0, 10, 10, 0.9]], dtype=float)}
    gt = {"img": np.array([[5.5, 0, 15.5, 10]], dtype=float)}  # IoU = 45/155
    assert iou_matrix(det["img"], gt["img"])[0, 0] < 0.5
    assert average_precision(det, gt) == pytest.approx(0.0)
    assert average_precision(det, gt, iou_thr=0.2) == pytest.approx(1.0)


def test_ap_penalises_false_positives_by_where_they_are_ranked():
    """VOC-style all-point interpolation has a genuine ranking asymmetry, and it
    is worth pinning explicitly rather than discovering it from a paper table:

      * a false positive ranked ABOVE the true positive drops AP to 0.5, because
        precision at the moment of the hit is 1/2;
      * a false positive ranked BELOW every true positive leaves AP at 1.0,
        because the interpolated precision envelope only asks "what was the best
        precision at or above this recall".

    Anyone comparing two models with slightly different confidence calibration
    needs to know this: trailing low-confidence junk does not show up in AP.
    """
    gt = {"img": np.array([[0, 0, 10, 10]], dtype=float)}

    fp_first = {"img": np.array([[50, 50, 60, 60, 0.95], [0, 0, 10, 10, 0.80]], dtype=float)}
    assert average_precision(fp_first, gt) == pytest.approx(0.5, abs=1e-9)

    fp_last = {"img": np.array([[0, 0, 10, 10, 0.95], [50, 50, 60, 60, 0.80]], dtype=float)}
    assert average_precision(fp_last, gt) == pytest.approx(1.0, abs=1e-9)


def test_ap_rewards_ranking():
    """Same detections, better ranking -> higher AP. Without this, the metric is
    insensitive to confidence ordering and cannot distinguish models."""
    gt = {"a": np.array([[0, 0, 10, 10]], float), "b": np.array([[0, 0, 10, 10]], float)}
    good = {
        "a": np.array([[0, 0, 10, 10, 0.9]], float),
        "b": np.array([[50, 50, 60, 60, 0.8]], float),  # false positive ranked last
    }
    bad = {
        "a": np.array([[50, 50, 60, 60, 0.9]], float),  # false positive ranked first
        "b": np.array([[0, 0, 10, 10, 0.8]], float),
    }
    assert average_precision(good, gt) > average_precision(bad, gt)


def test_ap_is_nan_without_ground_truth():
    """Undefined, not zero: a class with no positives has no AP."""
    assert np.isnan(average_precision({}, {}))


def test_map_averages_over_classes():
    det = {
        "c0": {"i": np.array([[0, 0, 10, 10, 0.9]], float)},
        "c1": {"i": np.array([[0, 0, 10, 10, 0.9]], float)},
    }
    gt = {
        "c0": {"i": np.array([[0, 0, 10, 10]], float)},
        "c1": {"i": np.array([[50, 50, 60, 60]], float)},  # never detected
    }
    res = mean_average_precision(det, gt)
    assert res["num_classes"] == 2
    assert res["mAP50"] == pytest.approx(0.5, abs=1e-9)


def test_map_with_no_detections_is_nan_not_zero():
    res = mean_average_precision({}, {"c0": {"i": np.array([[0, 0, 1, 1]], float)}})
    assert np.isnan(res["mAP50"])


# --------------------------------------------------------------------------- #
# annotation loading
# --------------------------------------------------------------------------- #
def test_load_annotations_coco_format(tmp_path):
    p = tmp_path / "ann.json"
    p.write_text(json.dumps({
        "images": [{"id": 1}],
        "categories": [{"id": 7, "name": "person"}],
        "annotations": [
            {"image_id": 1, "category_id": 7, "bbox": [10, 20, 30, 40]},
        ],
    }))
    ann = load_annotations(p)
    assert "person" in ann
    assert ann["person"]["1"].shape == (1, 4)
    np.testing.assert_allclose(ann["person"]["1"][0], [10, 20, 40, 60])


def test_load_annotations_per_image_directory(tmp_path):
    d = tmp_path / "anno_json"
    d.mkdir()
    (d / "0001.json").write_text(json.dumps({
        "image_id": "0001",
        "annotations": [{"bbox": [0, 0, 5, 5], "category": "car"}],
    }))
    ann = load_annotations(d)
    np.testing.assert_allclose(ann["car"]["0001"][0], [0, 0, 5, 5])


def test_load_annotations_supports_xyxy_format(tmp_path):
    d = tmp_path / "anno_json"
    d.mkdir()
    (d / "0002.json").write_text(json.dumps({
        "image_id": "0002",
        "annotations": [{"xmin": 1, "ymin": 2, "xmax": 5, "ymax": 9, "category": "bike"}],
    }))
    ann = load_annotations(d)
    np.testing.assert_allclose(ann["bike"]["0002"][0], [1, 2, 5, 9])


def test_missing_annotations_raise_a_clear_error(tmp_path):
    """The honesty rule: absent ground truth must be an explicit failure so the
    caller reports 'unavailable', never a fabricated mAP."""
    with pytest.raises(FileNotFoundError, match="cannot be computed without"):
        load_annotations(tmp_path / "nothing_here.json")


def test_unavailable_payload_has_no_numbers():
    res = unavailable("annotations missing for this split")
    assert res["status"] == "unavailable"
    assert "mAP50" not in res
    assert all(isinstance(v, str) for v in res.values())


# --------------------------------------------------------------------------- #
def test_nms_removes_duplicates():
    from fusionedge.metrics.yolo_map import _nms

    dets = [
        {"box": np.array([0, 0, 10, 10], float), "score": 0.9, "class": 0},
        {"box": np.array([0.5, 0, 10.5, 10], float), "score": 0.8, "class": 0},
        {"box": np.array([50, 50, 60, 60], float), "score": 0.7, "class": 0},
    ]
    kept = _nms(dets, 0.5, 0.001)
    assert len(kept) == 2, "near-duplicate must be suppressed"
    assert kept[0]["score"] == pytest.approx(0.9), "the highest score survives"


def test_nms_keeps_different_classes():
    from fusionedge.metrics.yolo_map import _nms

    dets = [
        {"box": np.array([0, 0, 10, 10], float), "score": 0.9, "class": 0},
        {"box": np.array([0, 0, 10, 10], float), "score": 0.8, "class": 1},
    ]
    assert len(_nms(dets, 0.5, 0.001)) == 2


def test_backend_resolution_reports_all_failures(monkeypatch):
    """If no backend is usable the error must name both, not degrade silently."""
    import fusionedge.metrics.yolo_map as ym

    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("not installed")

    monkeypatch.setattr(ym, "UltralyticsBackend", Boom)
    monkeypatch.setattr(ym, "OnnxRuntimeBackend", Boom)
    with pytest.raises(RuntimeError) as exc:
        ym.resolve_backend(YoloConfig())
    assert "ultralytics" in str(exc.value) and "onnxruntime" in str(exc.value)