"""Result-table generation must never hand you a misleading table.

The properties pinned here are the ones that keep a README honest:

  * rows from different datasets can never share a table,
  * rows computed on different dataset manifests can never share a table,
  * two different YOLO inference backends can never share a table,
  * smoke runs are excluded unless explicitly requested, and then labelled,
  * an ablation delta is computed against the full model and signed correctly,
  * a missing number renders as `--`, never as 0.

No numbers are invented here. Every test writes its own metrics.json fixture and
checks the transformation.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fusionedge_make_figures", REPO_ROOT / "scripts" / "make_figures.py"
)
mf = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(mf)


def write_run(
    exp: Path,
    name: str,
    metrics: dict,
    *,
    dataset_id: str = "MSRS",
    manifest: str = "a" * 64,
    smoke: bool = False,
    backend: str | None = None,
    params: int = 22339,
) -> Path:
    d = exp / name
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": {"name": name, "params": params},
        "dataset": {
            "dataset_id": dataset_id,
            "manifest_sha256": manifest,
            "splits": {"train": 1083, "val": 361},
        },
        "evaluation": {"num_images": 361, "elapsed_s": 1.0},
        "metrics": metrics,
        "smoke": smoke,
    }
    if backend:
        payload["downstream"] = {"yolov5x6": {"mAP50": 0.5, "backend": backend}}
    (d / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
    (d / "benchmark.json").write_text(
        json.dumps({
            "params": params,
            "checkpoint_mb": 0.09,
            "gpu": "NVIDIA GeForce RTX 4070 Laptop GPU",
            "latency": [
                {"size": [128, 128], "latency_ms_median": 13.6, "latency_ms_p95": 25.4, "fps": 73.6, "peak_vram_mb": 23},
                {"size": [480, 640], "latency_ms_median": 33.1, "latency_ms_p95": 33.2, "fps": 30.2, "peak_vram_mb": 286},
            ],
        }),
        encoding="utf-8",
    )
    return d


M_FULL = {
    "EN": 6.5, "AG": 0.0045, "SF": 0.011, "MI": 3.2,
    "MS_SSIM_ir": 0.85, "MS_SSIM_vi": 0.90,
}


@pytest.fixture
def exp(tmp_path: Path) -> Path:
    return tmp_path / "experiments"


# --------------------------------------------------------------------------- #
# provenance gating
# --------------------------------------------------------------------------- #
def test_refuses_to_mix_datasets(exp: Path):
    write_run(exp, "fusionedge_main", M_FULL, dataset_id="MSRS")
    write_run(exp, "baseline_simplecnn", M_FULL, dataset_id="MFNet")
    with pytest.raises(SystemExit, match="spanning datasets"):
        mf.check_single_dataset(mf.load_runs(exp, include_smoke=True))


def test_refuses_to_mix_manifests(exp: Path):
    write_run(exp, "fusionedge_main", M_FULL, manifest="a" * 64)
    write_run(exp, "baseline_simplecnn", M_FULL, manifest="b" * 64)
    with pytest.raises(SystemExit, match="manifests"):
        mf.check_single_dataset(mf.load_runs(exp, include_smoke=True))


def test_refuses_to_mix_yolo_backends(exp: Path):
    write_run(exp, "fusionedge_main", M_FULL, backend="ultralytics")
    write_run(exp, "baseline_simplecnn", M_FULL, backend="onnxruntime")
    with pytest.raises(SystemExit, match="backends"):
        mf.check_single_dataset(mf.load_runs(exp, include_smoke=True))


def test_accepts_a_single_consistent_dataset(exp: Path):
    write_run(exp, "fusionedge_main", M_FULL, backend="ultralytics")
    write_run(exp, "baseline_simplecnn", M_FULL, backend="ultralytics")
    ds, manifest, backend = mf.check_single_dataset(
        mf.load_runs(exp, include_smoke=True)
    )
    assert ds == "MSRS" and backend == "ultralytics"
    assert manifest == "a" * 64


def test_unavailable_backend_is_a_distinct_state(exp: Path):
    write_run(exp, "fusionedge_main", M_FULL)  # no downstream key
    ds, manifest, backend = mf.check_single_dataset(
        mf.load_runs(exp, include_smoke=True)
    )
    assert backend is None or backend == "unavailable"


# --------------------------------------------------------------------------- #
# smoke gating
# --------------------------------------------------------------------------- #
def test_smoke_runs_are_excluded_by_default(exp: Path):
    write_run(exp, "fusionedge_main", M_FULL)
    write_run(exp, "smoke_run", M_FULL, smoke=True)
    names = {r["_name"] for r in mf.load_runs(exp, include_smoke=False)}
    assert "smoke_run" not in names
    assert "fusionedge_main" in names


def test_smoke_runs_included_on_request(exp: Path):
    write_run(exp, "fusionedge_main", M_FULL)
    write_run(exp, "smoke_run", M_FULL, smoke=True)
    names = {r["_name"] for r in mf.load_runs(exp, include_smoke=True)}
    assert "smoke_run" in names


def test_smoke_inclusion_is_loudly_labelled(exp: Path, tmp_path: Path, capsys):
    write_run(exp, "fusionedge_main", M_FULL, smoke=True)
    mf.main.__globals__  # noqa: B018 - reference for clarity
    import sys

    argv = sys.argv
    sys.argv = ["make_figures.py", "--exp", str(exp), "--include-smoke"]
    try:
        mf.main()
    finally:
        sys.argv = argv
    out = capsys.readouterr().out
    assert "WARNING: smoke runs included" in out
    assert "NOT results" in out


# --------------------------------------------------------------------------- #
# table content
# --------------------------------------------------------------------------- #
def test_quality_table_contains_dataset_stamp(exp: Path):
    write_run(exp, "fusionedge_main", M_FULL)
    runs = mf.load_runs(exp, include_smoke=True)
    table = mf.quality_table(runs, "MSRS", "a" * 64, None)
    assert "MSRS" in table
    assert "a" * 16 in table
    assert "test" in table, "MSRS held-out split is named test"
    assert "do not edit by hand" in table


def test_quality_table_lists_only_real_metrics(exp: Path):
    write_run(exp, "fusionedge_main", M_FULL)
    runs = mf.load_runs(exp, include_smoke=True)
    table = mf.quality_table(runs, "MSRS", "a" * 64, None)
    assert "6.5000" in table  # EN
    assert "0.0045" in table  # AG


def test_ablation_delta_sign_is_correct(exp: Path):
    """An ablation *better* than the full model must show a POSITIVE delta."""
    worse = {**M_FULL, "EN": 6.0, "AG": 0.0040}
    write_run(exp, "fusionedge_main", M_FULL)
    write_run(exp, "ablation_no_edge", worse)
    runs = mf.load_runs(exp, include_smoke=True)
    table = mf.ablation_table(runs, "fusionedge_main")
    assert table is not None
    line = [ln for ln in table.splitlines() if "no edge" in ln][0]
    # EN: 6.0 - 6.5 = -0.5 (component was helping)
    assert "-0.5000" in line
    # AG: 0.0040 - 0.0045 = -0.0005
    assert "-0.0005" in line


def test_ablation_better_than_full_shows_positive_delta(exp: Path):
    better = {**M_FULL, "EN": 6.9}
    write_run(exp, "fusionedge_main", M_FULL)
    write_run(exp, "ablation_no_edge", better)
    runs = mf.load_runs(exp, include_smoke=True)
    table = mf.ablation_table(runs, "fusionedge_main")
    line = [ln for ln in table.splitlines() if "no edge" in ln][0]
    assert "+0.4000" in line


def test_ablation_missing_metric_renders_as_dashes_not_zero(exp: Path):
    partial = {"EN": 6.0}  # no AG, no MS-SSIM
    write_run(exp, "fusionedge_main", M_FULL)
    write_run(exp, "ablation_no_edge", partial)
    runs = mf.load_runs(exp, include_smoke=True)
    table = mf.ablation_table(runs, "fusionedge_main")
    line = [ln for ln in table.splitlines() if "no edge" in ln][0]
    assert "--" in line, "missing values must not be treated as zero"


def test_ablation_table_is_none_without_the_full_run(exp: Path):
    write_run(exp, "ablation_no_edge", M_FULL)
    runs = mf.load_runs(exp, include_smoke=True)
    assert mf.ablation_table(runs, "fusionedge_main") is None


def test_ablation_table_is_none_without_ablations(exp: Path):
    write_run(exp, "fusionedge_main", M_FULL)
    runs = mf.load_runs(exp, include_smoke=True)
    assert mf.ablation_table(runs, "fusionedge_main") is None


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def test_fmt_handles_nan_and_none():
    assert mf.fmt(None) == "n/a"
    assert mf.fmt(float("nan")) == "n/a"
    assert mf.fmt(1.23456, 3) == "1.235"


def test_benchmark_table_uses_the_full_resolution_row(exp: Path):
    write_run(exp, "fusionedge_main", M_FULL)
    runs = mf.load_runs(exp, include_smoke=True)
    table = mf.benchmark_table(runs)
    assert "33.10" in table, "must use the 480x640 median, not the 128x128 one"
    assert "13.60" not in table
    assert "RTX 4070" in table


def test_load_runs_skips_directories_without_metrics(exp: Path):
    (exp / "empty_run").mkdir(parents=True)
    write_run(exp, "fusionedge_main", M_FULL)
    names = {r["_name"] for r in mf.load_runs(exp, include_smoke=True)}
    assert names == {"fusionedge_main"}


def test_load_runs_skips_corrupt_metrics(exp: Path):
    write_run(exp, "fusionedge_main", M_FULL)
    bad = exp / "broken"
    bad.mkdir(parents=True)
    (bad / "metrics.json").write_text("{not json", encoding="utf-8")
    names = {r["_name"] for r in mf.load_runs(exp, include_smoke=True)}
    assert "broken" not in names