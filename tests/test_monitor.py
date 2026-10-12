"""Monitoring dashboard: it must report only what is on disk, and never lie.

A progress dashboard that invents or smooths over numbers is worse than none --
it is trusted precisely because it looks authoritative. These tests pin that the
page is a faithful projection of the artefacts, that ETA comes from measured
times rather than the config, and that smoke/failed runs cannot be mistaken for
results.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "fusionedge_monitor", REPO_ROOT / "scripts" / "monitor.py"
)
mon = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(mon)


def write_run(
    exp: Path,
    name: str,
    *,
    epochs_total: int = 60,
    losses: list[float] | None = None,
    epoch_times: list[float] | None = None,
    smoke: bool = False,
    with_metrics: bool = True,
    with_val: bool = True,
) -> Path:
    d = exp / name
    d.mkdir(parents=True, exist_ok=True)

    losses = losses or [10.0, 8.0, 6.0]
    times = epoch_times or [11.0] * len(losses)

    header = ["epoch", "phase", "loss", "intensity", "epoch_time_s"]
    if with_val:
        header += ["val_EN", "val_MS_SSIM_ir"]
    lines = [",".join(header)]
    for i, (l, t) in enumerate(zip(losses, times)):
        row = [str(i), "train", f"{l}", "0.3", f"{t}"]
        if with_val:
            # validation only on even epochs, as the trainer does
            row += [f"{6.0 - i * 0.1:.4f}", f"{0.85 - i * 0.01:.4f}"] if i % 2 == 0 else ["", ""]
        lines.append(",".join(row))
    (d / "metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (d / "config.json").write_text(json.dumps({
        "train": {"epochs": epochs_total, "best_metric": "MS_SSIM_ir"},
        "dataset": {"id": "MSRS", "train_split": "train", "eval_split": "val"},
        "model": {"name": "baseline", "cfg": {}},
        "_stamped": {
            "git_commit": "abc123def456",
            "dataset_manifest_sha256": "f" * 64,
            "environment": {"gpu": "Test GPU"},
        },
    }))
    (d / "log.txt").write_text("line one\nline two\nlast line\n")
    (d / "checkpoints").mkdir(exist_ok=True)
    (d / "checkpoints" / "best.pt").write_bytes(b"x")

    if with_metrics:
        (d / "metrics.json").write_text(json.dumps({
            "metrics": {"EN": 6.5, "AG": 0.004},
            "evaluation": {"num_images": 361},
            "dataset": {"dataset_id": "MSRS"},
            "smoke": smoke,
        }))
    return d


@pytest.fixture
def exp(tmp_path: Path) -> Path:
    e = tmp_path / "experiments"
    e.mkdir()
    return e


# --------------------------------------------------------------------------- #
# faithful reporting
# --------------------------------------------------------------------------- #
def test_reads_losses_from_csv(exp: Path):
    write_run(exp, "r", losses=[10.0, 8.0, 6.0])
    r = mon.collect_run(exp / "r")
    assert r["losses"] == [10.0, 8.0, 6.0]
    assert r["epochs_done"] == 3
    assert r["epochs_total"] == 60
    assert r["complete"] is False


def test_reports_completion_at_the_final_epoch(exp: Path):
    write_run(exp, "r", epochs_total=3, losses=[10.0, 8.0, 6.0])
    r = mon.collect_run(exp / "r")
    assert r["complete"] is True


def test_ignores_nan_and_inf_in_losses(exp: Path):
    """A NaN in the CSV must not poison the chart or the 'last loss' readout."""
    d = write_run(exp, "r", losses=[10.0])
    (d / "metrics.csv").write_text(
        "epoch,phase,loss,epoch_time_s\n0,train,10.0,11\n1,train,nan,11\n",
        encoding="utf-8",
    )
    r = mon.collect_run(d.parent / "r")
    assert all(v == v for v in r["losses"])
    assert r["losses"] == [10.0]


def test_handles_empty_and_missing_csv(exp: Path):
    d = exp / "empty"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"train": {"epochs": 60}}))
    r = mon.collect_run(d)
    assert r["losses"] == []
    assert r["epochs_done"] == 0


def test_handles_corrupt_config(exp: Path):
    """A half-written config must not 500 the whole page."""
    d = write_run(exp, "r")
    (d / "config.json").write_text("{truncated", encoding="utf-8")
    r = mon.collect_run(d)
    assert r["name"] == "r"  # survived, with defaults


def test_survives_missing_metrics_json(exp: Path):
    write_run(exp, "r", with_metrics=False)
    r = mon.collect_run(exp / "r")
    assert r["has_metrics_json"] is False
    assert r["final_metrics"] is None
    mon.render_run_page(r)  # must not raise


# --------------------------------------------------------------------------- #
# ETA
# --------------------------------------------------------------------------- #
def test_eta_uses_measured_epoch_times(exp: Path):
    """5 epochs left at 10 s/epoch -> ~50 s. If ETA came from the config or a
    constant it would not track the actual per-epoch cost."""
    write_run(exp, "r", epochs_total=10, losses=[1.0] * 5, epoch_times=[10.0] * 5)
    r = mon.collect_run(exp / "r")
    assert r["eta_s"] == pytest.approx(50.0, abs=1.0)


def test_eta_is_none_when_complete(exp: Path):
    write_run(exp, "r", epochs_total=2, losses=[1.0, 1.0])
    assert mon.collect_run(exp / "r")["eta_s"] is None


def test_eta_uses_recent_epochs_not_a_stale_average(exp: Path):
    """A fast start should not make ETA permanently optimistic."""
    write_run(
        exp, "r", epochs_total=20,
        losses=[1.0] * 10,
        epoch_times=[60.0] * 5 + [5.0] * 5,
    )
    r = mon.collect_run(exp / "r")
    assert r["eta_s"] == pytest.approx(50.0, abs=1.0), "should use the recent 5 epochs"


def test_eta_none_when_no_epoch_times(exp: Path):
    d = write_run(exp, "r")
    (d / "metrics.csv").write_text("epoch,phase,loss\n0,train,5.0\n", encoding="utf-8")
    assert mon.collect_run(d)["eta_s"] is None


# --------------------------------------------------------------------------- #
# honesty labelling
# --------------------------------------------------------------------------- #
def test_smoke_run_is_labelled_in_the_page(exp: Path):
    write_run(exp, "r", smoke=True)
    page = mon.render_run_page(mon.collect_run(exp / "r"))
    assert "SMOKE RUN" in page
    assert "NOT results" in page


def test_failed_run_is_labelled_and_excluded_by_default(exp: Path):
    write_run(exp, "_failed/broken")
    names = [d.name for d in mon.discover_runs(exp)]
    assert "broken" not in names
    names_inc = [d.name for d in mon.discover_runs(exp, include_failed=True)]
    assert "broken" in names_inc
    page = mon.render_run_page(mon.collect_run(exp / "_failed" / "broken"))
    assert "PRESERVED FAILED RUN" in page


def test_failed_run_uses_a_distinct_banner_colour(exp: Path):
    write_run(exp, "_failed/broken")
    page = mon.render_run_page(mon.collect_run(exp / "_failed" / "broken"))
    assert "banner fail" in page


def test_non_run_directories_are_excluded(exp: Path):
    write_run(exp, "real")
    (exp / "_failed").mkdir(exist_ok=True)
    (exp / "some_stray_dir").mkdir(exist_ok=True)
    names = [d.name for d in mon.discover_runs(exp)]
    assert names == ["real"]


def test_page_states_the_dataset_and_manifest(exp: Path):
    write_run(exp, "r")
    page = mon.render_run_page(mon.collect_run(exp / "r"))
    assert "MSRS" in page
    assert "ffffffffffffffff" in page, "manifest prefix must be shown"


def test_eta_is_marked_as_an_estimate(exp: Path):
    write_run(exp, "r", epochs_total=60, losses=[1.0] * 5)
    page = mon.render_run_page(mon.collect_run(exp / "r"))
    assert "estimate" in page


# --------------------------------------------------------------------------- #
# rendering robustness
# --------------------------------------------------------------------------- #
def test_page_renders_with_no_data_at_all(exp: Path):
    d = exp / "bare"
    d.mkdir()
    page = mon.render_run_page(mon.collect_run(d))
    assert "<html" in page and "</html>" in page


def test_page_escapes_html_in_run_names(exp: Path):
    d = exp / "r"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"train": {"epochs": 1}}))
    page = mon.render_run_page(mon.collect_run(d))
    assert "<script>" not in page


def test_index_page_lists_runs(exp: Path):
    write_run(exp, "aaa")
    write_run(exp, "bbb", smoke=True)
    page = mon.render_index([mon.collect_run(exp / "aaa"), mon.collect_run(exp / "bbb")])
    assert "/run/aaa" in page and "/run/bbb" in page
    assert "SMOKE" in page


def test_index_page_handles_no_runs(exp: Path):
    page = mon.render_index([])
    assert "No runs found" in page


def test_sparkline_handles_short_and_empty_series():
    assert "spark" in mon.sparkline([])
    assert "spark" in mon.sparkline([1.0])


def test_sparkline_flat_series_does_not_divide_by_zero():
    """rng == 0 when every value is identical; a naive (v-lo)/rng divides by 0."""
    out = mon.sparkline([2.0] * 5)
    assert "polyline" in out and "NaN" not in out


def test_fmt_secs():
    assert mon.fmt_secs(None) == "--"
    assert mon.fmt_secs(45) == "45s"
    assert mon.fmt_secs(90) == "1m 30s"
    assert mon.fmt_secs(3700) == "1h 01m"


def test_best_metric_tracking(exp: Path):
    """best.pt is selected on one declared metric; the page must show that same
    metric's best, not the last value."""
    write_run(exp, "r", losses=[5.0] * 4)
    r = mon.collect_run(exp / "r")
    assert r["best_key"] == "MS_SSIM_ir"
    assert r["best_val"] is not None
    assert r["best_epoch"] is not None


def test_benchmark_is_surfaced_when_present(exp: Path):
    d = write_run(exp, "r")
    (d / "benchmark.json").write_text(json.dumps({
        "params": 13667,
        "checkpoint_mb": 0.05,
        "gpu": "Test GPU",
        "latency": [{"size": [480, 640], "latency_ms_median": 14.2,
                     "latency_ms_p95": 15.0, "fps": 70.0, "peak_vram_mb": 198}],
    }))
    page = mon.render_run_page(mon.collect_run(d))
    assert "Efficiency" in page and "13,667" in page and "14.20" in page