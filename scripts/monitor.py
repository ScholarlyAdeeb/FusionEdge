#!/usr/bin/env python
"""Live training dashboard for a FusionEdge run.

Serves a local web page that reads a run directory and shows progress, ETA, the
loss curve and validation metrics.

    python scripts/monitor.py                          # newest run, port 8000
    python scripts/monitor.py --run fusionedge_main
    python scripts/monitor.py --runs --port 8080       # sweep overview

Design constraints:

  * **Read-only.** It never opens a checkpoint for writing and never imports the
    trainer. A monitoring tool that can mutate the thing it monitors is a hazard,
    especially one you leave running.
  * **No invented numbers.** Everything displayed comes from files on disk:
    metrics.csv, config.json, log.txt, metrics.json, benchmark.json. If a field
    is missing the UI says so rather than showing a plausible stand-in.
  * **Smoke runs are labelled.** A run with `"smoke": true` gets a warning banner,
    so a 4-image pipeline check is never mistaken for a result.
  * **Stdlib only.** http.server, no web framework, no charting library -- the
    page has to keep working when nothing new is installed.

The ETA is computed from measured epoch times, not from the config, so it
accounts for a slow first epoch. It is explicitly marked as an estimate.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

EXP_DIR = REPO_ROOT / "experiments"
# Runs kept deliberately for the record; they are evidence, not active runs.
FAILED_PREFIX = "_failed"

REFRESH_SECONDS = 3


# --------------------------------------------------------------------------- #
# data collection
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 -- a half-written file must not kill the page
        return None


def _read_history(run_dir: Path) -> list[dict]:
    csv_path = run_dir / "metrics.csv"
    if not csv_path.exists():
        return []
    try:
        with csv_path.open(encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:  # noqa: BLE001
        return []


def _floats(values: list[str]) -> list[float]:
    out = []
    for v in values:
        try:
            f = float(v)
            if f == f and abs(f) != float("inf"):
                out.append(f)
        except (TypeError, ValueError):
            pass
    return out


def collect_run(run_dir: Path) -> dict:
    """Everything the page needs about one run."""
    cfg = _read_json(run_dir / "config.json") or {}
    metrics = _read_json(run_dir / "metrics.json")
    bench = _read_json(run_dir / "benchmark.json")
    rows = _read_history(run_dir)

    train_rows = [r for r in rows if r.get("phase", "train") == "train"]
    epochs_total = int(
        (cfg.get("train") or {}).get("epochs", 0) or len(train_rows)
    )
    epochs_done = len(train_rows)

    losses = _floats([r.get("loss", "") for r in train_rows])
    epoch_times = _floats([r.get("epoch_time_s", "") for r in train_rows])

    # per-term training loss, for the stacked view
    terms = ["intensity", "gradient", "ssim", "edge", "color"]
    term_series = {t: _floats([r.get(t, "") for r in train_rows]) for t in terms}

    # validation metrics that were actually logged
    val_keys = sorted({k[4:] for r in rows for k in r if k.startswith("val_")})
    val_series = {k: _floats([r.get(f"val_{k}", "") for r in rows]) for k in val_keys}

    best_key = (cfg.get("train") or {}).get("best_metric")
    best_val = None
    best_epoch = None
    if best_key and val_series.get(best_key):
        vals = val_series[best_key]
        best_val = max(vals)
        best_epoch = vals.index(best_val)

    # ETA from measured throughput over the recent epochs.
    eta_s = None
    if epochs_total and epochs_done < epochs_total and epoch_times:
        recent = epoch_times[-5:]
        if recent:
            per_epoch = sum(recent) / len(recent)
            eta_s = per_epoch * (epochs_total - epochs_done)

    stamps = cfg.get("_stamped", {})
    is_smoke = bool(metrics.get("smoke")) if metrics else False
    # A run under experiments/_failed/ is preserved evidence of a failure, not
    # an active run. Detected from the path so it works even if the directory is
    # renamed, and so the FAILED banner cannot be lost.
    is_failed = FAILED_PREFIX in Path(run_dir).parts or run_dir.name.startswith(
        FAILED_PREFIX
    )

    ckpt_dir = run_dir / "checkpoints"
    ckpts = (
        sorted(p.name for p in ckpt_dir.glob("*.pt")) if ckpt_dir.is_dir() else []
    )

    last_log = ""
    log_path = run_dir / "log.txt"
    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            last_log = lines[-1] if lines else ""
        except Exception:  # noqa: BLE001
            pass

    return {
        "name": run_dir.name,
        "path": str(run_dir),
        "epochs_done": epochs_done,
        "epochs_total": epochs_total,
        "complete": epochs_total > 0 and epochs_done >= epochs_total,
        "losses": losses,
        "term_series": term_series,
        "epoch_times": epoch_times,
        "val_series": val_series,
        "best_key": best_key,
        "best_val": best_val,
        "best_epoch": (best_epoch + 1) if best_epoch is not None else None,
        "eta_s": eta_s,
        "is_smoke": is_smoke,
        "is_failed": is_failed,
        "final_metrics": (metrics or {}).get("metrics"),
        "dataset_id": ((metrics or {}).get("dataset") or {}).get("dataset_id")
        or (cfg.get("dataset") or {}).get("id"),
        "num_eval_images": ((metrics or {}).get("evaluation") or {}).get("num_images"),
        "params": (bench or {}).get("params"),
        "benchmark": bench,
        "checkpoints": ckpts,
        "last_log_line": last_log,
        "manifest": (stamps.get("dataset_manifest_sha256") or "")[:16],
        "git_commit": (stamps.get("git_commit") or "")[:12],
        "gpu": ((stamps.get("environment") or {}).get("gpu")),
        "model_name": (cfg.get("model") or {}).get("name"),
        "has_metrics_json": metrics is not None,
    }


def discover_runs(exp_dir: Path, include_failed: bool = False) -> list[Path]:
    """Find run directories, including those preserved under `_failed/`.

    `_failed` is a real directory containing genuine evidence, so its children
    are discovered too (and labelled FAILED in the UI) rather than skipped
    entirely -- hiding them would defeat the purpose of preserving them. Only the
    `_failed` container itself and other `_`-prefixed bookkeeping directories are
    ignored.
    """
    if not exp_dir.is_dir():
        return []

    def has_run_files(d: Path) -> bool:
        return (d / "metrics.csv").exists() or (d / "config.json").exists()

    runs: list[Path] = []
    for d in sorted(exp_dir.iterdir()):
        if not d.is_dir():
            continue
        if d.name.startswith("_"):
            if not d.name.startswith(FAILED_PREFIX):
                continue
            # _failed/ holds run directories one level down
            if include_failed:
                runs.extend(sorted(c for c in d.iterdir() if c.is_dir() and has_run_files(c)))
            continue
        if has_run_files(d):
            runs.append(d)
    return runs


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
def fmt_secs(s: float | None) -> str:
    if s is None:
        return "--"
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


def sparkline(values: list[float], width: int = 560, height: int = 90) -> str:
    """Inline SVG polyline. No charting library, no client-side JS."""
    vals = [v for v in values if v == v]
    if len(vals) < 2:
        return (
            f'<svg viewBox="0 0 {width} {height}" class="spark">'
            f'<line x1="0" y1="{height / 2}" x2="{width}" y2="{height / 2}" '
            f'stroke="#333" stroke-width="1"/></svg>'
            "<p class='hint'>need &gt;= 2 points</p>"
        )
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    pad = 6
    step = (width - 2 * pad) / (len(vals) - 1)

    pts = []
    for i, v in enumerate(vals):
        x = pad + i * step
        y = height - pad - ((v - lo) / rng) * (height - 2 * pad)
        pts.append(f"{x:.1f},{y:.1f}")

    polyline = " ".join(pts)
    area = f"{pad},{height - pad} {polyline} {width - pad},{height - pad}"
    return (
        f'<svg viewBox="0 0 {width} {height}" class="spark" '
        f'preserveAspectRatio="none">'
        f'<polygon points="{area}" fill="rgba(96,165,250,.10)"/>'
        f'<polyline points="{polyline}" fill="none" stroke="#60a5fa" '
        f'stroke-width="1.8" stroke-linejoin="round"/>'
        f"</svg>"
        f'<div class="axis"><span>{lo:.4f}</span>'
        f'<span>min {lo:.4f} / max {hi:.4f}</span>'
        f'<span>{hi:.4f}</span></div>'
    )


def bar(pct: float, label: str, cls: str = "bar") -> str:
    pct = max(0.0, min(100.0, pct))
    return (
        f'<div class="{cls}"><div class="fill" style="width:{pct:.2f}%"></div>'
        f'<div class="barlabel">{html.escape(label)}</div></div>'
    )


def render_run_page(run: dict) -> str:
    done, total = run["epochs_done"], run["epochs_total"]
    pct = (100.0 * done / total) if total else 0.0

    banners = []
    if run["is_failed"]:
        banners.append(
            '<div class="banner fail">PRESERVED FAILED RUN &mdash; these numbers '
            "are not results. See NOTES.md in this directory.</div>"
        )
    elif run["is_smoke"]:
        banners.append(
            '<div class="banner warn">SMOKE RUN &mdash; pipeline check only. '
            "These metrics are NOT results.</div>"
        )
    if run["complete"]:
        banners.append('<div class="banner ok">Run complete.</div>')

    # Every numeric readout must tolerate an empty history: a run that has just
    # been created has no rows yet, and the page has to render rather than 500.
    last_loss = f"{run['losses'][-1]:.4f}" if run["losses"] else "--"
    best_loss = f"{min(run['losses']):.4f}" if run["losses"] else "--"
    recent_times = run["epoch_times"][-5:]
    epoch_time = (
        fmt_secs(sum(recent_times) / len(recent_times)) if recent_times else "--"
    )

    if run["complete"]:
        status = "complete"
        eta_txt = "done"
    elif total:
        status = f"epoch {done}/{total}"
        eta_txt = f"~{fmt_secs(run['eta_s'])} left (estimate)"
    else:
        status = "starting"
        eta_txt = "--"

    # validation table
    val_rows = ""
    for key in sorted(run["val_series"]):
        vals = run["val_series"][key]
        if not vals:
            continue
        marker = ""
        if key == run["best_key"]:
            best_i = run["best_epoch"]
            if best_i and best_i <= len(vals):
                marker = (
                    f' <span class="tag">best {vals[best_i - 1]:.4f} '
                    f"@ epoch {best_i}</span>"
                )
        val_rows += (
            f"<tr><td>{html.escape(key)}</td>"
            f'<td class="num">{vals[-1]:.4f}</td>'
            f'<td class="num">{min(vals):.4f}</td>'
            f'<td class="num">{max(vals):.4f}</td>'
            f'<td class="sparkcell">{sparkline(vals, 260, 44)}{marker}</td></tr>'
        )
    val_table = (
        "<table><thead><tr><th>validation metric</th><th>latest</th>"
        "<th>min</th><th>max</th><th>trend</th></tr></thead>"
        f"<tbody>{val_rows}</tbody></table>"
        if val_rows
        else '<p class="hint">no validation metrics logged yet</p>'
    )

    terms = ""
    for name, vals in run["term_series"].items():
        if vals:
            terms += (
                f'<div class="term"><span class="termname">{html.escape(name)}</span>'
                f'<span class="termval">{vals[-1]:.4f}</span></div>'
            )
    terms_html = f'<div class="terms">{terms}</div>' if terms else ""

    bench_html = ""
    if run["benchmark"]:
        b = run["benchmark"]
        rows = "".join(
            f"<tr><td>{lat['size'][0]}&times;{lat['size'][1]}</td>"
            f'<td class="num">{lat["latency_ms_median"]:.2f}</td>'
            f'<td class="num">{lat["latency_ms_p95"]:.2f}</td>'
            f'<td class="num">{lat.get("fps", 0):.1f}</td>'
            f'<td class="num">{lat.get("peak_vram_mb", "--")}</td></tr>'
            for lat in b.get("latency", [])
        )
        bench_html = (
            "<h2>Efficiency</h2><table><thead><tr><th>input</th>"
            "<th>median ms</th><th>p95 ms</th><th>fps</th>"
            f"<th>peak MiB</th></tr></thead><tbody>{rows}</tbody></table>"
            f'<p class="hint">{b.get("params", 0):,} parameters &middot; '
            f'{b.get("checkpoint_mb", 0):.2f} MB &middot; '
            f'{html.escape(str(b.get("gpu", "cpu")))}</p>'
        )

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="{REFRESH_SECONDS}">
<title>FusionEdge &middot; {html.escape(run['name'])}</title>
<style>
  :root {{
    --bg:#0f1117; --panel:#171a23; --line:#262a36; --fg:#e6e8ef;
    --dim:#8b93a7; --accent:#60a5fa; --ok:#34d399; --warn:#fbbf24; --bad:#f87171;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font:14px/1.55 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:28px 22px 64px; }}
  h1 {{ font-size:20px; margin:0 0 2px; font-weight:650; letter-spacing:-.01em; }}
  h2 {{ font-size:14px; margin:30px 0 10px; color:var(--dim);
        text-transform:uppercase; letter-spacing:.09em; font-weight:600; }}
  .sub {{ color:var(--dim); font-size:13px; margin-bottom:20px; }}
  .sub code {{ background:var(--panel); padding:1px 6px; border-radius:4px; font-size:12px; }}
  .panel {{ background:var(--panel); border:1px solid var(--line);
            border-radius:10px; padding:18px; margin-bottom:16px; }}
  .banner {{ padding:9px 14px; border-radius:8px; margin-bottom:14px;
             font-size:13px; font-weight:560; }}
  .banner.warn {{ background:rgba(251,191,36,.12); color:var(--warn); border:1px solid rgba(251,191,36,.3); }}
  .banner.fail {{ background:rgba(248,113,113,.12); color:var(--bad); border:1px solid rgba(248,113,113,.3); }}
  .banner.ok  {{ background:rgba(52,211,153,.12); color:var(--ok);  border:1px solid rgba(52,211,153,.3); }}
  .bar {{ position:relative; height:30px; background:#0c0e14;
          border:1px solid var(--line); border-radius:7px; overflow:hidden; }}
  .bar .fill {{ position:absolute; inset:0 auto 0 0;
                background:linear-gradient(90deg,#3b82f6,#60a5fa); transition:width .4s ease; }}
  .barlabel {{ position:absolute; inset:0; display:flex; align-items:center;
               padding:0 12px; font-size:12.5px; font-weight:560;
               text-shadow:0 1px 3px rgba(0,0,0,.85); }}
  .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
            gap:12px; margin-top:16px; }}
  .stat {{ background:#0c0e14; border:1px solid var(--line); border-radius:8px; padding:11px 13px; }}
  .stat .k {{ color:var(--dim); font-size:11px; text-transform:uppercase;
              letter-spacing:.07em; }}
  .stat .v {{ font-size:19px; font-weight:640; margin-top:3px;
              font-variant-numeric:tabular-nums; }}
  .spark {{ width:100%; height:90px; display:block; }}
  .sparkcell .spark {{ height:44px; }}
  .axis {{ display:flex; justify-content:space-between; color:var(--dim);
           font-size:11px; margin-top:-14px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th {{ text-align:left; color:var(--dim); font-weight:560; font-size:11px;
        text-transform:uppercase; letter-spacing:.06em;
        border-bottom:1px solid var(--line); padding:6px 8px; }}
  td {{ padding:6px 8px; border-bottom:1px solid rgba(38,42,54,.5); vertical-align:middle; }}
  td.num {{ font-variant-numeric:tabular-nums; text-align:right; width:96px; }}
  .tag {{ background:rgba(52,211,153,.14); color:var(--ok); font-size:10.5px;
          padding:1px 7px; border-radius:20px; margin-left:6px;
          white-space:nowrap; }}
  .terms {{ display:flex; flex-wrap:wrap; gap:9px; margin-top:12px; }}
  .term {{ background:#0c0e14; border:1px solid var(--line); border-radius:7px;
           padding:7px 11px; font-size:12px; }}
  .termname {{ color:var(--dim); margin-right:8px; }}
  .termval {{ font-variant-numeric:tabular-nums; font-weight:600; }}
  .hint {{ color:var(--dim); font-size:12px; margin:8px 0 0; }}
  .log {{ font-family:ui-monospace,"Cascadia Code",Consolas,monospace;
          font-size:12px; color:var(--dim); background:#0c0e14;
          border:1px solid var(--line); border-radius:8px; padding:10px 12px;
          overflow-x:auto; white-space:pre; }}
  a {{ color:var(--accent); }}
</style></head>
<body><div class="wrap">
  {''.join(banners)}
  <h1>{html.escape(run['name'])}</h1>
  <div class="sub">
    model <code>{html.escape(str(run['model_name']))}</code> &middot;
    dataset <code>{html.escape(str(run['dataset_id']))}</code> &middot;
    manifest <code>{html.escape(run['manifest'])}</code> &middot;
    commit <code>{html.escape(run['git_commit'])}</code>
    {(' &middot; GPU ' + html.escape(str(run['gpu']))) if run['gpu'] else ''}
  </div>

  <div class="panel">
    {bar(pct, f"{status} &mdash; {pct:.1f}% &nbsp;&middot;&nbsp; {eta_txt}")}
    <div class="stats">
      <div class="stat"><div class="k">epochs</div>
        <div class="v">{done}{' / ' + str(total) if total else ''}</div></div>
      <div class="stat"><div class="k">time left</div>
        <div class="v">{eta_txt.split()[0] if eta_txt != '--' else '--'}</div></div>
      <div class="stat"><div class="k">last loss</div>
        <div class="v">{last_loss}</div></div>
      <div class="stat"><div class="k">best loss</div>
        <div class="v">{best_loss}</div></div>
      {'' if run['losses'] else "<p class='hint'>no training steps logged yet</p>"}
      <div class="stat"><div class="k">epoch time</div>
        <div class="v">{epoch_time}</div></div>
      <div class="stat"><div class="k">checkpoints</div>
        <div class="v">{len(run['checkpoints'])}</div></div>
    </div>
  </div>

  <h2>Training loss</h2>
  <div class="panel">{sparkline(run['losses'])}{terms_html}</div>

  <h2>Validation ({html.escape(str(run['best_key']))} selects best.pt)</h2>
  <div class="panel">{val_table}</div>

  {bench_html}

  <h2>Latest log line</h2>
  <div class="log">{html.escape(run['last_log_line'] or '(none)')}</div>

  <p class="hint">
    Read-only view of <code>{html.escape(run['path'])}</code>.
    Every value is read from files on disk; refreshing every {REFRESH_SECONDS}s.
    ETA is an estimate from measured epoch times.
  </p>
</div></body></html>"""


def render_index(runs: list[dict]) -> str:
    if not runs:
        body = '<p class="hint">No runs found yet. Start one with:<br><code>python scripts/run_experiment.py --config configs/baseline.yaml</code></p>'
    else:
        rows = ""
        for r in runs:
            total = r["epochs_total"]
            done = r["epochs_done"]
            pct = (100.0 * done / total) if total else 0.0
            cls = "tag"
            if r["is_failed"]:
                tag = '<span class="tag" style="background:rgba(248,113,113,.16);color:#f87171">FAILED (preserved)</span>'
            elif r["is_smoke"]:
                tag = '<span class="tag" style="background:rgba(251,191,36,.16);color:#fbbf24">SMOKE</span>'
            elif r["complete"]:
                tag = '<span class="tag">complete</span>'
            else:
                tag = f'<span class="tag">{done}/{total or "?"}</span>'
            loss_cell = f"{r['losses'][-1]:.4f}" if r["losses"] else "--"
            rows += (
                f'<tr><td><a href="/run/{r["name"]}">{html.escape(r["name"])}</a> '
                f"{tag}</td>"
                f'<td>{html.escape(str(r["model_name"]))}</td>'
                f'<td>{html.escape(str(r["dataset_id"]))}</td>'
                f'<td style="width:220px">{bar(pct, f"{pct:.0f}%", "bar small")}</td>'
                f'<td class="num">{loss_cell}</td>'
                f'<td class="num">~{fmt_secs(r["eta_s"])}</td></tr>'
            )
        body = (
            "<table><thead><tr><th>run</th><th>model</th><th>dataset</th>"
            "<th>progress</th><th>loss</th><th>eta</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="{REFRESH_SECONDS}">
<title>FusionEdge runs</title><style>
:root {{ --bg:#0f1117; --panel:#171a23; --line:#262a36; --fg:#e6e8ef; --dim:#8b93a7; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg);
        font:14px/1.55 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif; }}
.wrap {{ max-width:1000px; margin:0 auto; padding:28px 22px; }}
h1 {{ font-size:20px; margin:0 0 18px; font-weight:650; }}
table {{ width:100%; border-collapse:collapse; }}
th {{ text-align:left; color:var(--dim); font-size:11px; text-transform:uppercase;
       letter-spacing:.06em; border-bottom:1px solid var(--line); padding:7px 8px; }}
td {{ padding:8px; border-bottom:1px solid rgba(38,42,54,.5); }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; width:92px; }}
a {{ color:#60a5fa; text-decoration:none; }}
.bar {{ position:relative; height:22px; background:#0c0e14;
        border:1px solid var(--line); border-radius:6px; overflow:hidden; }}
.bar .fill {{ position:absolute; inset:0 auto 0 0;
              background:linear-gradient(90deg,#3b82f6,#60a5fa); }}
.barlabel {{ position:absolute; inset:0; display:flex; align-items:center;
             padding:0 9px; font-size:11.5px; }}
.hint {{ color:var(--dim); font-size:12.5px; }}
code {{ background:var(--panel); padding:1px 6px; border-radius:4px; }}
.tag {{ background:rgba(52,211,153,.14); color:#34d399; font-size:10.5px;
        padding:1px 7px; border-radius:20px; margin-left:7px; white-space:nowrap; }}
</style></head><body><div class="wrap"><h1>FusionEdge runs</h1>{body}
<p class="hint">Auto-refreshes every {REFRESH_SECONDS}s &middot; read-only.</p>
</div></body></html>"""


# --------------------------------------------------------------------------- #
# server
# --------------------------------------------------------------------------- #
def make_handler(run_dirs: list[Path] | None, exp_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args) -> None:  # silence per-request logging
            pass

        def _send(self, body: str, code: int = 200) -> None:
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def handle_one_request(self) -> None:
            # A browser that navigates away mid-refresh leaves the socket closed.
            # Without this, socketserver logs a BrokenPipeError traceback and the
            # monitor looks broken when it is working fine. Every day this page
            # refreshes itself, so disconnects are routine, not exceptional.
            try:
                super().handle_one_request()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                self.close_connection = True

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                if path in ("/", "/index.html"):
                    runs = [collect_run(d) for d in discover_runs(exp_dir)]
                    self._send(render_index(runs))
                    return

                if path == "/api/runs":
                    runs = [collect_run(d) for d in discover_runs(exp_dir)]
                    payload = json.dumps({"runs": runs}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                if path.startswith("/run/"):
                    name = path[len("/run/"):]
                    target = (exp_dir / name).resolve()
                    # containment check: never serve outside experiments/
                    if not str(target).startswith(str(exp_dir.resolve())) or not target.is_dir():
                        self._send("<h1>no such run</h1>", 404)
                        return
                    self._send(render_run_page(collect_run(target)))
                    return

                self._send("<h1>not found</h1>", 404)
            except Exception as exc:  # noqa: BLE001
                # the page must survive a half-written file or a vanished run
                self._send(f"<h1>monitor error</h1><pre>{html.escape(exc)}</pre>", 500)

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=None, help="run directory name to show")
    ap.add_argument("--runs", action="store_true", help="show the sweep overview")
    ap.add_argument("--exp", default="experiments")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--once", action="store_true",
                    help="print the summary once and exit (no server)")
    args = ap.parse_args()

    exp_dir = (REPO_ROOT / args.exp).resolve()

    if args.run and not args.runs:
        target = exp_dir / args.run
        if not target.is_dir():
            print(f"no such run: {target}", file=sys.stderr)
            return 2
        print(json.dumps(collect_run(target), indent=2)[:2000])
        if args.once:
            return 0
    elif args.once:
        for d in discover_runs(exp_dir):
            r = collect_run(d)
            print(f"{r['name']:<40} {r['epochs_done']}/{r['epochs_total']} "
                  f"loss={r['losses'][-1] if r['losses'] else float('nan'):.4f}")
        return 0

    host, port = args.host, args.port
    for attempt in range(20):
        try:
            httpd = ThreadingHTTPServer((host, port), make_handler(None, exp_dir))
            break
        except OSError:
            port += 1
    else:
        print("could not bind a port", file=sys.stderr)
        return 2

    url = f"http://{host}:{port}/"
    print(f"monitor serving {exp_dir}")
    print(f"  {url}")
    if args.run:
        print(f"  {url}run/{args.run}")
    print("Ctrl-C to stop.")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())