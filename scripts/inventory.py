#!/usr/bin/env python
"""Experiment inventory, generated from the run directories (nothing typed by hand).

    python scripts/inventory.py            # writes docs/EXPERIMENTS.md + results/inventory/<stamp>.json

For every experiments/<run> (and experiments/_failed/<run>) it records the status,
epochs completed, training time, best epoch, parameters, latency, peak VRAM, and
audit/gate outcome, read from config.yaml, metrics.csv, best.pt, benchmark.json,
results/checks/<run>.json and results/checks/<run>_audit.txt. Configs whose run
directory does not exist yet are listed as pending.

Status:
  accepted    gate json passed AND audit passed
  superseded  a SUPERSEDED marker, or NOTES.md headed "SUPERSEDED"
  failed      under experiments/_failed/
  incomplete  no metrics.json (still training, or stopped)
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import yaml  # noqa: E402

EXP = REPO_ROOT / "experiments"
CHECKS = REPO_ROOT / "results" / "checks"
# experiments/baseline_simplecnn_v2_sigmoid was moved to _failed/baseline_fp16_amp
# (same run: identical metrics.csv and config apart from line endings)
MOVED = {"baseline_fp16_amp": "previously experiments/baseline_simplecnn_v2_sigmoid (moved, not deleted)"}


def first_line(p: Path) -> str:
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("**Status") or line.strip().startswith("superseded"):
                return line.strip().strip("*").replace("**", "")
        return p.read_text(encoding="utf-8").splitlines()[0].lstrip("# ")
    except (OSError, IndexError):
        return ""


def describe(run: Path, failed: bool) -> dict:
    d = {"id": run.name, "dir": run.relative_to(REPO_ROOT).as_posix()}
    cfg_p = run / "config.yaml"
    cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8")) if cfg_p.exists() else {}
    m = cfg.get("model", {})
    d["model"] = m.get("name")
    d["model_cfg"] = {k: v for k, v in (m.get("cfg") or {}).items()
                      if k in ("output_space", "residual_mode", "width", "depth")}
    d["seed"] = cfg.get("seed")
    d["loader_seed"] = (cfg.get("data") or {}).get("loader_seed", 0 if cfg else None)
    d["epochs_cfg"] = (cfg.get("train") or {}).get("epochs")
    rows = []
    if (run / "metrics.csv").exists():
        rows = [r for r in csv.DictReader((run / "metrics.csv").open(encoding="utf-8")) if r.get("phase") == "train"]
    d["epochs_done"] = len(rows)
    d["train_minutes"] = round(float(rows[-1]["wall_time_s"]) / 60, 1) if rows and rows[-1].get("wall_time_s") else None
    best = run / "checkpoints" / "best.pt"
    if best.exists():
        try:
            p = torch.load(best, map_location="cpu", weights_only=False)
            d["best_epoch"] = p.get("best_epoch")
            d["best_metric"] = p.get("best_metric")
        except Exception as exc:  # noqa: BLE001
            d["best_epoch"] = f"unreadable: {type(exc).__name__}"
    d["last_pt"] = (run / "checkpoints" / "last.pt").relative_to(REPO_ROOT).as_posix() \
        if (run / "checkpoints" / "last.pt").exists() else None
    if (run / "benchmark.json").exists():
        b = json.loads((run / "benchmark.json").read_text(encoding="utf-8"))
        d["params"] = b.get("params")
        for lat in b.get("latency", []):
            if list(lat.get("size", [])) == [480, 640]:
                d["latency_ms_480x640"] = lat.get("latency_ms_median")
                d["peak_vram_mb_480x640"] = lat.get("peak_vram_mb")
    d["test_evaluated"] = (run / "metrics.json").exists()
    g = CHECKS / f"{run.name}.json"
    a = CHECKS / f"{run.name}_audit.txt"
    d["gate"] = (json.loads(g.read_text(encoding="utf-8")).get("passed") if g.exists() else None)
    if a.exists():
        txt = a.read_bytes().decode("utf-16" if a.read_bytes()[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8", "replace")
        d["audit"] = "AUDIT PASSED" in txt
    else:
        d["audit"] = None
    if failed:
        d["status"] = "failed"
    elif (run / "SUPERSEDED").exists() or first_line(run / "NOTES.md").upper().startswith("STATUS: STOPPED") \
            or "SUPERSEDED" in ((run / "NOTES.md").read_text(encoding="utf-8")[:200].upper() if (run / "NOTES.md").exists() else ""):
        d["status"] = "superseded"
    elif not d["test_evaluated"]:
        d["status"] = "incomplete"
    elif d["gate"] and d["audit"]:
        d["status"] = "accepted"
    else:
        d["status"] = "evaluated, not accepted"
    note = (run / "SUPERSEDED").read_text(encoding="utf-8").strip() if (run / "SUPERSEDED").exists() else first_line(run / "NOTES.md")
    d["note"] = MOVED.get(run.name, "") + ("; " if run.name in MOVED and note else "") + note
    return d


def main() -> int:
    runs = [describe(p, False) for p in sorted(EXP.iterdir()) if p.is_dir() and not p.name.startswith("_")]
    runs += [describe(p, True) for p in sorted((EXP / "_failed").iterdir()) if p.is_dir()]
    have = {r["id"] for r in runs}
    pending = []
    for c in sorted((REPO_ROOT / "configs").glob("*.yaml")):
        name = yaml.safe_load(c.read_text(encoding="utf-8")).get("run_name")
        if name and name not in have and c.name != "default.yaml":
            pending.append({"id": name, "config": c.relative_to(REPO_ROOT).as_posix()})
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = REPO_ROOT / "results" / "inventory"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"inventory_{stamp}.json").write_text(json.dumps({"generated": stamp, "runs": runs, "pending": pending},
                                                            indent=2, default=str), encoding="utf-8")

    fmt = lambda v, f="{}": "–" if v is None else f.format(v)  # noqa: E731
    L = ["# Experiment inventory", "",
         f"Generated by `python scripts/inventory.py` at {stamp} from the run directories, "
         "`results/checks/` and each run's `benchmark.json`. Do not edit by hand; re-run the script.", "",
         "Status: **accepted** = audit and acceptance gate both passed; **superseded** / **failed** = kept with "
         "their notes, excluded from every comparison; **incomplete** = no evaluation yet (training or stopped).", "",
         "| run id | status | model / head | seed (loader) | epochs | train min | best.pt epoch | params | latency 480×640 (ms) | peak VRAM (MB) | audit | gate |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        head = r["model_cfg"].get("residual_mode") or r["model_cfg"].get("output_space") or ""
        L.append(f"| `{r['id']}` | {r['status']} | {r['model'] or '–'} / {head or '–'} | {fmt(r['seed'])} ({fmt(r['loader_seed'])}) | "
                 f"{r['epochs_done']}/{fmt(r['epochs_cfg'])} | {fmt(r['train_minutes'])} | {fmt(r.get('best_epoch'))} | "
                 f"{fmt(r.get('params'), '{:,}')} | {fmt(r.get('latency_ms_480x640'), '{:.1f}')} | "
                 f"{fmt(r.get('peak_vram_mb_480x640'), '{:.0f}')} | {fmt(r['audit'])} | {fmt(r['gate'])} |")
    L += ["", "## Notes per run", ""]
    for r in runs:
        L.append(f"- `{r['dir']}`: {r['note'] or '–'}" + (f" Reported checkpoint: `{r['last_pt']}`." if r['status'] == 'accepted' else ""))
    L += ["", "## Pending (config exists, run directory does not)", ""]
    L += [f"- `{p['id']}` — `{p['config']}`" for p in pending] or ["- none"]
    L += ["", "`best.pt epoch` is 0-based as stored in the checkpoint; it is informational only — every reported "
          "number uses `last.pt` (docs/PROTOCOL.md)."]
    (REPO_ROOT / "docs" / "EXPERIMENTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
