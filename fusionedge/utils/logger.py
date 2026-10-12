"""Logging to stdout + a per-run log file, with a CSV metric sink.

Every run owns `experiments/<run_name>/`. Inside it:
    config.yaml / config.json   stamped provenance (see utils.config)
    log.txt                      stdout+stderr transcript, tee'd live
    metrics.csv                  one row per evaluation, append-only
    metrics.json                 final consolidated metrics for the run
    checkpoints/                 last.pt, best.pt, epoch_<n>.pt
Append-only CSV plus per-epoch checkpoints is what makes a run resumable after
an interruption without losing or double-counting history.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any, Iterable


class RunLogger:
    """Small facade around logging + CSV so the training loop stays readable."""

    def __init__(self, run_dir: str | Path, name: str = "fusionedge", level: int = logging.INFO):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        self._start = time.time()

        self.log = logging.getLogger(f"fusionedge.{self.run_dir.name}")
        self.log.setLevel(level)
        self.log.propagate = False
        for h in list(self.log.handlers):
            self.log.removeHandler(h)

        fmt = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
        )
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        self.log.addHandler(sh)

        fh = logging.FileHandler(self.run_dir / "log.txt", mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        self.log.addHandler(fh)

        self._csv_path = self.run_dir / "metrics.csv"
        self._csv_fields: list[str] | None = None

    # ---------------- text ----------------
    def info(self, msg: str) -> None:
        self.log.info(msg)

    def warn(self, msg: str) -> None:
        self.log.warning(msg)

    def error(self, msg: str) -> None:
        self.log.error(msg)

    def section(self, title: str) -> None:
        self.log.info("-" * 72)
        self.log.info(title)
        self.log.info("-" * 72)

    def kv(self, d: dict[str, Any], title: str = "config") -> None:
        self.log.info(f"{title}:")
        for k, v in d.items():
            self.log.info(f"  {k:<34} {v}")

    # ---------------- csv ----------------
    def log_metrics(self, row: dict[str, Any]) -> None:
        """Append one evaluation row, growing the schema if new keys appear.

        When a later row introduces a column that earlier rows lack, the file is
        rewritten once with the widened header (earlier rows get an empty value)
        rather than silently dropping the column or silently writing a ragged
        file that pandas would mis-parse.
        """
        row = {"wall_time_s": round(time.time() - self._start, 2), **row}
        existing = self._csv_fields

        if existing is None and self._csv_path.exists() and self._csv_path.stat().st_size > 0:
            existing = self._csv_path.read_text(encoding="utf-8").splitlines()[0].split(",")

        if not existing:
            # first row of a fresh file: it defines the schema
            self._csv_fields = list(row.keys())
            with self._csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._csv_fields).writeheader()
            existing = self._csv_fields

        # A logger re-attached to an existing run (e.g. after a resume) reads the
        # header from disk; it must adopt it as its own schema, otherwise the
        # append below would have no fieldnames.
        self._csv_fields = existing

        new_keys = [k for k in row if k not in existing]
        if new_keys:
            old_fields = list(existing)
            fields = old_fields + new_keys
            self._csv_fields = fields
            prior = list(self._read_rows())
            with self._csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in prior:
                    w.writerow({k: r.get(k, "") for k in fields})

        with self._csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self._csv_fields).writerow(
                {k: row.get(k, "") for k in self._csv_fields}
            )

    def _read_rows(self) -> Iterable[dict[str, str]]:
        if not self._csv_path.exists():
            return []
        with self._csv_path.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def metrics_rows(self) -> Iterable[dict[str, str]]:
        if not self._csv_path.exists():
            return []
        with self._csv_path.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def close(self) -> None:
        for h in list(self.log.handlers):
            h.flush()
            h.close()
            self.log.removeHandler(h)