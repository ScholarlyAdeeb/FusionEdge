"""YAML configuration with dotted access, inheritance and provenance stamping.

A run's config is the single source of truth for what was trained: it is copied
verbatim into `experiments/<run>/config.yaml` and additionally stamped with the
dataset id + manifest hash, the git commit, the resolved torch/cuda versions and
the full argv. That stamped copy -- not the source YAML -- is what
`scripts/make_figures.py` reads when it builds result tables, which is what makes
"every number traces to a log" checkable rather than aspirational.
"""

from __future__ import annotations

import copy
import datetime as _dt
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    """dict subclass with attribute + dotted-key access."""

    def __getattr__(self, item: str) -> Any:
        try:
            v = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        return Config(v) if isinstance(v, dict) and not isinstance(v, Config) else v

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """Look up `a.b.c`, returning `default` if any level is missing."""
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self, default=str))


def _deep_update(base: dict, other: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in other.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(
    path: str | Path,
    overrides: dict | None = None,
) -> Config:
    """Load a YAML config, following a `_base_` key for inheritance.

    `overrides` are applied last, so CLI flags always win over the file.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    base_ref = raw.pop("_base_", None)
    if base_ref:
        base_path = (path.parent / base_ref).resolve()
        raw = _deep_update(load_config(base_path), raw)

    cfg = Config(raw)
    if overrides:
        cfg = Config(_deep_update(cfg, overrides))
    cfg["_config_path"] = str(path.relative_to(REPO_ROOT)) if _is_relative(path) else str(path)
    return cfg


def _is_relative(p: Path) -> bool:
    try:
        p.relative_to(REPO_ROOT)
        return True
    except ValueError:
        return False


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def environment_fingerprint() -> dict:
    """Everything about the machine that could change a number."""
    import torch

    fp: dict[str, Any] = {
        "python": _dt.datetime.now().strftime("%Y%m%d"),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        fp["cuda"] = torch.version.cuda
        fp["gpu"] = torch.cuda.get_device_name(0)
        fp["capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    return fp


def stamp(cfg: Config, dataset_registry: dict | None = None, argv: list | None = None) -> Config:
    """Return a copy of `cfg` carrying provenance metadata.

    `dataset_registry` is the parsed data/registry/<dataset_id>.json. Its
    `dataset_id` and `manifest_sha256` are copied in verbatim so a metrics row
    can never be attributed to the wrong dataset bytes.
    """
    out = Config(copy.deepcopy(cfg))
    out["_stamped"] = {
        "at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
        "environment": environment_fingerprint(),
        "argv": argv or [],
    }
    if dataset_registry is not None:
        out["_stamped"]["dataset_id"] = dataset_registry.get("dataset_id")
        out["_stamped"]["dataset_status"] = dataset_registry.get("status")
        out["_stamped"]["dataset_manifest_sha256"] = dataset_registry.get(
            "manifest_sha256"
        )
        out["_stamped"]["dataset_splits"] = dataset_registry.get("splits")
        out["_stamped"]["dataset_retrieved_at"] = dataset_registry.get("retrieved_at")
    return out


def dump_config(cfg: Config, path: str | Path) -> None:
    """Write the config next to a run's logs, as YAML plus a JSON sidecar.

    The JSON sidecar exists so `scripts/make_figures.py` can read configs without
    a YAML dependency and without re-implementing config inheritance.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(json.loads(json.dumps(cfg, default=str)), f, sort_keys=False)
    path.with_suffix(".json").write_text(json.dumps(cfg, indent=2, default=str))