"""Print the resolved config and parameter count of every experiment config.

Run this before launching a sweep: it catches a config that references an unknown
model, an unknown key, or an architecture whose size no longer matches the
baseline (which would quietly turn the comparison into a capacity comparison).

    python scripts/show_configs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fusionedge.utils.config import load_config  # noqa: E402
from fusionedge.utils.registry import build_model  # noqa: E402


def main() -> int:
    configs = sorted(p for p in (REPO_ROOT / "configs").glob("*.yaml") if p.name != "default.yaml")
    rows = []
    for p in configs:
        cfg = load_config(p)
        model = build_model(cfg["model"]["name"], cfg["model"].get("cfg", {}))
        n = sum(q.numel() for q in model.parameters() if q.requires_grad)
        rows.append((p.name, cfg["run_name"], cfg["model"]["name"], n,
                     cfg["dataset"]["id"], cfg["train"]["epochs"], cfg["seed"]))

    width = max(len(r[0]) for r in rows)
    print(f"{'config'.ljust(width)}  {'run_name':<22} {'model':<11} "
          f"{'params':>9}  dataset   epochs  seed")
    print("-" * (width + 62))
    for name, run, mname, n, ds, ep, seed in rows:
        print(f"{name.ljust(width)}  {run:<22} {mname:<11} {n:>9,}  "
              f"{ds:<8}  {ep:>6}  {seed}")

    sizes = [r[3] for r in rows]
    if sizes:
        print(f"\nparameter range: {min(sizes):,} .. {max(sizes):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())