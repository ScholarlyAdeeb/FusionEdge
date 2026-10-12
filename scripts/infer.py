#!/usr/bin/env python
"""Single-pair inference CLI.

    python scripts/infer.py --ir ir.png --vi vi.png \
        --weights experiments/fusionedge_main/checkpoints/best.pt \
        --out results/fused.png

Refuses to run on weights that look untrained: an untrained FusionEdge returns the
visible image unchanged, which looks like a plausible fusion but carries no
infrared information. See fusionedge/engine/infer.py for the full guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fusionedge.engine.infer import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())