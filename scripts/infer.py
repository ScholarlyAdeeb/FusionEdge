#!/usr/bin/env python
"""Single-pair inference CLI.

    python scripts/infer.py --ir ir.png --vi vi.png \
        --weights experiments/fusionedge_main/checkpoints/last.pt \
        --out results/fused.png

    python scripts/infer.py --run experiments/fusionedge_main --fp32 \
        --ir data/MSRS/canonical/train/ir/00096D.png \
        --vi data/MSRS/canonical/train/vi/00096D.png --out results/demo/fused.png

`--run` loads that experiment's model config and `checkpoints/last.pt` (the
protocol's reported checkpoint). Refuses weights that look untrained: an
untrained model returns a fixed, unlearned blend that looks plausible. See
fusionedge/engine/infer.py for the full guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fusionedge.engine.infer import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())