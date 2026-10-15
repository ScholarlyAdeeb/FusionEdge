#!/usr/bin/env python
"""Apply the pre-registered seed rule (docs/PROTOCOL.md, "Seed repeats").

    python scripts/seed_rule.py results/analysis/<stamp>/summary.json

Reads the holdout metrics that scripts/analyse_fusion.py wrote for all six runs
(last.pt, same 108 holdout images) and, per metric:

  mean and sample SD over the 3 seeds of each model;
  paired d_s = baseline+inject - FusionEdge for seeds 1234, 1235, 1236;
  clear difference  <=>  |mean diff| > max(SD_a, SD_b)  AND  all d_s share a sign.

Verdict: backbone NOT supported if FusionEdge is not clearly better on >= 5/8;
SUPPORTED if FusionEdge is clearly better on >= 5/8; otherwise MIXED.
For every metric here higher is better (EN, AG, SF, MI, SSIM*, MS-SSIM*).
Writes seed_rule.json and seed_rule.md next to the summary.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

A = ["baseline_simplecnn_inject", "baseline_simplecnn_inject_s1235", "baseline_simplecnn_inject_s1236"]
B = ["fusionedge_main", "fusionedge_main_s1235", "fusionedge_main_s1236"]
SEEDS = [1234, 1235, 1236]


def main() -> int:
    path = Path(sys.argv[1])
    s = json.loads(path.read_text(encoding="utf-8"))
    missing = [r for r in A + B if r not in s["runs"]]
    if missing:
        print(f"REFUSING: runs missing from {path}: {missing}")
        return 2
    metrics = list(s["runs"][A[0]]["metrics"])
    rows, fe_clear, bi_clear = [], 0, 0
    for k in metrics:
        a = [s["runs"][r]["metrics"][k] for r in A]
        b = [s["runs"][r]["metrics"][k] for r in B]
        d = [x - y for x, y in zip(a, b)]
        ma, mb, sa, sb = st.mean(a), st.mean(b), st.stdev(a), st.stdev(b)
        same_sign = all(x > 0 for x in d) or all(x < 0 for x in d)
        clear = abs(ma - mb) > max(sa, sb) and same_sign
        winner = ("FusionEdge" if mb > ma else "baseline+inject") if clear else "no clear difference"
        fe_clear += winner == "FusionEdge"
        bi_clear += winner == "baseline+inject"
        rows.append({"metric": k, "baseline_inject": a, "fusionedge": b, "mean_bi": ma, "sd_bi": sa,
                     "mean_fe": mb, "sd_fe": sb, "paired_diff": d, "clear": clear, "result": winner})
    matches_or_beats = len(metrics) - fe_clear
    if matches_or_beats >= 5:
        verdict = "FusionEdge backbone NOT supported"
    elif fe_clear >= 5:
        verdict = "FusionEdge backbone SUPPORTED"
    else:
        verdict = "MIXED (reported per metric, no overall claim)"
    out = {"rule": "docs/PROTOCOL.md 'Seed repeats'", "seeds": SEEDS, "rows": rows,
           "fusionedge_clearly_better": fe_clear, "baseline_inject_clearly_better": bi_clear,
           "baseline_inject_matches_or_beats": matches_or_beats, "verdict": verdict,
           "source": str(path)}
    (path.parent / "seed_rule.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    lines = ["# Seed rule (pre-registered, docs/PROTOCOL.md)", "",
             "| metric | baseline+inject mean ± SD | FusionEdge mean ± SD | paired d (1234, 1235, 1236) | result |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['metric']} | {r['mean_bi']:.4f} ± {r['sd_bi']:.4f} | {r['mean_fe']:.4f} ± {r['sd_fe']:.4f} | "
                     + ", ".join(f"{x:+.4f}" for x in r["paired_diff"]) + f" | {r['result']} |")
    lines += ["", f"FusionEdge clearly better on {fe_clear}/8; baseline+inject clearly better on {bi_clear}/8; "
              f"baseline+inject matches or beats on {matches_or_beats}/8.", "", f"**Verdict: {verdict}**"]
    (path.parent / "seed_rule.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
