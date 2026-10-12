"""Latency attribution: where does FusionEdge's full-resolution cost go?

The measured end-to-end latency of FusionEdge at 480x640 is roughly 2.3x the
baseline's while its parameter count is only ~1.6x. Parameters alone therefore do
not explain the gap, and an unexplained factor is exactly the kind of thing that
gets hand-waved in a paper. This script decomposes the forward pass per module
so the cost can be attributed rather than guessed.

Read the output as: "attention/gating at full resolution is (or is not) the
dominant term, and here is the evidence". No fusion quality is measured here and
nothing in this file is a quality claim.

    python scripts/profile_latency.py
    python scripts/profile_latency.py --size 480 640 --iters 100
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fusionedge.utils.registry import build_model  # noqa: E402


@torch.inference_mode()
def time_fn(fn, warmup: int = 20, iters: int = 100) -> float:
    """Median wall-clock ms over `iters`, after `warmup` untimed calls."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def count_conv(module: nn.Module) -> int:
    return sum(
        1 for m in module.modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", nargs=2, type=int, default=[480, 640])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--width", type=int, default=32)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required for this profile")
        return 1

    h, w = args.size
    dev = torch.device("cuda")
    ir = torch.rand(1, 1, h, w, device=dev)
    vi = torch.rand(1, 3, h, w, device=dev)

    print(f"input {h}x{w}, batch 1, fp32, CUDA, {args.iters} iters (median)\n")

    # ---------------- whole-model comparison ---------------- #
    print(f"{'model':<14}{'params':>10}{'total ms':>12}{'ms/1Mpx':>10}")
    totals = {}
    for name, cfg in (
        ("baseline", {"width": 32, "depth": 4}),
        ("fusionedge", {"width": args.width, "depth": 2}),
    ):
        model = build_model(name, cfg).to(dev).eval()
        ms = time_fn(lambda: model(ir, vi), iters=args.iters)
        totals[name] = ms
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name:<14}{n:>10,}{ms:>12.2f}{ms / (h * w / 1e6):>10.2f}")
        del model
    print()

    # ---------------- FusionEdge breakdown ---------------- #
    model = build_model(
        "fusionedge", {"width": args.width, "depth": 2}
    ).to(dev).eval()

    stem = time_fn(lambda: (model._stem(ir, True), model._stem(vi, False)), iters=args.iters)
    f_ir, f_vi = model._stem(ir, True), model._stem(vi, False)
    blocks = time_fn(lambda: [model.blocks[b](f_ir, f_vi) for b in range(len(model.blocks))],
                     iters=args.iters)
    b_ir, b_vi = f_ir, f_vi
    for b in model.blocks:
        b_ir, b_vi = b(b_ir, b_vi)
    se = time_fn(lambda: (model.se_ir(b_ir), model.se_vi(b_vi)), iters=args.iters)
    s_ir, s_vi = model.se_ir(b_ir), model.se_vi(b_vi)
    cross = time_fn(lambda: model.cross(s_ir, s_vi), iters=args.iters)
    c_ir, c_vi = model.cross(s_ir, s_vi)
    e_ir, e_vi = model.edge(ir), model.edge(vi)
    edge = time_fn(lambda: (model.edge(ir), model.edge(vi)), iters=args.iters)
    gate = time_fn(lambda: (model.gate_ir(c_ir, e_ir), model.gate_vi(c_vi, e_vi)),
                   iters=args.iters)
    detail = time_fn(lambda: model.detail(c_ir), iters=args.iters)
    d_out = model.detail(c_ir)
    head = time_fn(lambda: model.head(d_out), iters=args.iters)

    total = totals["fusionedge"]
    rows = [
        ("stem (both)", stem, count_conv(model.stem) + count_conv(model.stem_ir or nn.Identity())),
        ("branch blocks", blocks, sum(count_conv(b.block) for b in model.blocks)),
        ("SE-lite", se, count_conv(model.se_ir) + count_conv(model.se_vi)),
        ("edge (Sobel)", edge, 0),
        ("cross-modal", cross, count_conv(model.cross)),
        ("fusion gate", gate, count_conv(model.gate_ir) + count_conv(model.gate_vi)),
        ("detail head", detail, count_conv(model.detail)),
        ("output head", head, count_conv(model.head)),
    ]
    accounted = sum(ms for _, ms, _ in rows)
    print(f"{'stage':<20}{'ms':>9}{'% total':>10}{'convs':>7}")
    print("-" * 46)
    for name, ms, convs in rows:
        print(f"{name:<20}{ms:>9.2f}{100 * ms / total:>9.1f}%{convs:>7}")
    print("-" * 46)
    print(f"{'sum of stages':<20}{accounted:>9.2f}{100 * accounted / total:>9.1f}%")
    print(f"{'measured total':<20}{total:>9.2f}")
    print(
        f"\nThe unaccounted {total - accounted:.2f} ms is data movement, "
        f"elementwise ops (softmax, clamp, add) and kernel-launch overhead that\n"
        f"are not inside a single timed submodule."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())