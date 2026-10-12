"""Quick GPU capability + throughput check.

Not an experiment and not a result: it reports hardware capability, per-model
parameter counts, CUDA-event latency at two input sizes, peak VRAM, and
training-step cost at the configured batch/crop. Run before a sweep so a memory
or throughput surprise is discovered on a 30-second check rather than 40 minutes
into a training run.

    python scripts/gpu_check.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fusionedge.engine.benchmark import benchmark_model  # noqa: E402
from fusionedge.losses import FusionLoss  # noqa: E402
from fusionedge.utils.registry import build_model  # noqa: E402


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA is not available -- stopping (this project's runs need a GPU)")
        return 1

    props = torch.cuda.get_device_properties(0)
    print(f"GPU        : {props.name}")
    print(f"VRAM       : {props.total_memory / 2**30:.1f} GiB")
    print(f"capability : sm_{props.major}{props.minor}")
    print(f"torch      : {torch.__version__} (CUDA {torch.version.cuda})")
    print()

    models = [
        ("baseline", {"width": 32, "depth": 4}),
        ("fusionedge", {"width": 32, "depth": 2}),
    ]
    loss_fn = FusionLoss().cuda()

    for name, cfg in models:
        model = build_model(name, cfg).cuda()
        bench = benchmark_model(
            model, name=name, sizes=[[128, 128], [480, 640]],
            warmup=50, iters=200, device="cuda",
        )
        print(f"{name}: {bench['params']:,} params, {bench['checkpoint_mb']:.2f} MB fp32")
        for lat in bench["latency"]:
            print(f"   {lat['size'][0]}x{lat['size'][1]}: "
                  f"{lat['latency_ms_median']:.2f} ms median | "
                  f"{lat['latency_ms_p95']:.2f} ms p95 | "
                  f"{lat['fps']:.1f} fps | peak {lat['peak_vram_mb']:.0f} MiB")

        # training-step cost at the configured batch/crop
        bs, crop = 8, 128
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        ir = torch.rand(bs, 1, crop, crop).cuda()
        vi = torch.rand(bs, 3, crop, crop).cuda()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(5):  # warm up allocator + kernels
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss, _ = loss_fn(model(ir, vi).float(), ir, vi)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n = 20
        for _ in range(n):
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss, _ = loss_fn(model(ir, vi).float(), ir, vi)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n * 1000
        peak = torch.cuda.max_memory_allocated() / 2**20
        steps_per_s = 1000 / ms
        print(f"   train step bs={bs} @{crop}x{crop}: {ms:.1f} ms "
              f"({steps_per_s:.0f} steps/s) | peak {peak:.0f} MiB "
              f"({peak / 1024:.0%} of VRAM)")
        print()

        del model, opt, scaler, ir, vi
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())