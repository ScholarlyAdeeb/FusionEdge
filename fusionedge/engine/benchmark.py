"""Model benchmark: parameters, checkpoint size, latency, peak VRAM.

Latency protocol (stated because "inference time" is meaningless without one):
  * CUDA events for timing, not wall-clock strings,
  * `torch.inference_mode()`, fp32 unless asked otherwise,
  * a warmup phase whose length is itself measured and reported, because the
    first calls include cuDNN autotuning and lazy kernel loading,
  * median AND p95 over N timed iterations, plus the whole run's mean,
  * deterministic kernels OFF here (they are on during training), since cudnn.
    benchmark mode picks faster algorithms and latency is the thing being
    measured,
  * batch size 1, because that is the deployment case,
  * VRAM is `torch.cuda.max_memory_allocated` for the model alone.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def checkpoint_size_mb(model: nn.Module, path: str | Path | None = None) -> float:
    """Size of the weights as stored on disk (fp32) or as they would be stored."""
    if path is not None and Path(path).exists():
        return Path(path).stat().st_size / 2**20
    n = count_parameters(model, trainable_only=False)
    return n * 4 / 2**20  # fp32


@torch.inference_mode()
def benchmark_latency(
    model: nn.Module,
    size: tuple[int, int] = (480, 640),
    *,
    device: str = "cuda",
    warmup: int = 50,
    iters: int = 200,
    amp: bool = False,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Time full forward passes. Returns latency in milliseconds."""
    dev = torch.device(device)
    model = model.to(dev).eval()
    h, w = size
    ir = torch.rand(batch_size, 1, h, w, device=dev)
    vi = torch.rand(batch_size, 3, h, w, device=dev)

    use_cuda = dev.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    # --- warmup (also reported: a long warmup is itself a finding) --- #
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(max(0, warmup)):
        if amp and use_cuda:
            with torch.autocast("cuda", dtype=torch.float16):
                model(ir, vi)
        else:
            model(ir, vi)
    if use_cuda:
        torch.cuda.synchronize()
    warmup_s = time.perf_counter() - t0

    # --- timed --- #
    times: list[float] = []
    if use_cuda:
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            if amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    model(ir, vi)
            else:
                model(ir, vi)
            ends[i].record()
        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    else:
        for _ in range(iters):
            t = time.perf_counter()
            model(ir, vi)
            times.append((time.perf_counter() - t) * 1000.0)

    times.sort()
    out = {
        "size": [h, w],
        "batch_size": batch_size,
        "iters": iters,
        "warmup_iters": warmup,
        "warmup_s": round(warmup_s, 3),
        "latency_ms_median": round(times[len(times) // 2], 4),
        "latency_ms_mean": round(sum(times) / len(times), 4),
        "latency_ms_p95": round(times[int(len(times) * 0.95) - 1], 4),
        "latency_ms_min": round(times[0], 4),
        "fps": round(1000.0 / times[len(times) // 2], 2),
        "amp": amp,
        "device": str(dev),
    }

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
        model(ir, vi)
        torch.cuda.synchronize()
        out["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 2)
        out["gpu"] = torch.cuda.get_device_name(0)

    model.to("cpu")
    return out


def benchmark_model(
    model: nn.Module,
    *,
    name: str,
    weights: str | Path | None = None,
    sizes: list[list[int]] | None = None,
    warmup: int = 50,
    iters: int = 200,
    device: str = "cuda",
    amp: bool = False,
) -> dict[str, Any]:
    sizes = sizes or [[128, 128], [480, 640]]
    result: dict[str, Any] = {
        "model": name,
        "params": count_parameters(model),
        "params_all": count_parameters(model, trainable_only=False),
        "checkpoint_mb": round(checkpoint_size_mb(model, weights), 3),
        "latency": [
            benchmark_latency(
                model, tuple(s), device=device, warmup=warmup, iters=iters, amp=amp
            )
            for s in sizes
        ],
        "torch": torch.__version__,
    }
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
    return result


def write_benchmark(result: dict[str, Any], out_json: str | Path) -> Path:
    out = Path(out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out