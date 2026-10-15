#!/usr/bin/env python
"""Run one experiment end to end: train -> evaluate -> benchmark.

    python scripts/run_experiment.py --config configs/baseline.yaml
    python scripts/run_experiment.py --config configs/ablation_no_edge.yaml --smoke

Every artefact is written under `experiments/<run_name>/`:
    config.yaml/json   stamped provenance (dataset id + manifest hash + git commit)
    log.txt            full transcript
    metrics.csv        per-epoch history
    metrics.json       final held-out metrics with dataset provenance
    benchmark.json     params / size / latency / VRAM
    checkpoints/       last.pt, best.pt

`--smoke` runs a couple of epochs on a handful of images purely to prove the
pipeline is healthy. Smoke runs write their own directory and are marked
`"smoke": true` in metrics.json so they can never be mistaken for a real result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fusionedge.data import load_registry  # noqa: E402
from fusionedge.engine.benchmark import benchmark_model, write_benchmark  # noqa: E402
from fusionedge.engine.evaluate import run_evaluation  # noqa: E402
from fusionedge.engine.train import Trainer  # noqa: E402
from fusionedge.utils.checkpoint import load_model_weights  # noqa: E402
from fusionedge.utils.config import load_config  # noqa: E402
from fusionedge.utils.registry import build_model  # noqa: E402


def parse_overrides(pairs: list[str]) -> dict:
    """--set train.epochs=2 -> {"train": {"epochs": 2}} (YAML-parsed values)."""
    import yaml

    out: dict = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--set expects key=value, got {p!r}")
        key, _, raw = p.partition("=")
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", default=None, help="override run_name")
    ap.add_argument("--out", default="experiments")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. train.epochs=2")
    ap.add_argument(
        "--progress",
        choices=["auto", "always", "never"],
        default=None,
        help="force the progress bar (use 'always' when running in the background)",
    )
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify the pipeline")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=parse_overrides(args.set))
    if args.name:
        cfg["run_name"] = args.name
    if args.progress:
        cfg.setdefault("train", {})["progress"] = args.progress

    smoke = args.smoke
    if smoke:
        cfg.setdefault("train", {})
        cfg["train"]["epochs"] = min(2, int(cfg["train"].get("epochs", 2)))
        cfg["train"]["eval_every"] = 1
        cfg["train"]["eval_max_images"] = 4
        cfg.setdefault("data", {})
        cfg["data"]["limit_train"] = 8
        cfg["data"]["limit_eval"] = 4
        cfg["data"]["num_workers"] = 0
        cfg["data"]["batch_size"] = min(2, int(cfg["data"].get("batch_size", 2)))

    dataset_id = cfg["dataset"]["id"]
    try:
        registry = load_registry(dataset_id)
    except Exception as exc:  # noqa: BLE001
        print(f"\nBLOCKED: dataset '{dataset_id}' is not usable.\n  {exc}\n", file=sys.stderr)
        print(
            "Acquisition is a hard precondition: FusionEdge refuses to train on\n"
            "synthetic data, and refuses to report numbers against a dataset whose\n"
            "provenance is unknown. Fix the dataset first:\n"
            "  python scripts/download_data.py --dataset auto\n",
            file=sys.stderr,
        )
        return 2

    run_dir = Path(args.out) / str(cfg["run_name"])
    device = args.device or cfg.get("device")

    weights = None
    if not args.skip_train:
        trainer = Trainer(cfg, run_dir, registry=registry, device=device,
                          argv=sys.argv)
        trainer.fit(resume_from=args.resume)
        # Which checkpoint is reported is a protocol decision fixed in the config,
        # not chosen after looking at results. `last` = fixed training budget,
        # identical for every model; `best` = selected on the validation split.
        final = str(cfg.get("train", {}).get("final_weights", "best")).lower()
        if final not in ("best", "last"):
            raise SystemExit(f"train.final_weights must be best|last, got {final!r}")
        weights = run_dir / "checkpoints" / f"{final}.pt"
        if not weights.exists():
            weights = run_dir / "checkpoints" / "last.pt"
    else:
        weights = Path(args.resume) if args.resume else None
        if weights is None:
            print("--skip-train requires --resume <checkpoint>", file=sys.stderr)
            return 2

    # ---- evaluation ---------------------------------------------------- #
    from fusionedge.data import build_dataloaders

    eval_split = cfg["dataset"].get("eval_split", "val")
    _, val_loader, reg2 = build_dataloaders(
        dataset_id=dataset_id,
        batch_size=1,
        num_workers=int(cfg.get("data", {}).get("num_workers", 2)),
        crop=None,
        train_split=cfg["dataset"].get("train_split", "train"),
        eval_split=eval_split,
        limit_eval=cfg.get("data", {}).get("limit_eval"),
    )
    payload = run_evaluation(
        weights,
        cfg["model"]["name"],
        cfg["model"].get("cfg", {}),
        val_loader,
        reg2,
        device=device or ("cuda"),
        amp=bool(cfg.get("train", {}).get("amp", True)),
        out_json=run_dir / "metrics.json",
        max_images=cfg.get("train", {}).get("eval_max_images"),
    )
    payload["smoke"] = smoke
    payload["protocol"] = {
        "final_weights": str(weights),
        "eval_split": eval_split,
        "selection_split": (
            "holdout of train" if float(cfg.get("data", {}).get("holdout_frac", 0) or 0) > 0
            else eval_split
        ),
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2))

    print(f"\nheld-out metrics ({reg2['dataset_id']}, n={payload['evaluation']['num_images']}):")
    for k, v in sorted(payload["metrics"].items()):
        print(f"  {k:<14} {v:.4f}")

    # ---- benchmark ------------------------------------------------------ #
    if not args.skip_benchmark:
        model = build_model(cfg["model"]["name"], cfg["model"].get("cfg", {}))
        load_model_weights(model, weights, device="cpu")
        bcfg = cfg.get("benchmark", {})
        bench = benchmark_model(
            model,
            name=cfg["model"]["name"],
            weights=weights,
            sizes=bcfg.get("sizes", [[128, 128], [480, 640]]),
            warmup=int(bcfg.get("warmup", 50)),
            iters=int(bcfg.get("iters", 200)),
            device=device or "cuda",
            amp=bool(bcfg.get("dtype") == "float16"),
        )
        bench["run_name"] = cfg["run_name"]
        bench["dataset_id"] = reg2.get("dataset_id")
        write_benchmark(bench, run_dir / "benchmark.json")
        print(f"\nbenchmark: {bench['params']:,} params, "
              f"{bench['checkpoint_mb']:.2f} MB")
        for lat in bench["latency"]:
            print(f"  {lat['size'][0]}x{lat['size'][1]}: "
                  f"{lat['latency_ms_median']:.2f} ms median ({lat['fps']:.1f} fps)")

    if smoke:
        print("\n(smoke run -- metrics are NOT a result; re-run without --smoke)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())