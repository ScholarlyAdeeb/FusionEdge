#!/usr/bin/env python
"""README figures from the accepted runs, on MSRS model-selection HOLDOUT images.

    python scripts/make_showcase.py

Writes (JPEG, quality 90, committed so the README renders):
  docs/assets/showcase_holdout.jpg  rows = scenes; IR | visible | each run (last.pt, fp32)
  docs/assets/gate_holdout.jpg      FusionEdge: visible | fused | gate g | dY (brighter = red)
The scenes are fixed below (two day, two night, all from the 108-image holdout of
every run's holdout_manifest.json), chosen to show the failure modes discussed in
docs/RESULTS.md: hot-object texture, night brightening, colour.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from fusionedge.data import FusionPairDataset, load_registry  # noqa: E402
from fusionedge.engine.evaluate import tensor_to_uint8, tensor_to_uint8_gray  # noqa: E402
from fusionedge.models.colour import rgb_to_ycbcr  # noqa: E402
from fusionedge.utils.checkpoint import load_model_weights  # noqa: E402
from fusionedge.utils.registry import build_model  # noqa: E402

SCENES = ["00057D.png", "00096D.png", "00868N.png", "00950N.png"]
RUNS = [("baseline_simplecnn_v6", "baseline v6 (Y head)"),
        ("baseline_simplecnn_inject", "baseline + inject head"),
        ("fusionedge_main", "FusionEdge (inject head)")]
TILE = (320, 240)


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> int:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hm = json.loads((REPO_ROOT / "experiments" / RUNS[0][0] / "holdout_manifest.json").read_text(encoding="utf-8"))
    for s in SCENES:
        assert s in hm["holdout_ids"], f"{s} is not a holdout image"
    ds = FusionPairDataset(load_registry("MSRS")["root"], hm["source_split"], pairs=SCENES)
    models = {}
    for run, _ in RUNS:
        cfg = yaml.safe_load((REPO_ROOT / "experiments" / run / "config.yaml").read_text(encoding="utf-8"))
        m = build_model(cfg["model"]["name"], cfg["model"]["cfg"]).to(dev).eval()
        load_model_weights(m, REPO_ROOT / "experiments" / run / "checkpoints" / "last.pt", device=str(dev))
        models[run] = m
    gate = {}
    models["fusionedge_main"].head.register_forward_hook(lambda mod, i, o: gate.__setitem__("g", torch.sigmoid(o)))

    rows, grows = [], []
    with torch.no_grad():
        for i, sid in enumerate(SCENES):
            b = ds[i]
            ir, vi = b["ir"][None].to(dev), b["vi"][None].to(dev)
            r8 = cv2.cvtColor(tensor_to_uint8_gray(ir.cpu()), cv2.COLOR_GRAY2RGB)
            v8 = tensor_to_uint8(vi.cpu())
            tiles = [label(cv2.resize(r8, TILE), f"IR  {sid[:-4]}"), label(cv2.resize(v8, TILE), "visible")]
            fused = {}
            for run, name in RUNS:
                f = models[run](ir, vi).float().clamp(0, 1)
                fused[run] = f
                tiles.append(label(cv2.resize(tensor_to_uint8(f.cpu()), TILE), name))
            rows.append(np.concatenate(tiles, axis=1))
            if i in (1, 2):  # one day, one night scene for the gate figure
                models["fusionedge_main"](ir, vi)
                g8 = (gate["g"][0, 0].cpu().numpy() * 255).astype(np.uint8)
                g8 = cv2.cvtColor(cv2.applyColorMap(g8, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
                dy = (rgb_to_ycbcr(fused["fusionedge_main"])[:, :1] - rgb_to_ycbcr(vi)[:, :1])[0, 0].cpu().numpy()
                d8 = np.clip(dy / 0.4 * 255, 0, 255).astype(np.uint8)  # 0 = unchanged (black), 0.4+ = full red
                d8 = cv2.cvtColor(cv2.applyColorMap(d8, cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB)
                grows.append(np.concatenate([
                    label(cv2.resize(v8, TILE), f"visible  {sid[:-4]}"),
                    label(cv2.resize(tensor_to_uint8(fused['fusionedge_main'].cpu()), TILE), "FusionEdge fused"),
                    label(cv2.resize(g8, TILE), "gate g (0 dark .. 1 yellow)"),
                    label(cv2.resize(d8, TILE), "dY added (black 0 .. 0.4)")], axis=1))
    out = REPO_ROOT / "docs" / "assets"
    out.mkdir(parents=True, exist_ok=True)
    for name, img in [("showcase_holdout.jpg", np.concatenate(rows, axis=0)),
                      ("gate_holdout.jpg", np.concatenate(grows, axis=0))]:
        cv2.imwrite(str(out / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"wrote docs/assets/{name} {img.shape[1]}x{img.shape[0]} "
              f"{(out / name).stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
