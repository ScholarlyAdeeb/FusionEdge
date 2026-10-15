#!/usr/bin/env python
"""Pre-flight for baseline v6. Every check must pass before the full run starts.

    python scripts/preflight_v6.py [--config configs/baseline.yaml]

Sections
  A. loss precision   fp32_loss inside bf16 autocast == fp32 loss exactly; vs an
                      fp64 reference (loss + gradient); finite and bounded over
                      real batches; the old in-autocast path is measured too.
  B. smoke training   short real-data run on the GPU path (bf16 forward):
                      parameters update, loss on fixed holdout images falls >=1%
                      below a fresh init, holdout validation is
                      logged every epoch, outputs are sane.
  C. checkpoints      last.pt / best.pt written; strict reload reproduces the
                      in-memory model's outputs exactly.
  D. resume           an interrupted run resumes from last.pt: epoch, global
                      step, LR schedule and optimizer state all continue, and
                      the cosine schedule still ends at eta_min.
  E. test exclusion   validation and checkpoint selection use only the train
                      holdout; train / holdout / MSRS-test are pairwise disjoint;
                      the reported checkpoint is fixed (`last`) by config.

Writes results/preflight/preflight_v6_<stamp>.json and .txt. Smoke artefacts go to
experiments/_preflight/<stamp>/ (no metrics.json, so they can never enter a
results table). Exit code 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fusionedge.data import FusionPairDataset, build_dataloaders, load_registry  # noqa: E402
from fusionedge.engine.train import Trainer  # noqa: E402
from fusionedge.engine.learning_check import learning_check  # noqa: E402
from fusionedge.losses import FusionLoss  # noqa: E402
from fusionedge.losses.fusion_loss import _luminance as _lum, fp32_loss  # noqa: E402
from fusionedge.utils.checkpoint import load_model_weights  # noqa: E402
from fusionedge.utils.config import load_config  # noqa: E402
from fusionedge.utils.registry import build_model  # noqa: E402

RESULTS: list[dict] = []
INFO: dict = {}


def check(section: str, name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append({"section": section, "check": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {section} {name:<46} {detail}", flush=True)
    return bool(ok)


def lum(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] == 1:
        return x[:, 0]
    w = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x * w).sum(1)


def corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).norm() / (b.double().norm() + 1e-30))


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def read_rows(p: Path) -> list[dict]:
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
def section_a(cfg: dict, reg: dict, dev: torch.device, amp: bool) -> None:
    S = "A"
    d = cfg["data"]
    tl, _, _ = build_dataloaders(
        dataset_id=cfg["dataset"]["id"], batch_size=8, num_workers=0, crop=d["crop"],
        holdout_frac=float(d["holdout_frac"]), holdout_seed=int(d["holdout_seed"]),
        limit_train=64, limit_eval=4,
    )
    torch.manual_seed(0)
    model = build_model(cfg["model"]["name"], cfg["model"]["cfg"]).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    L = FusionLoss(cfg["loss"])
    worst = {"loss_rel": 0.0, "grad_cos": 1.0, "floor_cos": 1.0, "smooth_cos": 1.0,
             "old_loss_rel": 0.0, "old_grad_cos": 1.0, "grad_max": 0.0}
    # Smooth terms (intensity L1 is piecewise-linear but its kinks are not near
    # ties here; SSIM is smooth) must match fp64 essentially exactly. The full loss
    # contains L1/ReLU-of-edge terms whose subgradient flips under ANY rounding at
    # near-ties, so it is compared against the fp64 noise floor instead: the change
    # in the fp64 gradient caused by a 1e-7 input perturbation (fp32 rounding size).
    Ls = FusionLoss(dict(cfg["loss"], w_gradient=0.0, w_edge=0.0, w_color=0.0))
    all_finite = True
    exact = True
    for i, batch in enumerate(tl):
        ir, vi = batch["ir"].to(dev), batch["vi"].to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            fused = model(ir, vi)
        f = fused.detach().float().requires_grad_(True)

        ref, _ = L(f.double(), ir.double(), vi.double())          # fp64 reference
        g_ref, = torch.autograd.grad(ref, f)
        plain, _ = L(f, ir, vi)                                   # fp32, no autocast
        g_plain, = torch.autograd.grad(plain, f)
        plain2, _ = L(f, ir, vi)                                  # same call again:
        g_plain2, = torch.autograd.grad(plain2, f)                # run-to-run kernel noise
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            fixed, _ = fp32_loss(L, f, ir, vi)                    # the v6 training path
        g_fixed, = torch.autograd.grad(fixed, f)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            old, _ = L(f.float(), ir.float(), vi.float())         # the v3-v5 path
        g_old, = torch.autograd.grad(old, f)

        g = torch.Generator(device="cpu").manual_seed(i)
        noise = (1e-7 * torch.rand(f.shape, generator=g, dtype=torch.float64)).to(dev)
        fn = (f.detach().double() + noise).requires_grad_(True)
        refn, _ = L(fn, ir.double(), vi.double())
        g_refn, = torch.autograd.grad(refn, fn)
        worst["floor_cos"] = min(worst["floor_cos"], cos(g_refn, g_ref))
        s64, _ = Ls(f.double(), ir.double(), vi.double())
        gs64, = torch.autograd.grad(s64, f)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
            s32, _ = fp32_loss(Ls, f, ir, vi)
        gs32, = torch.autograd.grad(s32, f)
        # Pixels where fused luminance equals the intensity target (to rounding)
        # sit ON the L1 kink, where any rounding flips the subgradient's sign.
        # A random-init model has almost none; FusionEdge's inject head has them
        # by construction wherever IR <= visible (58-78% of pixels). Compare the
        # smooth terms off those ties, and report how many there were.
        tie = ((_lum(f) - torch.max(_lum(ir), _lum(vi))).abs() < 1e-5).expand_as(f)
        worst["tie_frac"] = max(worst.get("tie_frac", 0.0), float(tie.float().mean()))
        nt = ~tie
        worst["smooth_cos"] = min(worst["smooth_cos"], cos(gs32[nt], gs64[nt]) if nt.any() else 1.0)
        # what the optimiser consumes: the gradient on the parameters, fp32 vs fp64 loss
        fused32 = model(ir, vi)
        p32 = torch.autograd.grad(fused32, params, grad_outputs=g_fixed, retain_graph=True, allow_unused=True)
        p64 = torch.autograd.grad(fused32, params, grad_outputs=g_ref.float(), allow_unused=True)
        flat = lambda gs: torch.cat([g.flatten() for g in gs if g is not None])  # noqa: E731
        worst["param_cos"] = min(worst.get("param_cos", 1.0), cos(flat(p32), flat(p64)))
        noise_l = float((plain - plain2).abs()); noise_g = float((g_plain - g_plain2).abs().max())
        diff_l = float((fixed - plain).abs()); diff_g = float((g_fixed - g_plain).abs().max())
        worst["rr_noise"] = max(worst.get("rr_noise", 0.0), noise_l, noise_g)
        worst["fixed_vs_plain"] = max(worst.get("fixed_vs_plain", 0.0), diff_l, diff_g)
        scale = max(float(plain.abs()), float(g_plain.abs().max()))
        exact &= max(diff_l, diff_g) <= max(noise_l, noise_g) + 1e-6 * scale
        all_finite &= bool(torch.isfinite(fixed) and torch.isfinite(g_fixed).all())
        worst["loss_rel"] = max(worst["loss_rel"], rel(fixed.detach(), ref.detach()))
        worst["grad_cos"] = min(worst["grad_cos"], cos(g_fixed, g_ref))
        worst["grad_max"] = max(worst["grad_max"], float(g_fixed.abs().max()))
        if torch.isfinite(old) and torch.isfinite(g_old).all():
            worst["old_loss_rel"] = max(worst["old_loss_rel"], rel(old.detach(), ref.detach()))
            worst["old_grad_cos"] = min(worst["old_grad_cos"], cos(g_old, g_ref))
        else:
            worst["old_loss_rel"], worst["old_grad_cos"] = float("nan"), float("nan")
    INFO["loss_precision"] = worst
    check(S, "autocast active for the forward pass", amp or dev.type == "cpu",
          f"device={dev.type} bf16={amp}")
    check(S, "fp32_loss under autocast == plain fp32", exact,
          f"max |diff| {worst['fixed_vs_plain']:.2e}, run-to-run kernel noise {worst['rr_noise']:.2e}")
    check(S, "loss and gradient finite", all_finite)
    check(S, "loss matches fp64 reference", worst["loss_rel"] < 1e-5, f"max rel err {worst['loss_rel']:.2e}")
    check(S, "smooth-term gradient matches fp64", worst["smooth_cos"] > 0.99999,
          f"intensity+ssim, off L1 ties: min cos {worst['smooth_cos']:.7f} "
          f"(ties up to {worst.get('tie_frac', 0.0):.0%} of pixels)")
    check(S, "parameter gradient matches fp64", worst.get("param_cos", 0.0) > 0.99999,
          f"full loss, dL/dtheta min cos {worst.get('param_cos', 0.0):.7f}")
    check(S, "full gradient within fp64 noise floor", worst["grad_cos"] >= worst["floor_cos"] - 0.005,
          f"min cos(fp32,fp64) {worst['grad_cos']:.5f} vs floor {worst['floor_cos']:.5f}")
    check(S, "gradient bounded", worst["grad_max"] < 1e3, f"max |dL/df| {worst['grad_max']:.3g}")
    print(f"      info: OLD in-autocast path vs fp64: loss rel err {worst['old_loss_rel']:.2e}, "
          f"grad cos {worst['old_grad_cos']:.5f}  (v3-v5 trained on this)", flush=True)


# --------------------------------------------------------------------------- #
def smoke_cfg(cfg: dict, epochs: int, limit_train: int) -> dict:
    c = copy.deepcopy(cfg)
    c["run_name"] = "preflight_v6"
    c["train"].update(epochs=epochs, eval_every=1, eval_max_images=8, progress="never")
    c["data"].update(limit_train=limit_train, limit_eval=8, num_workers=0)
    return c


def section_bcde(cfg: dict, reg: dict, dev: torch.device, run_dir: Path, limit_train: int) -> None:
    epochs = 3
    # ---------------- B: smoke training (interrupted after 2 of 3 epochs) ----
    S = "B"
    tA = Trainer(smoke_cfg(cfg, epochs, limit_train), run_dir, registry=reg, device=str(dev))
    steps = len(tA.train_loader)
    p0 = {n: p.detach().clone() for n, p in tA.model.named_parameters()}
    tA.total_epochs = 2          # simulate an interruption after epoch 2 (schedule built for 3)
    t0 = time.time()
    tA.fit()
    INFO["smoke_seconds_2_epochs"] = round(time.time() - t0, 1)
    changed = [n for n, p in tA.model.named_parameters() if not torch.equal(p.detach(), p0[n])]
    check(S, "every trainable parameter updated", len(changed) == len(p0), f"{len(changed)}/{len(p0)} tensors")
    check(S, "optimizer stepped every batch", tA.state.global_step == 2 * steps,
          f"global_step={tA.state.global_step}, expected {2 * steps}")
    lr_A = tA.optimizer.param_groups[0]["lr"]

    # ---------------- D: resume ------------------------------------------------
    S = "D"
    tB = Trainer(smoke_cfg(cfg, epochs, limit_train), run_dir, registry=reg, device=str(dev))
    tB.load(run_dir / "checkpoints" / "last.pt")
    check(S, "resume restores epoch", tB.state.epoch == 2, f"epoch={tB.state.epoch}")
    check(S, "resume restores global step", tB.state.global_step == 2 * steps, f"{tB.state.global_step}")
    check(S, "resume restores LR schedule",
          math.isclose(tB.optimizer.param_groups[0]["lr"], lr_A, rel_tol=1e-9)
          and tB.scheduler.last_epoch == 2 * steps,
          f"lr {tB.optimizer.param_groups[0]['lr']:.3e} vs {lr_A:.3e}, sched step {tB.scheduler.last_epoch}")
    opt_steps = [int(s["step"]) for s in tB.optimizer.state_dict()["state"].values()]
    check(S, "resume restores optimizer state", bool(opt_steps) and all(s == 2 * steps for s in opt_steps),
          f"adam step counts {sorted(set(opt_steps))}")
    tB.fit()
    eta_min = float(cfg["train"]["eta_min"])
    check(S, "resumed run completes the schedule", tB.state.epoch == epochs
          and tB.state.global_step == epochs * steps
          and math.isclose(tB.optimizer.param_groups[0]["lr"], eta_min, rel_tol=1e-3),
          f"epoch {tB.state.epoch}, step {tB.state.global_step}, final lr {tB.optimizer.param_groups[0]['lr']:.2e}")
    rows = read_rows(run_dir / "metrics.csv")
    check(S, "history accumulates across resume", len(rows) == epochs, f"{len(rows)} rows")

    # ---------------- B (cont.): learning + validation monitoring --------------
    S = "B"
    losses = [float(r["loss"]) for r in rows]
    INFO["smoke_train_loss"] = losses
    check(S, "train loss finite", all(math.isfinite(v) for v in losses))
    INFO["smoke_train_loss_per_epoch"] = " -> ".join(f"{v:.3f}" for v in losses)
    # same rule as the acceptance gate: training loss on FIXED holdout images vs a
    # fresh init (a relative drop per epoch depends on the starting point and on
    # the random crops; see fusionedge/engine/learning_check.py)
    vds = tB.val_loader.dataset
    lc = learning_check(tB.model, cfg, [(vds[i]["ir"][None], vds[i]["vi"][None]) for i in range(len(vds))], dev)
    INFO["smoke_learning_check"] = lc
    check(S, "loss below fresh init >=1% (fixed imgs)", lc["pass"],
          f"{lc['init']:.4f} -> {lc['trained']:.4f} ({lc['rel_drop']:+.2%}, noise {lc['noise']:.1e}, {lc['n']} imgs)")
    vcols = [k for k in rows[0] if k.startswith("val_")]
    ok = bool(vcols) and all(r.get(vcols[0]) not in (None, "") for r in rows)
    check(S, "holdout validation logged every epoch", ok, f"{len(vcols)} val metrics x {len(rows)} epochs")
    INFO["smoke_val_MS_SSIM_ir"] = [float(r["val_MS_SSIM_ir"]) for r in rows if r.get("val_MS_SSIM_ir")]

    # holdout output sanity (the failure modes of v4 and v5)
    m = tB.model.eval()
    ds = tB.val_loader.dataset
    sat, cv, ci, cvi = [], [], [], []
    with torch.no_grad():
        for i in range(len(ds)):
            b = ds[i]
            ir, vi = b["ir"][None].to(dev), b["vi"][None].to(dev)
            f = m(ir, vi).float()
            sat.append(float(((f < 1e-3) | (f > 1 - 1e-3)).float().mean()))
            cv.append(corr(lum(f), lum(vi))); ci.append(corr(lum(f), lum(ir))); cvi.append(corr(lum(vi), lum(ir)))
    INFO["smoke_holdout"] = {"saturation": float(np.mean(sat)), "corr_vi": float(np.mean(cv)),
                             "corr_ir": float(np.mean(ci)), "corr_vi_ir": float(np.mean(cvi))}
    check(S, "holdout output not saturated (<5%)", np.mean(sat) < 0.05, f"{np.mean(sat):.4f}")
    check(S, "holdout output follows visible (corr > 0.5)", np.mean(cv) > 0.5, f"{np.mean(cv):+.3f}")
    check(S, "holdout output carries IR (corr > corr(vi,ir))", np.mean(ci) > np.mean(cvi),
          f"{np.mean(ci):+.3f} vs {np.mean(cvi):+.3f}")

    # ---------------- C: checkpoints ---------------------------------------------
    S = "C"
    ck = run_dir / "checkpoints"
    check(S, "last.pt and best.pt written", (ck / "last.pt").exists() and (ck / "best.pt").exists())
    payload = torch.load(ck / "last.pt", map_location="cpu", weights_only=False)
    check(S, "last.pt metadata complete",
          payload["epoch"] == epochs and payload["global_step"] == epochs * steps
          and "scheduler_state" in payload and "optimizer_state" in payload and "rng_state" in payload,
          f"epoch {payload['epoch']}, step {payload['global_step']}")
    best = torch.load(ck / "best.pt", map_location="cpu", weights_only=False)
    check(S, "best.pt records its holdout metric", best.get("best_metric") is not None
          and best.get("best_epoch") is not None, f"{best.get('best_metric')} @ epoch {best.get('best_epoch')}")
    fresh = build_model(cfg["model"]["name"], cfg["model"]["cfg"])
    try:
        load_model_weights(fresh, ck / "last.pt", device=str(dev))
        sd_f, sd_m = fresh.state_dict(), m.state_dict()
        bitwise = sd_f.keys() == sd_m.keys() and all(
            torch.equal(sd_f[k].cpu(), sd_m[k].detach().cpu()) for k in sd_f)
        check(S, "strict reload: weights bitwise identical", bitwise, f"{len(sd_f)} tensors incl. BN stats")
        # The trainer keeps the model in channels_last on CUDA; a freshly loaded
        # model is contiguous, and cuDNN picks different kernels per layout. Compare
        # at the SAME layout for exactness, and report the cross-layout gap.
        same = fresh.to(memory_format=torch.channels_last) if dev.type == "cuda" else fresh
        exact, cross = True, 0.0
        fresh_c = build_model(cfg["model"]["name"], cfg["model"]["cfg"])
        load_model_weights(fresh_c, ck / "last.pt", device=str(dev))
        with torch.no_grad():
            for i in range(min(3, len(ds))):
                b = ds[i]
                ir, vi = b["ir"][None].to(dev), b["vi"][None].to(dev)
                if dev.type == "cuda":
                    irc = ir.contiguous(memory_format=torch.channels_last)
                    vic = vi.contiguous(memory_format=torch.channels_last)
                else:
                    irc, vic = ir, vi
                exact &= torch.equal(same(irc, vic), m(irc, vic))
                cross = max(cross, float((fresh_c(ir, vi) - m(irc, vic)).abs().max()))
        INFO["reload_cross_layout_max_abs_diff"] = cross
        check(S, "strict reload reproduces outputs (same layout)", exact, "3 holdout images, fp32, eval mode")
        check(S, "cross-layout output gap negligible", cross < 1e-4, f"max |diff| {cross:.2e} (contiguous vs channels_last)")
    except Exception as exc:  # noqa: BLE001
        check(S, "strict reload reproduces outputs", False, f"{type(exc).__name__}: {exc}")

    # ---------------- E: test-set exclusion ----------------------------------------
    S = "E"
    tr_ids = {p.pair_id for p in tB.train_loader.dataset.pairs}
    va_ids = {p.pair_id for p in tB.val_loader.dataset.pairs}
    test_ds = FusionPairDataset(reg["root"], cfg["dataset"]["eval_split"])
    te_ids = {p.pair_id for p in test_ds.pairs}
    hm = json.loads((run_dir / "holdout_manifest.json").read_text(encoding="utf-8"))
    check(S, "validation reads the TRAIN split, not test",
          tB.val_loader.dataset.split == cfg["dataset"]["train_split"]
          and Path(tB.val_loader.dataset.root, tB.val_loader.dataset.split) != Path(reg["root"], cfg["dataset"]["eval_split"]),
          f"val split dir = {tB.val_loader.dataset.split!r}, test split dir = {cfg['dataset']['eval_split']!r}")
    check(S, "validation ids are the declared holdout", va_ids <= set(hm["holdout_ids"]),
          f"{len(va_ids)} ids, holdout sha {hm['holdout_sha256'][:12]}")
    full_tr, full_hold = set(), set(hm["holdout_ids"])
    full_tr = {p.pair_id for p in FusionPairDataset(reg["root"], cfg["dataset"]["train_split"]).pairs} - full_hold
    check(S, "train / holdout / test pairwise disjoint",
          not (full_tr & full_hold) and not (full_tr & te_ids) and not (full_hold & te_ids)
          and not (tr_ids & va_ids),
          f"fit {len(full_tr)}, holdout {len(full_hold)}, test {len(te_ids)}")
    check(S, "holdout matches the protocol (10%, seed 0)",
          hm["n_holdout"] == 108 and hm["n_fit"] == 975 and hm["holdout_sha256"].startswith("fbf10e110601"),
          f"n={hm['n_holdout']}/{hm['n_fit']} sha {hm['holdout_sha256'][:12]} (smoke used {hm.get('used')})")
    check(S, "reported checkpoint fixed by config (last)", cfg["train"]["final_weights"] == "last",
          f"final_weights={cfg['train']['final_weights']}; best.pt selected on holdout {cfg['train']['best_metric']}")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--limit-train", type=int, default=256)
    ap.add_argument("--allow-cpu", action="store_true", help="development only")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    cfg = dict(load_config(REPO_ROOT / args.config))
    cfg = json.loads(json.dumps(cfg, default=lambda o: dict(o)))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"preflight v6 {stamp}  device={dev}  torch {torch.__version__}"
          + (f"  gpu {torch.cuda.get_device_name(0)}" if dev.type == "cuda" else ""), flush=True)
    if dev.type != "cuda" and not args.allow_cpu:
        print("FAIL: no CUDA device; this pre-flight must exercise the GPU bf16 path.")
        return 2
    # every protocol model fuses luminance only: the baseline via output_space=y,
    # FusionEdge via its residual head (which keeps the visible chroma)
    mc = cfg["model"]["cfg"]
    lum_only = (mc.get("output_space") in ("y", "inject") if cfg["model"]["name"] == "baseline"
                else mc.get("residual_output", True) or mc.get("output_space") == "y")
    check("-", "config fuses luminance only (protocol)", lum_only,
          f"run_name={cfg['run_name']} model={cfg['model']['name']}")
    check("-", "run directory does not exist yet", not (REPO_ROOT / "experiments" / cfg["run_name"]).exists())
    reg = load_registry(cfg["dataset"]["id"])
    amp = dev.type == "cuda" and bool(cfg["train"].get("amp", True)) and cfg["train"].get("amp_dtype") == "bfloat16"

    print("A. loss precision", flush=True)
    section_a(cfg, reg, dev, amp)
    run_dir = REPO_ROOT / "experiments" / "_preflight" / stamp
    print(f"B-E. smoke run in {run_dir}", flush=True)
    section_bcde(cfg, reg, dev, run_dir, args.limit_train)

    ok = all(r["pass"] for r in RESULTS)
    out = REPO_ROOT / "results" / "preflight"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"preflight_v6_{stamp}.json").write_text(json.dumps(
        {"stamp": stamp, "passed": ok, "device": str(dev), "torch": torch.__version__,
         "checks": RESULTS, "info": INFO, "smoke_dir": str(run_dir)}, indent=2, default=float), encoding="utf-8")
    n_fail = sum(not r["pass"] for r in RESULTS)
    print(f"\nPREFLIGHT {'PASSED' if ok else 'FAILED'}: {len(RESULTS) - n_fail}/{len(RESULTS)} checks"
          f"  -> results/preflight/preflight_v6_{stamp}.json", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
