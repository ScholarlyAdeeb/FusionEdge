# Architecture

This page describes the three models compared in [RESULTS.md](RESULTS.md), the output head they share, and the training loss. Every tensor shape and parameter count below comes from forward hooks on the trained configurations at a 480×640 input (batch 1). Nothing is taken from intent.

## Inputs and outputs (all models)

| tensor | shape | range | notes |
|---|---|---|---|
| `ir` | (B, 1, H, W) | [0, 1] | infrared, single channel |
| `vi` | (B, 3, H, W) | [0, 1] | visible, RGB |
| output | (B, 3, H, W) | [0, 1] | fused RGB |

All models fuse **luminance only**. The output's Cb/Cr chroma is taken unchanged from the visible image (BT.601 full range, `fusionedge/models/colour.py`). Infrared has no colour, so any chroma a network produced would be invented. Baseline v4 predicted RGB and produced false colour, which is why every later model fuses luminance only (see [EXPERIMENTS.md](EXPERIMENTS.md) and `docs/PROTOCOL.md`).

## The inject head (FusionEdge and baseline + inject)

Both inject-head models end in the same function, `colour.inject_ir`:

```
g  = sigmoid(head(x))                  (B,1,H,W)   head: 3x3 conv, C -> 1, zero-initialised
Y  = Y_vi + g * relu(IR - Y_vi)        (B,1,H,W)
out = recolour(Y, vi)                  (B,3,H,W)   Y + visible Cb/Cr -> RGB, clamped to [0,1]
```

- **Brighten only:** the gate decides how much of the infrared *excess* over the visible luminance to add, per pixel. The output can only get brighter than the visible image, and only where IR is brighter. Wherever IR ≤ Y_vi, the output equals the visible image exactly.
- **Start:** with the zero-initialised head, g = 0.5 everywhere, so an untrained model is the fixed rule "add half of any brighter IR". Both inject-head models therefore produce the same output at initialisation (unit test `test_baseline_inject_starts_at_the_same_rule_as_fusionedge`).
- **Why it replaced the original head:** the original was `clamp(vi + delta)` (`residual_mode: add`), and it never left the identity. At the identity the L1/ReLU loss terms sit on their kinks, and the 1,280-step smoke runs moved the output by at most 0.001 on average. The evidence is in `docs/PROTOCOL.md`, "FusionEdge output head". `add` remains selectable for reproducibility only.

**Limitations of the inject head:**
- **No darkening:** it cannot darken. Any information that would require lowering the visible luminance is out of reach. That suits MSRS, where thermal targets are bright, but it is a modelling assumption.
- **Hot-object texture:** where an object is very hot (the truck wheels in `docs/assets/showcase_holdout.jpg`), g near 1 replaces the visible texture with a near-uniform bright disc.
- **Night noise:** at night the visible image is near-black, so IR is brighter almost everywhere. The head then injects IR sensor grain on road surfaces as well as the targets (see `docs/assets/gate_holdout.jpg`, bottom row).
- **Gate in empty regions:** where IR ≤ Y_vi the gate has no effect and receives no gradient. Its value there (shown in the gate maps) is meaningless.

## FusionEdge (`fusionedge_main`, 22,339 parameters)

```
ir (B,1,H,W) --repeat to 3ch--> stem (shared) --> f_ir (B,32,H,W)
vi (B,3,H,W) -----------------> stem (shared) --> f_vi (B,32,H,W)
2 x PairedResidualBlock (same weights for both branches, dilations 1, 2)
SE-lite per branch          f_ir, f_vi (B,32,H,W)
CrossModalExchange          8x8 pooled tokens (T = 64), 2 heads; each branch attends to the other,
                            result upsampled bilinearly and added: f + proj(attn)   (B,32,H,W) each
Sobel edges on the RAW inputs   e_ir (B,1,H,W) from ir, e_vi (B,1,H,W) from vi (max over channels)
ChannelGate per branch      [f ; e] -> logits l_ir, l_vi (B,H,W)
alpha = softmax([l_ir, l_vi]) over the modality axis   (B,2,H,W)
fused = alpha_ir * f_ir + alpha_vi * f_vi               (B,32,H,W)
detail: 2 x ResidualSeparableBlock                      (B,32,H,W)
head + inject_ir                                        (B,3,H,W)
```

| module | parameters | in → out (480×640) |
|---|---|---|
| stem (ConvBNAct 3→32, shared) | 928 | (1,3,H,W) → (1,32,H,W) |
| blocks (2 paired residual separable) | 5,504 | (1,32,H,W) → (1,32,H,W) per branch |
| se_ir, se_vi | 552 each | (1,32,H,W) → (1,32,H,W) |
| cross (q, k, v, proj 1×1) | 4,224 | 2 × (1,32,H,W) → 2 × (1,32,H,W) |
| edge (fixed Sobel) | 0 | (1,1 or 3,H,W) → (1,1,H,W) |
| gate_ir, gate_vi | 2,393 each | (1,32,H,W) + (1,1,H,W) → (1,H,W) |
| detail (2 residual separable) | 5,504 | (1,32,H,W) → (1,32,H,W) |
| head (3×3, 32→1) | 289 | (1,32,H,W) → (1,1,H,W) |
| **total** | **22,339** | |

Residual separable block: depthwise k×k → pointwise 1×1 → BN → LeakyReLU(0.2), then depthwise → pointwise → BN, plus a skip connection, followed by a final LeakyReLU.

Every component has a config switch (`use_cross_modal`, `use_se`, `use_edge_guidance`, `adaptive_fusion`, `separable`, `shared_stem`), each with an ablation config in `configs/ablation_*.yaml`. **None of the ablations has been run** (see [RESULTS.md](RESULTS.md)).

## Baseline backbone (`baseline_simplecnn_v6`, `baseline_simplecnn_inject`, 13,089 parameters)

```
[ir repeated to 3ch ; vi] (B,6,H,W) -> stem ConvBNAct 6->32 -> 4 x ResidualSeparableBlock -> head 3x3 32->1
```

| module | parameters | in → out (480×640) |
|---|---|---|
| stem | 1,792 | (1,6,H,W) → (1,32,H,W) |
| blocks (4 residual separable) | 11,008 | (1,32,H,W) → (1,32,H,W) |
| head | 289 | (1,32,H,W) → (1,1,H,W) |

The two runs differ only in the head:
- **v6** (`output_space: y`): Y = sigmoid(head), recoloured with the visible chroma. It can brighten or darken anywhere.
- **baseline + inject** (`output_space: inject`): the inject head above, with an identical, zero-initialised head conv.

## Training loss (identical for every model)

From `fusionedge/losses/fusion_loss.py`, computed in fp32 outside the bf16 autocast block, with weights from `configs/default.yaml`:

| term | weight | definition |
|---|---|---|
| intensity | 8 | L1(Y_fused, max(Y_ir, Y_vi)) |
| gradient | 10 | L1(Sobel(Y_fused), max(Sobel(Y_ir), Sobel(Y_vi))) |
| ssim | 2 | 1 − ½(SSIM(fused, ir) + SSIM(fused, vi)) |
| edge | 1 | mean(relu(max edge of the sources − edge of fused)) |
| color | 1 | L1 of the fused chroma against the visible chroma |

The loss has no ground-truth fused image, because MSRS has none. A lower loss means closer agreement with these hand-designed targets, not "more correct".

## Cost (measured, `benchmark.json` of each run)

The benchmark uses batch size 1, fp32, CUDA events, and cuDNN benchmark mode on an RTX 4070 Laptop GPU. TF32 is enabled for latency only (`docs/PROTOCOL.md`).

| model | params | median latency 480×640 | peak VRAM 480×640 |
|---|---|---|---|
| baseline v6 | 13,089 | 16.7 ms | 199 MB |
| baseline + inject | 13,089 | 16.9 ms | 199 MB |
| FusionEdge | 22,339 | 47.0 ms | 454 MB |

FusionEdge uses 1.7× the parameters of the baseline but runs 2.8× slower, so parameter count alone understates its cost. Which modules account for the latency has **not** been profiled. Two full-resolution branches, the gates and the per-pixel softmax are plausible causes but unmeasured. `scripts/profile_latency.py` exists for that measurement.
