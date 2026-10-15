"""The controlled baseline: a plain dual-input residual CNN.

This is the model every claim about FusionEdge is measured against, so it is
deliberately uninteresting and deliberately *fair*:

  * same input convention (ir (B,1,H,W), vi (B,3,H,W)),
  * same output convention (fused (B,3,H,W) in [0,1]),
  * same loss object, same data pipeline, same schedule, same seed,
  * same order of magnitude of parameters, so the comparison is not "we made it
    lighter" but "we made it better at the same size".

Its only structural choices are the two obvious ones: concatenate the two
modalities, run residual blocks, predict the fused image directly. No attention,
no edge prior, no cross-modal interaction. If FusionEdge wins, it wins because
of those, not because of a different budget.

Output head: sigmoid
-------------------
The output is `sigmoid(conv)`, NOT `clamp(conv, 0, 1)`. This was a genuine bug
found by training the model, not a design preference, and the measurement is
recorded in experiments/_failed/baseline_clamp_bug/NOTES.md:

    with clamp:   58.3% of output pixels sat exactly on 0.0 or 1.0
                  loss 13.296 -> 13.296 over 60 Adam steps (no learning)
    with sigmoid: 0.1% saturated
                  loss 8.350 -> 8.181

`torch.clamp` has zero gradient outside its range, so a pixel driven to the
boundary stops contributing to the loss entirely. Because roughly half the
pixels ended up pinned there, the baseline was effectively frozen and its metrics
would have been meaningless.

Note the direction of the correction: this fix makes the BASELINE stronger, and
FusionEdge never had the problem (its residual output `clamp(vi + delta)` starts
at exactly the visible image, so almost nothing sits on the boundary). Leaving
the bug in place would have produced a comparison rigged in FusionEdge's favour.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ConvBNAct, ResidualSeparableBlock
from .colour import inject_ir, recolour


class SimpleCNNBaseline(nn.Module):
    """Concat -> residual separable blocks -> 3x3 conv head.

    Tensor flow (B = batch, C = width):
        ir (B,1,H,W) --repeat--> (B,3,H,W) --\
                                               concat -> (B,6,H,W)
        vi (B,3,H,W) ------------------/
        stem   6->C          -> (B,C,H,W)
        blocks C->C  x depth -> (B,C,H,W)
        head   C->3          -> (B,3,H,W)
        sigmoid               -> (B,3,H,W) in (0,1)

    The IR repeat-to-3 is the one thing here worth calling out: the baseline
    pays 3*C*1*1 extra stem parameters to make the concat dimension symmetric,
    which is the standard trick and precisely what FusionEdge's 1-channel IR
    stem avoids.
    """

    def __init__(
        self,
        width: int = 32,
        depth: int = 4,
        use_relu_head: bool = False,
        output_space: str = "rgb",
    ):
        super().__init__()
        self.width = width
        self.depth = depth

        self.stem = ConvBNAct(6, width, k=3, p=1)
        self.blocks = nn.Sequential(
            *[ResidualSeparableBlock(width, width, k=3) for _ in range(depth)]
        )
        # output_space 'y' (used from v5): predict fused luminance only and take
        # chroma from the visible image -- see fusionedge/models/colour.py. v4
        # predicted RGB and invented false colour (yellow foliage, green people).
        # output_space 'inject' (added 2026-10-15): FusionEdge's IR-injection head
        # (colour.inject_ir) on this backbone, zero-initialised exactly like
        # FusionEdge's, so both start from the same rule (g = 0.5) and the
        # comparison isolates the backbone. See docs/PROTOCOL.md.
        if output_space not in ("rgb", "y", "inject"):
            raise ValueError(f"output_space must be rgb|y|inject, got {output_space!r}")
        self.output_space = output_space
        self.head = nn.Conv2d(width, 3 if output_space == "rgb" else 1, 3, padding=1)
        if output_space == "inject":
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        self.act = nn.LeakyReLU(0.2, inplace=True) if use_relu_head else nn.Identity()

    def forward(self, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        if ir.shape[1] == 1:
            ir = ir.repeat(1, 3, 1, 1)
        x = torch.cat([ir, vi], dim=1)
        x = self.stem(x)
        x = self.blocks(x)
        if self.output_space == "inject":
            return inject_ir(self.head(x), ir[:, :1], vi)
        x = self.act(self.head(x))
        # SIGMOID, not clamp. See the module docstring: a hard clamp zeroes the
        # gradient for every pixel outside [0,1], and empirically ~50% of pixels
        # sat on the boundary, so the network could not learn at all.
        #
        # use_relu_head defaults to False (changed 2026-10-13). With LeakyReLU(0.2)
        # before the sigmoid, a dark pixel (MSRS is mostly night scenes) needs a
        # pre-activation near -20 to reach 0.02, which handicapped the baseline
        # and is the likely source of the |x|~27 head activations measured in
        # experiments/_failed/baseline_fp16_amp. Runs up to v3 used True.
        if self.output_space == "y":
            return recolour(torch.sigmoid(x), vi)
        return torch.sigmoid(x)


def build_baseline(cfg: dict | None = None) -> SimpleCNNBaseline:
    cfg = cfg or {}
    return SimpleCNNBaseline(
        width=int(cfg.get("width", 32)),
        depth=int(cfg.get("depth", 4)),
        use_relu_head=bool(cfg.get("use_relu_head", False)),
        output_space=str(cfg.get("output_space", "rgb")),
    )