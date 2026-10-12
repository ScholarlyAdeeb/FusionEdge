"""FusionEdge: a lightweight dual-branch fusion network with edge-aware adaptive
fusion.

This is the project's contribution, so every design choice below is written as a
claim that the ablation table in docs/EXPERIMENTS.md is expected to test. If an
ablation does not support a component, the component should be removed rather
than defended -- the flags in `__init__` exist to make that removal a config
change, not a rewrite.

Tensor-shape walkthrough (B=batch, C=width, H,W=spatial, T=64 pooled tokens)

    ir (B,1,H,W) ─┐
                  ├─ repeat IR to 3ch ─┐
    vi (B,3,H,W) ─┘                   ├─ shared stem (3→C) ─┐
                                       │                     ├─ F_ir (B,C,H,W)
                                       └─────────────────────┘
    (identical block applied to both, weights shared)

    F_ir, F_vi (B,C,H,W)
      ├─ SELite channel gate                       -> g_ir, g_vi (B,C,H,W)
      ├─ edge head: fixed Sobel on the *input*    -> E_ir, E_vi (B,1,H,W)
      ├─ cross-modal exchange on 8x8 pooled tokens -> modulated F_ir, F_vi
      └─ ChannelGate([F, E]) per modality          -> logits (B,H,W)
         A = softmax([l_ir, l_vi], dim=modality)   -> (B,H,W), sums to 1
         f = A_ir * F_ir + A_vi * F_vi             -> (B,C,H,W)

    residual detail head: 2 x residual separable block -> head -> ΔL (B,1,H,W)
    out = clamp(vi_rgb + ΔL, 0, 1)                 -> (B,3,H,W)

Why the output is a *residual on the visible image* rather than a fresh RGB
prediction: colour is not recoverable from infrared, so asking the network to
predict three channels from scratch invites it to invent chroma. Predicting a
luminance residual lets the network do the one thing it is good at (deciding
which edges to keep) while colour is inherited from the visible image by
construction.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    ChannelGate,
    ConvBNAct,
    PairedResidualBlock,
    SELite,
    SobelEdgeExtractor,
    ResidualSeparableBlock,
)


class CrossModalExchange(nn.Module):
    """Bounded cross-modal attention on a coarse token grid.

    Attention runs on an 8x8 adaptive average pooling of the feature maps, so
    the cost is 64x64 attention entries regardless of image size -- at 480x640
    that is the difference between 19200^2 and 64^2 attention logits. The
    result is broadcast back as a per-channel gain, which modulates the
    full-resolution features.

    This is deliberately *not* full-resolution cross-attention: the ablation
    `use_cross_modal: false` measures whether the coarse exchange earns its
    parameters at all.
    """

    def __init__(self, channels: int, grid: int = 8, heads: int = 2):
        super().__init__()
        self.grid = grid
        self.heads = heads
        self.head_dim = max(1, channels // heads)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) -> (B*heads, T, head_dim) with T = grid*grid."""
        b = x.shape[0]
        t = F.adaptive_avg_pool2d(x, self.grid).flatten(2)  # (B,C,T)
        t = t.transpose(1, 2)  # (B,T,C)
        t = t.reshape(b * self.heads, self.grid * self.grid, self.head_dim)
        return t

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x attends to y and vice versa. Returns (x', y') with y' shaped like x."""
        b, c, h, w = x.shape
        qx, kx, vx = self._tokens(self.q(x)), self._tokens(self.k(x)), self._tokens(self.v(x))
        qy, ky, vy = self._tokens(self.q(y)), self._tokens(self.k(y)), self._tokens(self.v(y))

        scale = self.head_dim**-0.5
        ax = torch.softmax(qx @ ky.transpose(-2, -1) * scale, dim=-1) @ vx
        ay = torch.softmax(qy @ kx.transpose(-2, -1) * scale, dim=-1) @ vx

        def to_map(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
            m = t.reshape(like.shape[0], self.heads, self.grid * self.grid, self.head_dim)
            m = m.permute(0, 2, 1, 3).reshape(like.shape[0], c, self.grid, self.grid)
            return F.interpolate(m, size=like.shape[-2:], mode="bilinear", align_corners=False)

        return self.proj(to_map(ax, x)) + x, self.proj(to_map(ay, y)) + y


class FusionEdge(nn.Module):
    """Dual-branch lightweight fusion network.

    Args (all are ablation switches -- see configs/ablation_*.yaml):
        width: channel count C of the branches.
        depth: residual separable blocks per branch.
        dilations: dilation rates for those blocks.
        use_cross_modal: coarse cross-modal exchange (default True).
        adaptive_fusion: per-pixel softmax gate (default True). When False the
            branches are averaged, which is the "plain concat/average" control.
        use_edge_guidance: condition the gate on Sobel edge maps (default True).
        use_se: per-branch channel attention (default True).
        separable: depthwise-separable blocks (default True). When False, dense
            3x3 convs of the same width are used -- a FLOP-matched control that
            isolates *separability* from *width*.
        residual_output: predict a luminance residual on the visible image
            (default True). When False the network regresses RGB directly.
        shared_stem: share the stem weights across modalities (default True).
    """

    def __init__(
        self,
        width: int = 32,
        depth: int = 2,
        dilations: tuple[int, ...] = (1, 2),
        *,
        use_cross_modal: bool = True,
        adaptive_fusion: bool = True,
        use_edge_guidance: bool = True,
        use_se: bool = True,
        separable: bool = True,
        residual_output: bool = True,
        shared_stem: bool = True,
        token_grid: int = 8,
    ):
        super().__init__()
        self.cfg = dict(
            width=width,
            depth=depth,
            dilations=list(dilations),
            use_cross_modal=use_cross_modal,
            adaptive_fusion=adaptive_fusion,
            use_edge_guidance=use_edge_guidance,
            use_se=use_se,
            separable=separable,
            residual_output=residual_output,
            shared_stem=shared_stem,
            token_grid=token_grid,
        )

        # ---- shared modality-agnostic stem -------------------------------
        # IR is replicated to 3 channels so ONE stem handles both modalities.
        # Replication costs nothing in parameters (the stem's first conv sees 3
        # input channels either way) and halves the stem compared to two stems.
        self.shared_stem = shared_stem
        self.stem = ConvBNAct(3, width, k=3, p=1)
        self.stem_ir = None if shared_stem else ConvBNAct(3, width, k=3, p=1)

        # ---- per-branch residual blocks (weights shared) ------------------
        blk = ResidualSeparableBlock if separable else _DenseResidualBlock
        dil = list(dilations)[:depth]
        dil = dil + [dil[-1]] * max(0, depth - len(dil))
        self.blocks = nn.ModuleList(
            [PairedResidualBlock(width, k=3, d=d, block_cls=blk) for d in dil]
        )

        # ---- channel attention (SE-lite) ----------------------------------
        self.se_ir = SELite(width) if use_se else nn.Identity()
        self.se_vi = SELite(width) if use_se else nn.Identity()

        # ---- edge head: fixed Sobel on the raw inputs ---------------------
        self.use_edge_guidance = use_edge_guidance
        self.edge = SobelEdgeExtractor()

        # ---- cross-modal exchange ----------------------------------------
        self.use_cross_modal = use_cross_modal
        self.cross = CrossModalExchange(width, grid=token_grid) if use_cross_modal else None

        # ---- adaptive fusion gate ----------------------------------------
        self.adaptive_fusion = adaptive_fusion
        self.gate_ir = ChannelGate(width)
        self.gate_vi = ChannelGate(width)

        # ---- reconstruction head -----------------------------------------
        self.residual_output = residual_output
        self.detail = nn.Sequential(*[ResidualSeparableBlock(width, width) for _ in range(2)])
        out_ch = 1 if residual_output else 3
        self.head = nn.Conv2d(width, out_ch, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # a zero-initialised head makes the model start as the identity
        # (output == visible image), which is a sane starting point for an
        # unsupervised objective and makes early training curves readable.

    # ------------------------------------------------------------------ #
    def _stem(self, x: torch.Tensor, is_ir: bool) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.stem(x) if (self.shared_stem or not is_ir) else self.stem_ir(x)

    def forward(self, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor:
        f_ir = self._stem(ir, True)
        f_vi = self._stem(vi, False)

        for block in self.blocks:
            f_ir, f_vi = block(f_ir, f_vi)

        f_ir = self.se_ir(f_ir)
        f_vi = self.se_vi(f_vi)

        if self.cross is not None:
            f_ir, f_vi = self.cross(f_ir, f_vi)

        if self.adaptive_fusion:
            if self.use_edge_guidance:
                e_ir, e_vi = self.edge(ir), self.edge(vi)
            else:
                # zero edges: the gate still sees the (zero) edge channel, so the
                # parameter count is unchanged and only the information is removed
                z = torch.zeros_like(f_ir[:, :1])
                e_ir = e_vi = z
            l_ir = self.gate_ir(f_ir, e_ir)  # (B,H,W)
            l_vi = self.gate_vi(f_vi, e_vi)
            # softmax over the MODALITY axis: every pixel gets a convex
            # combination of the two branches, so no pixel can be dropped
            alpha = torch.softmax(torch.stack([l_ir, l_vi], dim=1), dim=1)  # (B,2,H,W)
            a_ir, a_vi = alpha[:, 0:1], alpha[:, 1:2]
            fused = a_ir * f_ir + a_vi * f_vi
            self._last_alpha = alpha.detach()
        else:
            fused = 0.5 * (f_ir + f_vi)
            self._last_alpha = None

        x = self.detail(fused)
        delta = self.head(x)

        if self.residual_output:
            return torch.clamp(vi + delta, 0.0, 1.0)
        return torch.clamp(delta, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def fusion_weights(self, ir: torch.Tensor, vi: torch.Tensor) -> torch.Tensor | None:
        """Run a forward pass and return the per-pixel modality weights (B,2,H,W).

        Exposed so the edge-aware gate can be *visualised* and argued about --
        "the network prefers infrared where the thermal target is" is a claim
        that should be lookable at, not taken on trust.
        """
        self.forward(ir, vi)
        return self._last_alpha

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class _DenseResidualBlock(nn.Module):
    """Width-matched dense 3x3 control for the separability ablation.

    Same channel count and same number of blocks as the separable version, so
    the parameter/FLOP gap is explicit rather than accidental. It subclasses
    nn.Module directly (not the separable block) because the point is that the
    convolutions are *not* separable.
    """

    def __init__(self, cin: int, cout: int | None = None, k: int = 3, d: int = 1):
        super().__init__()
        cout = cin if cout is None else cout
        pad = d * (k - 1) // 2

        def conv(a: int, b: int, act: bool) -> nn.Sequential:
            layers: list[nn.Module] = [
                nn.Conv2d(a, b, k, padding=pad, dilation=d, bias=False),
                nn.BatchNorm2d(b),
            ]
            if act:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.conv1 = conv(cin, cout, act=True)
        self.conv2 = conv(cout, cout, act=False)
        self.shortcut = nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1, bias=False)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))


def build_fusionedge(cfg: dict[str, Any] | None = None) -> FusionEdge:
    cfg = dict(cfg or {})
    known = {
        "width", "depth", "dilations", "use_cross_modal", "adaptive_fusion",
        "use_edge_guidance", "use_se", "separable", "residual_output",
        "shared_stem", "token_grid",
    }
    unknown = set(cfg) - known
    if unknown:
        raise ValueError(f"unknown FusionEdge config keys: {sorted(unknown)}; valid: {sorted(known)}")
    return FusionEdge(**cfg)