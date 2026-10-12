"""Building blocks shared by the baseline and FusionEdge.

Design constraints that drove these choices:
  * 8 GB of VRAM and a laptop GPU -> depthwise-separable convolutions dominate
    dense 3x3 convs in the accuracy-per-FLOP band we care about (roughly
    20-100k params), so they are the default, not an afterthought.
  * A fusion network must align two *already aligned* modalities; it does not
    have to perform registration. Blocks are therefore translation-equivariant
    and global context is injected at one low-resolution stage rather than
    everywhere.
  * The edge extractor is parameter-free (fixed Sobel). Making it learnable
    would add 4 parameters and one more thing to explain; a fixed edge operator
    is a prior, and the ablation table can test whether that prior is worth it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# convolution primitives
# --------------------------------------------------------------------------- #
class ConvBNAct(nn.Sequential):
    """Dense conv -> BN -> LeakyReLU."""

    def __init__(
        self,
        cin: int,
        cout: int,
        k: int = 3,
        s: int = 1,
        d: int = 1,
        p: int | None = None,
        act: bool = True,
        negative_slope: float = 0.2,
    ):
        pad = p if p is not None else d * (k - 1) // 2
        layers: list[nn.Module] = [
            nn.Conv2d(cin, cout, k, stride=s, padding=pad, dilation=d, bias=False),
            nn.BatchNorm2d(cout),
        ]
        if act:
            layers.append(nn.LeakyReLU(negative_slope, inplace=True))
        super().__init__(*layers)


class DWConvBNAct(nn.Sequential):
    """Depthwise conv -> BN -> act. Groups == in_channels."""

    def __init__(self, channels: int, k: int = 3, d: int = 1, act: bool = True):
        pad = d * (k - 1) // 2
        layers: list[nn.Module] = [
            nn.Conv2d(channels, channels, k, padding=pad, dilation=d, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
        ]
        if act:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        super().__init__(*layers)


class SepConvBNAct(nn.Sequential):
    """Depthwise kxk -> pointwise 1x1 -> BN -> act.

    Cost for C channels at HxW: 9*C*H*W (dw) + C*C*H*W (pw) MACs, versus
    9*C*C*H*W for a dense 3x3 -- the reason these appear everywhere below.
    """

    def __init__(self, cin: int, cout: int, k: int = 3, d: int = 1, act: bool = True):
        pad = d * (k - 1) // 2
        layers: list[nn.Module] = [
            nn.Conv2d(cin, cin, k, padding=pad, dilation=d, groups=cin, bias=False),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout),
        ]
        if act:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        super().__init__(*layers)


# --------------------------------------------------------------------------- #
# residual units
# --------------------------------------------------------------------------- #
class ResidualSeparableBlock(nn.Module):
    """x + SepConv(act(SepConv(x))) with a 1x1 shortcut when widths differ.

    The pre-activation ordering (activate, then conv) is used instead of
    post-activation because the second separable conv's BN then sees an
    activation-free input, which trains a little more stably at low batch size.
    """

    def __init__(self, cin: int, cout: int | None = None, k: int = 3, d: int = 1):
        super().__init__()
        cout = cin if cout is None else cout
        self.conv1 = SepConvBNAct(cin, cout, k=k, d=d)
        self.conv2 = SepConvBNAct(cout, cout, k=k, d=1, act=False)
        self.shortcut = (
            nn.Identity() if cin == cout else nn.Conv2d(cin, cout, 1, bias=False)
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))


class DilatedSeparableBlock(ResidualSeparableBlock):
    """A residual block with an explicit dilation rate, for receptive-field growth."""


class PairedResidualBlock(nn.Module):
    """A residual block applied with *shared* weights to both modalities.

    Weight sharing across modalities is the cheap half of the parameter saving:
    one block instead of two. It also encodes an assumption worth stating --
    "the same local filter is valid in both modalities" -- which the ablation
    (`shared_stem: false`) tests by giving each branch its own weights.

    `block_cls` selects the per-modality block type; the default is the
    depthwise-separable residual block, and the dense variant is used by the
    separability ablation so that control is width- and depth-matched.
    """

    def __init__(
        self,
        channels: int,
        k: int = 3,
        d: int = 1,
        block_cls: type[nn.Module] = ResidualSeparableBlock,
    ):
        super().__init__()
        self.block = block_cls(channels, channels, k=k, d=d)

    def forward(self, ir: torch.Tensor, vi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.block(ir), self.block(vi)


# --------------------------------------------------------------------------- #
# gating / attention
# --------------------------------------------------------------------------- #
class SELite(nn.Module):
    """Squeeze-and-excite with a bottleneck ratio of 4 and a 1x1 output conv.

    "Lite" refers to the two choices that make it ~free: the bottleneck is
    C/4 rather than C/16 (cheaper, nearly the same effect at this width), and
    the channel weights are produced by a 1x1 conv rather than two FC layers, so
    the block is shape-agnostic.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(s), inplace=True)))
        return x * w


class ChannelGate(nn.Module):
    """Produce a per-modality scalar logit map from features + edges.

    Input  : (B, C+1, H, W) = [features ; edge map]
    Output : (B, H, W) real logits, later softmaxed over the modality axis.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels + 1, max(8, channels // 4), 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(max(8, channels // 4), 1, 1),
        )

    def forward(self, feat: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        """feat (B,C,H,W), edge (B,1,H,W) -> logits (B,H,W)."""
        logits = self.conv(torch.cat([feat, edge], dim=1))
        return logits.squeeze(1)


# --------------------------------------------------------------------------- #
# parameter-free edge operator
# --------------------------------------------------------------------------- #
class SobelEdgeExtractor(nn.Module):
    """Fixed Sobel gradient magnitude. Zero parameters, zero gradients.

    The operator is a prior, and this is the module that carries it. Output is
    edge energy in roughly [0, 1] (it is |grad| with reflect padding, so borders
    are valid rather than zeroed). Kept as a Module so it travels with
    `state_dict`/`to(device)` without any special-casing in the model.
    """

    def __init__(self, channels: int = 1):
        super().__init__()
        kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        ky = kx.t().contiguous()
        k = torch.stack([kx, ky])[:, None]  # (2,1,3,3)
        self.register_buffer("kernel", k, persistent=False)
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        k = self.kernel.to(dtype=x.dtype)  # (2,1,3,3)
        # (B,C,H,W) -> (B*C,1,H,W) so one single-channel kernel sweeps every plane
        xf = F.pad(x.reshape(b * c, 1, h, w), (1, 1, 1, 1), mode="replicate")
        gx = F.conv2d(xf, k[0:1])
        gy = F.conv2d(xf, k[1:2])
        mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
        return mag.view(b, c, h, w).amax(dim=1, keepdim=True)