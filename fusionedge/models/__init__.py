"""Reusable network blocks.

Naming: `*Separable` = depthwise-separable (depthwise kxk then pointwise 1x1);
`DilatedSeparable` adds a dilation rate to widen the receptive field at a
fraction of the cost of a dilated dense convolution.
"""

from .blocks import (
    ChannelGate,
    ConvBNAct,
    DilatedSeparableBlock,
    DWConvBNAct,
    PairedResidualBlock,
    SELite,
    SobelEdgeExtractor,
    ResidualSeparableBlock,
    SepConvBNAct,
)

__all__ = [
    "ConvBNAct",
    "DWConvBNAct",
    "SepConvBNAct",
    "ResidualSeparableBlock",
    "DilatedSeparableBlock",
    "PairedResidualBlock",
    "SELite",
    "ChannelGate",
    "SobelEdgeExtractor",
]