import functools

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as cp

from .utils import apply_layer, make_layer


class ResidualConvBlock(nn.Module):
    """Residual Convolutional Block"""

    def __init__(
        self, num_channels: int, bias: bool = True, memory_efficient: bool = False
    ):
        """
        Parameters:
            num_channels: Number of convolution kernels in the first layer.
            bias: Whether to use bias in the convolutional layers.
            memory_efficient: Whether to use checkpointing for memory efficiency.
        """
        super().__init__()
        self.mem_efficient = memory_efficient

        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(num_channels * 2, num_channels, 3, 1, 1, bias=bias)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.gelu(self.conv1(x))
        if self.mem_efficient:
            x2 = cp(apply_layer, [x, x1], self.conv2, use_reentrant=False)
        else:
            x2 = self.conv2(torch.cat((x, x1), 1))

        return x2 + x


class MapFeatureExtractor(nn.Module):
    """Map Feature Extractor used in the encoder and decoder of the model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_channels: int,
        num_res_blocks: int,
        memory_efficient: bool = False,
    ):
        """
        Parameters:
            in_channels: Number of channels in the input tensor.
            out_channels: Number of channels in the output tensor.
            num_channels: Number of channels in the intermediate convolution layers.
            num_res_blocks: Number of residual convolutional blocks.
            memory_efficient: Whether to use checkpointing for memory efficiency.
        """
        super().__init__()
        self.num_channels = num_channels
        self.out_channels = out_channels

        self.conv_first = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.gelu = nn.GELU()

        rcb = functools.partial(
            ResidualConvBlock,
            num_channels=num_channels,
            memory_efficient=memory_efficient,
        )
        self.rcb = make_layer(rcb, num_res_blocks)

        self.trunk_conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fea = self.conv_first(x)
        rcb = self.rcb(self.gelu(fea))
        # Encoder:
        if self.num_channels == self.out_channels:
            return fea + self.trunk_conv(rcb)
        # Decoder:
        else:
            return self.trunk_conv(rcb)
