"""
Defines a residual block without an SE block for ablation testing
"""

import torch.nn as nn
from unet_residual.depthwise_seperable_conv import DepthwiseSeperableConv


class ResidualBlock(nn.Module):
    """
    Defines a residual block with Depthwise Separable Convolution.
    The SE block has been removed for ablation purposes.
    Input -> [DSC -> DSC] + Input -> Output
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Define the main path (SE block removed)
        self.conv1 = DepthwiseSeperableConv(in_channels, out_channels)
        self.conv2 = DepthwiseSeperableConv(out_channels, out_channels)

        # Define the secondary shortcut path
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            # Use 1x1 convolution to match if dimensions change
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        """
        Forward pass with residual shortcut addition but no channel attention
        """
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)

        # Removed: out = self.se(out) (No channel attention)

        # Gradient Highway shortcut addition
        out = out + residual
        return out
