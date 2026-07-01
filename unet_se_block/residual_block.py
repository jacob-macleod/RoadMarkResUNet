"""
Defines a standard block with SE, without residual shortcuts
"""

import torch.nn as nn
from unet_se_block.depthwise_seperable_conv import DepthwiseSeperableConv
from unet_se_block.se_block import SEBlock


class ResidualBlock(nn.Module):
    """
    Defines a convolutional block with Depthwise Separable Convolution and SE.
    Residual shortcut paths have been removed for ablation purposes.
    Input -> DSC -> DSC -> SE -> Output
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Define the main sequential path
        self.conv1 = DepthwiseSeperableConv(in_channels, out_channels)
        self.conv2 = DepthwiseSeperableConv(out_channels, out_channels)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        """
        Forward pass without residual shortcut addition
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)

        # Removed: out = out + residual (No gradient highway shortcut)
        return out
