"""
Defines a residual block
"""

import torch.nn as nn
from mobile_res_u_net.depthwise_seperable_conv import DepthwiseSeperableConv
from mobile_res_u_net.se_block import SEBlock

class ResidualBlock(nn.Module):
    """
    Defines a residual block, with Depthwise Seperable Convolution
    Input -> [DSC -> DSC] + Input -> Output
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Define the main path
        self.conv1 = DepthwiseSeperableConv(in_channels, out_channels)
        self.conv2 = DepthwiseSeperableConv(out_channels, out_channels)
        self.se = SEBlock(out_channels)

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
        Forward the output
        """
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        # Allows the ResNet part to be applied - a Gradient Highway
        out = out + residual
        return out
