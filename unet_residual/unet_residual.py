"""
Define the mobule Res Unet model
"""

import torch.nn as nn
import torch
import cv2
from mobile_res_u_net.residual_block import ResidualBlock


class UnetResidual(nn.Module):
    """
    Define the mobule Res Unet model
    """

    def __init__(self, n_channels=3, n_classes=1):
        super(UnetResidual, self).__init__()
        self.inc = ResidualBlock(
            n_channels, 32
        )  # Start smaller for speed (32 instead of 64)

        # Downsampling
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(128, 256))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), ResidualBlock(256, 512))

        # Upsampling
        # Using Bilinear upsampling instead of ConvTranspose for speed & preventing checkerboard artifacts
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = ResidualBlock(512 + 256, 256)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv2 = ResidualBlock(256 + 128, 128)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv3 = ResidualBlock(128 + 64, 64)

        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv4 = ResidualBlock(64 + 32, 32)

        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x):
        """
        Pass foward
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5)
        if x.shape != x4.shape:
            x = nn.functional.interpolate(
                x, size=x4.shape[2:], mode="bilinear", align_corners=True
            )

        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)

        return self.outc(x)
