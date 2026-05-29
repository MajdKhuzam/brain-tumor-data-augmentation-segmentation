"""
model.py
--------
U-Net architecture for binary brain-tumour segmentation.
"""

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchsummary import summary

from config import DEVICE, IMAGE_SIZE


class DoubleConv(nn.Module):
    """Two consecutive Conv2d → BatchNorm → ReLU blocks."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=(32, 64, 128, 256)):
        super().__init__()
        self.downs        = nn.ModuleList()
        self.ups          = nn.ModuleList()
        self.pool         = nn.MaxPool2d(2, 2)
        self.out_channels = out_channels

        # Encoder
        for f in features:
            self.downs.append(DoubleConv(in_channels, f))
            in_channels = f

        # Decoder
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f * 2, f))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for i in range(0, len(self.ups), 2):
            x    = self.ups[i](x)
            skip = skip_connections[i // 2]
            if x.shape != skip.shape:
                x = TF.resize(x, size=skip.shape[2:])
            x = self.ups[i + 1](torch.cat((skip, x), dim=1))

        return self.final_conv(x)


def build_model(device=DEVICE):
    model = UNET(in_channels=1, out_channels=1).to(device)

    # Sanity check on a temporary instance to avoid polluting GPU memory
    _x = torch.randn(2, 1, IMAGE_SIZE, IMAGE_SIZE).to(device)
    _m = UNET().to(device)
    _o = _m(_x)
    assert _o.shape == _x.shape, f"Shape mismatch: {_o.shape} vs {_x.shape}"
    print(f"Input: {_x.shape}  →  Output: {_o.shape}  ✓")
    summary(_m, (1, IMAGE_SIZE, IMAGE_SIZE))
    del _x, _m, _o

    return model