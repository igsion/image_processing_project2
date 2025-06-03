import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import os
import time

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(UNet, self).__init__()
        self.enc1 = self._conv_block(in_channels, k)
        self.enc2 = self._conv_block(k, 2 * k)
        self.enc3 = self._conv_block(2 * k, 4 * k)
        self.bottleneck = self._conv_block(4 * k, 8 * k, dropout=True)
        self.up1 = nn.ConvTranspose2d(8 * k, 4 * k, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(4 * k, 2 * k, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(2 * k, k, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(8 * k, 4 * k, dropout=True)
        self.dec2 = self._conv_block(4 * k, 2 * k, dropout=True)
        self.dec3 = self._conv_block(2 * k, k, dropout=True)
        self.out = nn.Conv2d(k, out_channels, kernel_size=1)

    def _conv_block(self, in_ch, out_ch, dropout=False):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            layers.append(nn.Dropout(p=0.5))
        layers += [
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(F.max_pool2d(x1, kernel_size=2, stride=2))
        x3 = self.enc3(F.max_pool2d(x2, kernel_size=2, stride=2))
        x4 = self.bottleneck(F.max_pool2d(x3, kernel_size=2, stride=2))
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x5 = self.dec1(x)
        x = self.up2(x5)
        x = torch.cat([x, x2], dim=1)
        x6 = self.dec2(x)
        x = self.up3(x6)
        x = torch.cat([x, x1], dim=1)
        x = self.dec3(x)
        return self.out(x)

