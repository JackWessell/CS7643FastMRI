

import torch
import torch.nn as nn
from math import sqrt

from models.registry1 import decoder_entrypoints
from utils import utils


class ConvActNorm(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dialation=1,
                    groups=1, bias=True, act=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dialation,
                        groups, bias)
        self.norm = nn.GroupNorm(1, out_channels)
        self.act = act() if act is not None else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        return x


class ASSPP(nn.Module):
    """ Depthwise Separable Atrous Spatial Pyramid Pooling with residual"""

    def __init__(self, in_features, out_features=None, dw_size=3, act=nn.Hardswish):
        super().__init__()
        dilation_rates = [1,2,4,8]
        out_features = out_features if out_features else in_features
        self.fc1 = nn.Conv2d(in_features, out_features, 1, 1, 0)
        self.conv = nn.ModuleList(
            ConvActNorm(out_features, out_features, dw_size, 1, rate, rate,
                    groups=out_features, bias=False, act=None)
            for rate in dilation_rates
        )
        self.fc2 = nn.Conv2d(int(out_features*len(dilation_rates)), out_features, 1, 1, 0, bias=False)
        self.act = act()

    def forward(self, x):
        x = self.fc1(x)
        x = torch.cat([conv(x) for conv in self.conv], dim=1)
        x = self.fc2(self.act(x))
        x = nn.functional.interpolate(x, size=(80,80), mode='bilinear')
        return x

