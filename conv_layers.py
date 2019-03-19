import torch
import torch.nn as nn

import itertools

class TransConvLayer(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(TransConvLayer, self).__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
        )

    def forward(self, x):
        y = self.layers(x)
        return y

class ConvUpsample(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvUpsample, self).__init__()

        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
        )

    def forward(self, x):
        y = self.layers(x)
        return y
