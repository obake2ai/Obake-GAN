import torch
import argparse
import os
import numpy as np
import math
import sys
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.utils as vutils
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from noise_layers import *
from conv_layers import *

class NoiseGeneratorSimple(nn.Module):
    def __init__(self, opt):
        super(NoiseGeneratorSimple, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *block(128, 256, 0.1),
            *block(256, 512, 0.1),
            *block(512, 1024, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class ArgNoiseGeneratorSimple(nn.Module):
    def __init__(self, opt):
        super(ArgNoiseGeneratorSimple, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [AlgorithmicNoiseLayer(in_feat, out_feat, level, normalize)]
            return layers

        self.model = nn.Sequential(
            AlgorithmicNoiseLayer(opt.latent_dim, 128, 1, noise_seed=0, normalize=False),
            AlgorithmicNoiseLayer(128, 256, 1, noise_seed=1),
            AlgorithmicNoiseLayer(256, 512, 1, noise_seed=2),
            AlgorithmicNoiseLayer(512, 1024, 1, noise_seed=3),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class ArgNoiseGeneratorDeeper(nn.Module):
    def __init__(self, opt):
        super(ArgNoiseGeneratorDeeper, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [AlgorithmicNoiseLayer(in_feat, out_feat, level, normalize)]
            return layers

        self.model = nn.Sequential(
            AlgorithmicNoiseLayer(opt.latent_dim, 128, 1, noise_seed=0, normalize=False),
            AlgorithmicNoiseLayer(128, 256, 1, noise_seed=1),
            AlgorithmicNoiseLayer(256, 512, 1, noise_seed=2),
            AlgorithmicNoiseLayer(512, 512, 1, noise_seed=3),
            AlgorithmicNoiseLayer(512, 1024, 1, noise_seed=3),
            AlgorithmicNoiseLayer(1024, 1024, 1, noise_seed=3),
            AlgorithmicNoiseLayer(1024, 1024, 1, noise_seed=3),
            AlgorithmicNoiseLayer(1024, 1024, 1, noise_seed=3),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseDiscriminator(nn.Module):
    def __init__(self, opt):
        super(NoiseDiscriminator, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(int(np.prod(self.img_shape)), 512, 0.1),
            *block(512, 256, 0.1),
            *block(256, 1, 0.1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity

class NoiseGeneratorDeeper(nn.Module):
    def __init__(self, opt):
        super(NoiseGeneratorDeeper, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *block(128, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseGeneratorDeeperWider(nn.Module):
    def __init__(self, opt):
        super(NoiseGeneratorDeeperWider, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *block(128, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 10240, 0.1),
            *block(10240, 10240, 0.1),
            *block(10240, 10240, 0.1),
            *block(10240, 10240, 0.1),
            nn.Linear(10240, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseGeneratorDeeperWiderMini(nn.Module):
    def __init__(self, opt):
        super(NoiseGeneratorDeeperWiderMini, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *block(128, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 2048, 0.1),
            *block(2048, 2048, 0.1),
            *block(2048, 2048, 0.1),
            *block(2048, 2048, 0.1),
            nn.Linear(2048, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseGeneratorDeeperDeeper(nn.Module):
    def __init__(self, opt):
        super(NoiseGeneratorDeeperDeeper, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *block(128, 128, 0.1),
            *block(128, 256, 0.1),
            *block(256, 256, 0.1),
            *block(256, 256, 0.1),
            *block(256, 512, 0.1),
            *block(512, 512, 0.1),
            *block(512, 512, 0.1),
            *block(512, 512, 0.1),
            *block(512, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            *block(1024, 1024, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGenerator(nn.Module):
    def __init__(self, opt):
        super(NoiseResGenerator, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *block(128, 512, 0.1),
            *block(512, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorSand(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorSand, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 128),
            *block(128, 256, 0.1),
            *block(256, 512, 0.1),
            *block(512, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorSand2(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorSand2, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 128),
            *block(128, 256, 0.1),
            *block(256, 512, 0.1),
            *block(512, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *block(1024, 2048, 0.1),
            *resblock(2048, 2048, 0.1),
            nn.Linear(2048, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorIntent(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorIntent, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 1024, 0.1, normalize=False),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorEco(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorEco, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *block(128, 256, 0.1),
            *block(256, 512, 0.1),
            *block(512, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *block(1024,int(np.prod(self.img_shape)), 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorEcoLongA(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorEcoLongA, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *resblock(128, 128, 0.1),
            *block(128, 256, 0.1),
            *resblock(256, 256, 0.1),
            *block(256, 512, 0.1),
            *resblock(512, 512, 0.1),
            *block(512, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *block(1024,int(np.prod(self.img_shape)), 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorHead(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorHead, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 256),
            *block(256, 512, 0.1),
            *resblock(512, 512, 0.1),
            *block(512, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *block(1024,int(np.prod(self.img_shape)), 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorTail(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorTail, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1),
            *block(128, 256, 0.1),
            *block(256, 512, 0.1),
            *resblock(512, 512, 0.1),
            *block(512, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorEcoWide(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorEcoWide, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 128),
            *block(128, 256, 0.1),
            *block(256, 512, 0.1),
            *resblock(512, 512, 0.1),
            *block(512, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *block(1024, 4096, 0.1),
            *resblock(4096, 4096, 0.1),
            *block(4096, int(np.prod(self.img_shape)), 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorEcoWide2(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorEcoWide2, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 128),
            *block(128, 256, 0.1),
            *block(256, 512, 0.1),
            *block(512, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *block(1024, 4096, 0.1),
            *resblock(4096, 4096, 0.1),
            *resblock(4096, 4096, 0.1),
            *block(4096, int(np.prod(self.img_shape)), 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorEcoWideWide(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorEcoWideWide, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 128),
            *block(128, 256, 0.1),
            *block(256, 512, 0.1),
            *block(512, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *block(1024, 4096, 0.1),
            *resblock(4096, 4096, 0.1),
            *block(4096, 8192, 0.1),
            *resblock(8192, 8192, 0.1),
            *block(8192, int(np.prod(self.img_shape)), 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorEcoBottle(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorEcoBottle, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            nn.Linear(opt.latent_dim, 128),
            *block(128, 256, 0.1),
            *block(256, 512, 0.1),
            *block(512, 1024, 0.1),
            *block(1024, 4096, 0.1),
            *resblock(4096, 4096, 0.1),
            *block(4096, 8192, 0.1),
            *resblock(8192, 8192, 0.1),
            *block(8192, 1024, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorEcoLongB(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorEcoLongB, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *block(128, 256, 0.1),
            *block(256, 512, 0.1),
            *block(512, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *block(1024,int(np.prod(self.img_shape)), 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorEcoLongC(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorEcoLongC, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *resblock(128, 128, 0.1),
            *resblock(128, 128, 0.1),
            *block(128, 256, 0.1),
            *resblock(256, 256, 0.1),
            *resblock(256, 256, 0.1),
            *resblock(256, 256, 0.1),
            *block(256, 512, 0.1),
            *resblock(512, 512, 0.1),
            *resblock(512, 512, 0.1),
            *resblock(512, 512, 0.1),
            *resblock(512, 512, 0.1),
            *block(512, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *block(1024,int(np.prod(self.img_shape)), 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class ArgNoiseResGenerator(nn.Module):
    def __init__(self, opt):
        super(ArgNoiseResGenerator, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, seed, level, normalize=True):
            layers = [AlgorithmicNoiseLayer(in_feat, out_feat, seed, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, seed, level, normalize=True, shortcut=None):
            layers = [ArgNoiseBasicBlock(in_feat, out_feat, seed, 1, shortcut, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 10, 0.1, normalize=False),
            *block(128, 512, 20, 0.1),
            *block(512, 1024, 30, 0.1),
            *resblock(1024, 1024, 40, 0.1),
            *resblock(1024, 1024, 50, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class ArgNoiseResGeneratorLonger(nn.Module):
    def __init__(self, opt):
        super(ArgNoiseResGeneratorLonger, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, seed, level, normalize=True):
            layers = [AlgorithmicNoiseLayer(in_feat, out_feat, seed, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, seed, level, normalize=True, shortcut=None):
            layers = [ArgNoiseBasicBlock(in_feat, out_feat, seed, 1, shortcut, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 10, 0.1, normalize=False),
            *block(128, 512, 20, 0.1),
            *block(512, 1024, 30, 0.1),
            *resblock(1024, 1024, 40, 0.1),
            *resblock(1024, 1024, 50, 0.1),
            *resblock(1024, 1024, 60, 0.1),
            *resblock(1024, 1024, 70, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class ArgNoiseResGeneratorIntent(nn.Module):
    def __init__(self, opt):
        super(ArgNoiseResGeneratorIntent, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, seed, level, normalize=True):
            layers = [AlgorithmicNoiseLayer(in_feat, out_feat, seed, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, seed, level, normalize=True, shortcut=None):
            layers = [ArgNoiseBasicBlock(in_feat, out_feat, seed, 1, shortcut, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 1024, 10, 0.1, normalize=False),
            *resblock(1024, 1024, 40, 0.1),
            *resblock(1024, 1024, 50, 0.1),
            *resblock(1024, 1024, 60, 0.1),
            *resblock(1024, 1024, 70, 0.1),
            *resblock(1024, 1024, 80, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseResGeneratorW(nn.Module):
    def __init__(self, opt):
        super(NoiseResGeneratorW, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers
        def resblock(in_feat, out_feat, level, normalize=True):
            layers = [NoiseBasicBlock(in_feat, out_feat, stride=1, shortcut=None, level=level, normalize=normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *block(128, 1024, 0.1),
            *resblock(1024, 1024, 0.1),
            *block(1024, 5012, 0.1),
            *resblock(5012, 5012, 0.1),
            nn.Linear(5012, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img

class NoiseGenerator2Dv1(nn.Module):
    def __init__(self, opt):
        super(NoiseGenerator2Dv1, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            NoiseLayer2D(128 * 8, 128 * 4, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            NoiseLayer2D(128 * 4, 128 * 2, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            NoiseLayer2D(128 * 2, 128 * 1, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            NoiseLayer2D(128 * 1, channels, 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class NoiseGenerator2Dv1_(nn.Module):
    def __init__(self, opt):
        super(NoiseGenerator2Dv1_, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.l1 = nn.Sequential(
            NoiseLayer2D(128 * 8, 128 * 4, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            NoiseLayer2D(128 * 4, 128 * 2, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            NoiseLayer2D(128 * 2, 128 * 1, 0.1),
        )

        self.l2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
        )
        self.l3 = nn.Sequential(
            NoiseLayer2D(128 * 1, channels, 0.1),
        )
        self.l4 = nn.Sequential(
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        x1 = self.l1(x.view(-1, 128 * 8, 4, 4))
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        img = self.l3(x3)
        return img, x1, x2, x3

class NoiseGenerator2Dv2(nn.Module):
    def __init__(self, opt):
        super(NoiseGenerator2Dv2, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 2 * 2)

        self.model = nn.Sequential(
            NoiseLayer2D(128 * 8, 128 * 4, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(2, 2) -> (4, 4)
            NoiseLayer2D(128 * 4, 128 * 2, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            NoiseLayer2D(128 * 2, 128 * 2, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            NoiseLayer2D(128 * 2, 128 * 1, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            NoiseLayer2D(128 * 1, channels, 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 2, 2))
        return img

class NoiseGenerator2Dv3(nn.Module):
    def __init__(self, opt):
        super(NoiseGenerator2Dv3, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 1 * 1)

        self.model = nn.Sequential(
            NoiseLayer2D(128 * 8, 128 * 4, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(1, 1) -> (2, 2)
            NoiseLayer2D(128 * 4, 128 * 4, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(2, 2) -> (4, 4)
            NoiseLayer2D(128 * 4, 128 * 2, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            NoiseLayer2D(128 * 2, 128 * 2, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            NoiseLayer2D(128 * 2, 128 * 1, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            NoiseLayer2D(128 * 1, channels, 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 1, 1))
        return img

class NoiseGenerator2Dv4(nn.Module):
    def __init__(self, opt):
        super(NoiseGenerator2Dv4, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 1 * 1)

        self.model = nn.Sequential(
            NoiseLayer2D(128 * 8, 128 * 4, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(1, 1) -> (2, 2)
            NoiseLayer2D(128 * 4, 128 * 4, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(2, 2) -> (4, 4)
            NoiseLayer2D(128 * 4, 128 * 2, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            NoiseLayer2D(128 * 2, 128 * 2, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            NoiseLayer2D(128 * 2, 128 * 1, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            NoiseLayer2D(128 * 1, 128 * 1, 0.1),
            NoiseLayer2D(128 * 1, channels, 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 1, 1))
        return img

class NoiseGenerator2Dv5(nn.Module):
    def __init__(self, opt):
        super(NoiseGenerator2Dv5, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.model = nn.Sequential(
            NoiseBasicBlock2D(opt.latent_dim, opt.latent_dim, level=0.1),
            NoiseLayer2D(opt.latent_dim, 128 * 8, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(1, 1) -> (2, 2)
            NoiseBasicBlock2D(128 * 8, 128 * 8, level=0.1),
            NoiseLayer2D(128 * 8, 128 * 6, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(2, 2) -> (4, 4)
            NoiseBasicBlock2D(128 * 6, 128 * 6, level=0.1),
            NoiseLayer2D(128 * 6, 128 * 4, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            NoiseBasicBlock2D(128 * 4, 128 * 4, level=0.1),
            NoiseLayer2D(128 * 4, 128 * 2, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            NoiseBasicBlock2D(128 * 2, 128 * 2, level=0.1),
            NoiseLayer2D(128 * 2, 128 * 1, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            NoiseBasicBlock2D(128 * 1, 128 * 1, level=0.1),
            NoiseLayer2D(128 * 1, channels, 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z.view(-1, z.size(1), 1, 1))
        return img

class NoiseGenerator2Dv6(nn.Module):
    def __init__(self, opt, seed=None):
        super(NoiseGenerator2Dv6, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            NoiseBasicBlock2D(128 * 8, 128 * 8, level=0.1, seed=seed),
            NoiseLayer2D(128 * 8, 128 * 4, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            NoiseBasicBlock2D(128 * 4, 128 * 4, level=0.1, seed=seed),
            NoiseLayer2D(128 * 4, 128 * 2, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            NoiseBasicBlock2D(128 * 2, 128 * 2, level=0.1, seed=seed),
            NoiseLayer2D(128 * 2, 128 * 1, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            NoiseBasicBlock2D(128 * 1, 128 * 1, level=0.1, seed=seed),
            NoiseLayer2D(128 * 1, channels, 0.1, seed=seed),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class NoiseGenerator2Dv7(nn.Module):
    def __init__(self, opt):
        super(NoiseGenerator2Dv7, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 2 * 2)

        self.model = nn.Sequential(
            NoiseBasicBlock2D(128 * 8, 128 * 8, level=0.1),
            NoiseLayer2D(128 * 8, 128 * 6, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(2, 2) -> (4, 4)
            NoiseBasicBlock2D(128 * 6, 128 * 6, level=0.1),
            NoiseLayer2D(128 * 6, 128 * 4, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            NoiseBasicBlock2D(128 * 4, 128 * 4, level=0.1),
            NoiseLayer2D(128 * 4, 128 * 2, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            NoiseBasicBlock2D(128 * 2, 128 * 2, level=0.1),
            NoiseLayer2D(128 * 2, 128 * 1, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            NoiseBasicBlock2D(128 * 1, 128 * 1, level=0.1),
            NoiseLayer2D(128 * 1, channels, 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 2, 2))
        return img

class NoiseGenerator2Dv8(nn.Module):
    def __init__(self, opt):
        super(NoiseGenerator2Dv8, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 1 * 1)

        self.model = nn.Sequential(
            NoiseBasicBlock2D(128 * 8, 128 * 8, level=0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(1, 1) -> (2, 2)
            NoiseBasicBlock2D(128 * 8, 128 * 8, level=0.1),
            NoiseLayer2D(128 * 8, 128 * 6, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(2, 2) -> (4, 4)
            NoiseBasicBlock2D(128 * 6, 128 * 6, level=0.1),
            NoiseLayer2D(128 * 6, 128 * 4, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            NoiseBasicBlock2D(128 * 4, 128 * 4, level=0.1),
            NoiseLayer2D(128 * 4, 128 * 2, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            NoiseBasicBlock2D(128 * 2, 128 * 2, level=0.1),
            NoiseLayer2D(128 * 2, 128 * 1, 0.1),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            NoiseBasicBlock2D(128 * 1, 128 * 1, level=0.1),
            NoiseLayer2D(128 * 1, channels, 0.1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 1, 1))
        return img

class MTNoiseGenerator2Dv6(nn.Module):
    def __init__(self, opt, seed):
        super(MTNoiseGenerator2Dv6, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            MTNoiseBasicBlock2D(128 * 8, 128 * 8, level=0.1, seed=seed+0),
            MTNoiseLayer2D(128 * 8, 128 * 4, 0.1, seed=seed+10),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            MTNoiseBasicBlock2D(128 * 4, 128 * 4, level=0.1, seed=seed+20),
            MTNoiseLayer2D(128 * 4, 128 * 2, 0.1, seed=seed+30),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            MTNoiseBasicBlock2D(128 * 2, 128 * 2, level=0.1, seed=seed+40),
            MTNoiseLayer2D(128 * 2, 128 * 1, 0.1, seed=seed+50),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            MTNoiseBasicBlock2D(128 * 1, 128 * 1, level=0.1, seed=seed+60),
            MTNoiseLayer2D(128 * 1, channels, 0.1, seed=seed+70),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class MTNoiseGenerator2Dv6_512(nn.Module):
    def __init__(self, opt, seed=None):
        super(MTNoiseGenerator2Dv6_512, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            MTNoiseLayer2D(128 * 8, 128 * 8, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            MTNoiseLayer2D(128 * 8, 128 * 4, 0.1, seed=seed+10),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            MTNoiseLayer2D(128 * 4, 128 * 2, 0.1, seed=seed+20),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            MTNoiseLayer2D(128 * 2, 128 * 1, 0.1, seed=seed+30),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(32, 32) -> (64, 64)
            MTNoiseLayer2D(128 * 1, 128 * 1, 0.1, seed=seed+40),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(64, 64) -> (128, 128)
            MTNoiseLayer2D(128 * 1, 128 * 1, 0.1, seed=seed+50),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(128, 128) -> (256, 256)
            MTNoiseLayer2D(128 * 1, 128 * 1, 0.1, seed=seed+60),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(256, 256) -> (512, 512)
            MTNoiseLayer2D(128 * 1, channels, 0.1, seed=seed+70),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class MTNoiseGenerator2Dv6SND512(nn.Module):
    def __init__(self, opt, seed=None):
        super(MTNoiseGenerator2Dv6SND512, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            MTSNDNoiseLayer2D(128 * 8, 128 * 8, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            MTSNDNoiseLayer2D(128 * 8, 128 * 4, 0.1, seed=seed+10),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            MTSNDNoiseLayer2D(128 * 4, 128 * 2, 0.1, seed=seed+20),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            MTSNDNoiseLayer2D(128 * 2, 128 * 1, 0.1, seed=seed+30),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(32, 32) -> (64, 64)
            MTSNDNoiseLayer2D(128 * 1, 128 * 1, 0.1, seed=seed+40),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(64, 64) -> (128, 128)
            MTSNDNoiseLayer2D(128 * 1, 128 * 1, 0.1, seed=seed+50),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(128, 128) -> (256, 256)
            MTSNDNoiseLayer2D(128 * 1, 128 * 1, 0.1, seed=seed+60),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(256, 256) -> (512, 512)
            MTSNDNoiseLayer2D(128 * 1, channels, 0.1, seed=seed+70),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class MTNoiseGenerator2Dv6SND1024(nn.Module):
    def __init__(self, opt, seed=None):
        super(MTNoiseGenerator2Dv6SND1024, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            MTSNDNoiseLayer2D(128 * 8, 128 * 8, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            MTSNDNoiseLayer2D(128 * 8, 128 * 4, 0.1, seed=seed+10),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            MTSNDNoiseLayer2D(128 * 4, 128 * 2, 0.1, seed=seed+20),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            MTSNDNoiseLayer2D(128 * 2, 128 * 1, 0.1, seed=seed+30),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(32, 32) -> (64, 64)
            MTSNDNoiseLayer2D(128 * 1, 128 * 1, 0.1, seed=seed+40),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(64, 64) -> (128, 128)
            MTSNDNoiseLayer2D(128 * 1, 128 * 1, 0.1, seed=seed+50),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(128, 128) -> (256, 256)
            MTSNDNoiseLayer2D(128 * 1, 128 * 1, 0.1, seed=seed+60),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(256, 256) -> (512, 512)
            MTSNDNoiseLayer2D(128 * 1, 128 * 1, 0.1, seed=seed+70),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(512, 512) -> (1024), 1024)
            MTSNDNoiseLayer2D(128 * 1, channels, 0.1, seed=seed+90),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class MTNoiseGenerator2Dv6SND1024_x4(nn.Module):
    def __init__(self, opt, seed=None):
        super(MTNoiseGenerator2Dv6SND1024_x4, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            MTSNDNoiseLayer2D_x4(128 * 8, 128 * 8, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            MTSNDNoiseLayer2D_x4(128 * 8, 128 * 4, 0.1, seed=seed+10),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            MTSNDNoiseLayer2D_x4(128 * 4, 128 * 2, 0.1, seed=seed+20),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            MTSNDNoiseLayer2D_x4(128 * 2, 128 * 1, 0.1, seed=seed+30),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(32, 32) -> (64, 64)
            MTSNDNoiseLayer2D_x4(128 * 1, 128 * 1, 0.1, seed=seed+40),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(64, 64) -> (128, 128)
            MTSNDNoiseLayer2D_x4(128 * 1, 128 * 1, 0.1, seed=seed+50),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(128, 128) -> (256, 256)
            MTSNDNoiseLayer2D_x4(128 * 1, 128 * 1, 0.1, seed=seed+60),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(256, 256) -> (512, 512)
            MTSNDNoiseLayer2D_x4(128 * 1, 128 * 1, 0.1, seed=seed+70),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(512, 512) -> (1024), 1024)
            MTSNDNoiseLayer2D_x4(128 * 1, channels, 0.1, seed=seed+90),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class MTNoiseGenerator2Dv6SND1024_x4v2(nn.Module):
    def __init__(self, opt, seed=None):
        super(MTNoiseGenerator2Dv6SND1024_x4v2, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            MTSNDNoiseLayer2D_x4(128 * 8, 128 * 8, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            MTSNDNoiseLayer2D_x4(128 * 8, 128 * 8, 0.1, seed=seed+10),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            MTSNDNoiseLayer2D_x4(128 * 8, 128 * 4, 0.1, seed=seed+20),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            MTSNDNoiseLayer2D_x4(128 * 4, 128 * 4, 0.1, seed=seed+30),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(32, 32) -> (64, 64)
            MTSNDNoiseLayer2D_x4(128 * 4, 128 * 4, 0.1, seed=seed+40),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(64, 64) -> (128, 128)
            MTSNDNoiseLayer2D_x4(128 * 4, 128 * 2, 0.1, seed=seed+50),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(128, 128) -> (256, 256)
            MTSNDNoiseLayer2D_x4(128 * 2, 128 * 2, 0.1, seed=seed+60),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(256, 256) -> (512, 512)
            MTSNDNoiseLayer2D_x4(128 * 2, 128 * 1, 0.1, seed=seed+70),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(512, 512) -> (1024), 1024)
            MTSNDNoiseLayer2D_x4(128 * 1, channels, 0.1, seed=seed+90),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class MTNoiseGenerator2Dv6SND1024_x4v3(nn.Module):
    def __init__(self, opt, seed=None):
        super(MTNoiseGenerator2Dv6SND1024_x4v3, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            MTSNDNoiseLayer2D_x4(128 * 8, 128 * 8, level=0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            MTSNDNoiseLayer2D_x4(128 * 8, 128 * 6, level=0.1, seed=seed+10),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            MTSNDNoiseLayer2D_x4(128 * 6, 128 * 6, level=0.1, seed=seed+20),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            MTSNDNoiseLayer2D_x4(128 * 6, 128 * 4, level=0.1, seed=seed+30),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(32, 32) -> (64, 64)
            MTSNDNoiseLayer2D_x4(128 * 4, 128 * 4, level=0.1, seed=seed+40),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(64, 64) -> (128, 128)
            MTSNDNoiseLayer2D_x4(128 * 4, 128 * 2, level=0.1, seed=seed+50),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(128, 128) -> (256, 256)
            MTSNDNoiseLayer2D_x4(128 * 2, 128 * 2, level=0.1, seed=seed+60),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(256, 256) -> (512, 512)
            MTSNDNoiseLayer2D_x4(128 * 2, 128 * 1, level=0.1, seed=seed+70),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(512, 512) -> (1024), 1024)
            MTSNDNoiseLayer2D_x4(128 * 1, 128 * 1, level=0.1, seed=seed+80),
            MTSNDNoiseLayer2D_x4(128 * 1, channels, level=0.1, seed=seed+90),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class MTNoiseGenerator2Dv6SND1024_x4v4(nn.Module):
    def __init__(self, opt, seed=None):
        super(MTNoiseGenerator2Dv6SND1024_x4v4, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            MTSNDNoiseLayer2D_x4(128 * 8, 128 * 8, level=0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            MTSNDNoiseLayer2D_x4(128 * 8, 128 * 6, level=0.1, seed=seed+10),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            MTSNDNoiseLayer2D_x4(128 * 6, 128 * 6, level=0.1, seed=seed+20),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            MTSNDNoiseLayer2D_x4(128 * 6, 128 * 4, level=0.1, seed=seed+30),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(32, 32) -> (64, 64)
            MTSNDNoiseLayer2D_x4(128 * 4, 128 * 4, level=0.1, seed=seed+40),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(64, 64) -> (128, 128)
            MTSNDNoiseLayer2D_x4(128 * 4, 128 * 2, level=0.1, seed=seed+50),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(128, 128) -> (256, 256)
            MTSNDNoiseLayer2D_x4(128 * 2, 128 * 2, level=0.1, seed=seed+60),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(256, 256) -> (512, 512)
            MTSNDNoiseLayer2D_x4(128 * 2, 128 * 1, level=0.1, seed=seed+70),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(512, 512) -> (1024), 1024)
            MTSNDNoiseLayer2D_x4(128 * 1, channels, level=0.1, seed=seed+90),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class MTNoiseGenerator2Dv6SND2048x(nn.Module):
    def __init__(self, opt, seed=None):
        super(MTNoiseGenerator2Dv6SND2048x, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            MTSNDNoiseLayer2Dx(128 * 8, 128 * 8, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            MTSNDNoiseLayer2Dx(128 * 8, 128 * 8, 0.1, seed=seed+10),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            MTSNDNoiseLayer2Dx(128 * 8, 128 * 4, 0.1, seed=seed+20),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            MTSNDNoiseLayer2Dx(128 * 4, 128 * 4, 0.1, seed=seed+30),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(32, 32) -> (64, 64)
            MTSNDNoiseLayer2Dx(128 * 4, 128 * 4, 0.1, seed=seed+40),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(64, 64) -> (128, 128)
            MTSNDNoiseLayer2Dx(128 * 4, 128 * 2, 0.1, seed=seed+50),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(128, 128) -> (256, 256)
            MTSNDNoiseLayer2Dx(128 * 2, 128 * 2, 0.1, seed=seed+60),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(256, 256) -> (512, 512)
            MTSNDNoiseLayer2Dx(128 * 2, 128 * 1, 0.1, seed=seed+70),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(512, 512) -> (1024), 1024)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed+90),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(1024, 1024) -> (2048), 2048)
            MTSNDNoiseLayer2Dx(128 * 1, channels, 0.1, seed=seed+110),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class LCGNoiseGenerator2Dv6(nn.Module):
    def __init__(self, opt):
        super(LCGNoiseGenerator2Dv6, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            LCGNoiseBasicBlock2D(128 * 8, 128 * 8, level=0.1, seed=0),
            LCGNoiseLayer2D(128 * 8, 128 * 4, 0.1, seed=10),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            LCGNoiseBasicBlock2D(128 * 4, 128 * 4, level=0.1, seed=20),
            LCGNoiseLayer2D(128 * 4, 128 * 2, 0.1, seed=30),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            LCGNoiseBasicBlock2D(128 * 2, 128 * 2, level=0.1, seed=40),
            LCGNoiseLayer2D(128 * 2, 128 * 1, 0.1, seed=50),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            LCGNoiseBasicBlock2D(128 * 1, 128 * 1, level=0.1, seed=60),
            LCGNoiseLayer2D(128 * 1, channels, 0.1, seed=70),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class LCGNoiseGenerator2Dv6_(nn.Module):
    def __init__(self, opt, seed):
        super(LCGNoiseGenerator2Dv6_, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            LCGNoiseBasicBlock2D_(128 * 8, 128 * 8, level=0.1, seed=seed, size=[128*8, 4, 4]),
            LCGNoiseLayer2D_(128 * 8, 128 * 4, 0.1, seed=seed+10, size=[128*8, 4, 4]),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            LCGNoiseBasicBlock2D_(128 * 4, 128 * 4, level=0.1, seed=seed+20, size=[128*4, 8, 8]),
            LCGNoiseLayer2D_(128 * 4, 128 * 2, 0.1, seed=seed+30, size=[128*4, 8, 8]),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            LCGNoiseBasicBlock2D_(128 * 2, 128 * 2, level=0.1, seed=seed+40, size=[128*2, 16, 16]),
            LCGNoiseLayer2D_(128 * 2, 128 * 1, 0.1, seed=seed+50, size=[128*2, 16, 16]),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            LCGNoiseBasicBlock2D_(128 * 1, 128 * 1, level=0.1, seed=seed+60, size=[128*1, 32, 32]),
            LCGNoiseLayer2D_(128 * 1, channels, 0.1, seed=seed+70, size=[128*1, 32, 32]),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class LCGNoiseGenerator2Dv6_512(nn.Module):
    def __init__(self, opt, seed):
        super(LCGNoiseGenerator2Dv6_, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            LCGNoiseBasicBlock2D_(128 * 8, 128 * 8, level=0.1, seed=seed, size=[128*8, 4, 4]),
            LCGNoiseLayer2D_(128 * 8, 128 * 4, 0.1, seed=seed+10, size=[128*8, 4, 4]),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            LCGNoiseBasicBlock2D_(128 * 4, 128 * 4, level=0.1, seed=seed+20, size=[128*4, 8, 8]),
            LCGNoiseLayer2D_(128 * 4, 128 * 2, 0.1, seed=seed+30, size=[128*4, 8, 8]),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            LCGNoiseBasicBlock2D_(128 * 2, 128 * 2, level=0.1, seed=seed+40, size=[128*2, 16, 16]),
            LCGNoiseLayer2D_(128 * 2, 128 * 1, 0.1, seed=seed+50, size=[128*2, 16, 16]),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            LCGNoiseBasicBlock2D_(128 * 1, 128 * 1, level=0.1, seed=seed+60, size=[128*1, 32, 32]),
            LCGNoiseLayer2D_(128 * 1, 128 * 1, 0.1, seed=seed+70, size=[128*1, 32, 32]),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(32, 32) -> (64, 64)
            LCGNoiseBasicBlock2D_(128 * 1, 128 * 1, level=0.1, seed=seed+60, size=[128*1, 64, 64]),
            LCGNoiseLayer2D_(128 * 1, 128 * 1, 0.1, seed=seed+90, size=[128*1, 64, 64]),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(64, 64) -> (128, 128)
            LCGNoiseBasicBlock2D_(128 * 1, 128 * 1, level=0.1, seed=seed+60, size=[128*1, 128, 128]),
            LCGNoiseLayer2D_(128 * 1, 128 * 1, 0.1, seed=seed+110, size=[128*1, 128, 128]),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(128, 128) -> (256, 256)
            LCGNoiseBasicBlock2D_(128 * 1, 128 * 1, level=0.1, seed=seed+60, size=[128*1, 256, 256]),
            LCGNoiseLayer2D_(128 * 1, 128 * 1, 0.1, seed=seed+130, size=[128*1, 256, 256]),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(256, 256) -> (512, 512)
            LCGNoiseBasicBlock2D_(128 * 1, 128 * 1, level=0.1, seed=seed+60, size=[128*1, 512, 512]),
            LCGNoiseLayer2D_(128 * 1, channels, 0.1, seed=seed+150, size=[128*1, 512, 512]),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class NoiseResGenerator2Dv1(nn.Module):
    def __init__(self, opt, block, nblocks, level):
        super(NoiseResGenerator2Dv1, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        nfilters = opt.num_filters
        self.in_planes = 8*nfilters
        self.img_shape = (channels, opt.img_size, opt.img_size)
        self.pre_layer = nn.Linear(opt.latent_dim, 8*nfilters * 4 * 4)
        self.layer1 = self._make_layer(block, 8*nfilters, nblocks[0], level=level)
        self.layer2 = self._make_layer(block, 4*nfilters, nblocks[1], scale=2, level=level)
        self.layer3 = self._make_layer(block, 2*nfilters, nblocks[2], scale=2, level=level)
        self.layer4 = self._make_layer(block, 1*nfilters, nblocks[3], scale=2, level=level)
        self.layer5 = self._make_layer(block, channels, nblocks[4], scale=1, level=level)
        self.tanh = nn.Tanh()

    def _make_layer(self, block, planes, nblocks, scale=1, level=0.2):
        shortcut = None
        if scale != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                NoiseLayer2D(self.in_planes, planes * block.expansion, level=level),
                nn.Upsample(scale_factor=scale, mode='bilinear'),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, scale, shortcut, level=level))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, level=level))
        return nn.Sequential(*layers)

    def forward(self, z):
        x1 = self.pre_layer(z)                          #(128) -> (128*8, 4, 4)
        x2 = self.layer1(x1.view(-1, 128 * 8, 4, 4))    #(128*8, 4, 4) -> (128*8, 4, 4)
        x3 = self.layer2(x2)                            #(128*8, 4, 4) -> (128*4, 8, 8)
        x4 = self.layer3(x3)                            #(128*4, 8, 8) -> (128*2, 16, 16)
        x5 = self.layer4(x4)                            #(128*2, 16, 16) -> (128*1, 32, 32)
        x6 = self.layer5(x5)                            #(128*1, 32, 32) -> (nc, 32, 32)
        return self.tanh(x6)

class NoiseResGenerator2D1024(nn.Module):
    def __init__(self, opt, block, nblock, level):
        super(NoiseResGenerator2D1024, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        nfilters = opt.num_filters
        self.in_planes = 8*nfilters
        self.img_shape = (channels, opt.img_size, opt.img_size)
        self.pre_layer = nn.Linear(opt.latent_dim, 8*nfilters * 4 * 4)
        self.layer1 = self._make_layer(block, 8*nfilters, nblock, level=level)
        self.layer2 = self._make_layer(block, 8*nfilters, nblock, scale=2, level=level)
        self.layer3 = self._make_layer(block, 6*nfilters, nblock, scale=2, level=level)
        self.layer4 = self._make_layer(block, 6*nfilters, nblock, scale=2, level=level)
        self.layer5 = self._make_layer(block, 4*nfilters, nblock, scale=2, level=level)
        self.layer6 = self._make_layer(block, 4*nfilters, nblock, scale=2, level=level)
        self.layer7 = self._make_layer(block, 2*nfilters, nblock, scale=2, level=level)
        self.layer8 = self._make_layer(block, 2*nfilters, nblock, scale=2, level=level)
        self.layer9 = self._make_layer(block, 1*nfilters, nblock, scale=2, level=level)
        self.layer10 = self._make_layer(block, channels, 1, scale=1, level=level)
        self.tanh = nn.Tanh()

    def _make_layer(self, block, planes, nblocks, scale=1, level=0.2):
        shortcut = None
        if scale != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                NoiseLayer2D(self.in_planes, planes * block.expansion, level=level),
                nn.Upsample(scale_factor=scale, mode='bilinear'),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, scale, shortcut, level=level))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, level=level))
        return nn.Sequential(*layers)

    def forward(self, z):
        x1 = self.pre_layer(z)                          #(1) -> (4)
        x2 = self.layer1(x1.view(-1, 128 * 8, 4, 4))
        x3 = self.layer2(x2)                            #(4) -> (8)
        x4 = self.layer3(x3)                            #(8) -> (16)
        x5 = self.layer4(x4)                            #(16) -> (32)
        x6 = self.layer5(x5)                            #(32) -> (64)
        x7 = self.layer6(x6)                            #(64) -> (128)
        x8 = self.layer7(x7)                            #(128) -> (256)
        x9 = self.layer8(x8)                            #(256) -> (512)
        x10 = self.layer9(x9)                            #(512) -> (1024)
        x11 = self.layer10(x10)                            #(512) -> (1024)
        return self.tanh(x11)

class NoiseGenerator2Dv6_512(nn.Module):
    def __init__(self, opt, seed=None):
        super(NoiseGenerator2Dv6_512, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            NoiseLayer2D(128 * 8, 128 * 8, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            NoiseLayer2D(128 * 8, 128 * 4, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            NoiseLayer2D(128 * 4, 128 * 2, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            NoiseLayer2D(128 * 2, 128 * 1, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(32, 32) -> (64, 64)
            NoiseLayer2D(128 * 1, 128 * 1, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(64, 64) -> (128, 128)
            NoiseLayer2D(128 * 1, 128 * 1, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(128, 128) -> (256, 256)
            NoiseLayer2D(128 * 1, 128 * 1, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(256, 256) -> (512, 512)
            NoiseLayer2D(128 * 1, channels, 0.1, seed=seed),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class NoiseGenerator2Dv7_512(nn.Module):
    def __init__(self, opt, seed=None):
        super(NoiseGenerator2Dv7_512, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            MTSNDNoiseLayer2Dx(128 * 8, 128 * 8, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            NMTSNDNoiseLayer2Dx(128 * 8, 128 * 4, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            MTSNDNoiseLayer2Dx(128 * 4, 128 * 2, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            MTSNDNoiseLayer2Dx(128 * 2, 128 * 1, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(32, 32) -> (64, 64)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(64, 64) -> (128, 128)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(128, 128) -> (256, 256)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(256, 256) -> (512, 512)
            MTSNDNoiseLayer2Dx(128 * 1, channels, 0.1, seed=seed),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class NoiseConvGenerator2D_512(nn.Module):
    def __init__(self, opt, seed=None):
        super(NoiseConvGenerator2D_512, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            MTSNDNoiseLayer2Dx(128 * 8, 128 * 8, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            MTSNDNoiseLayer2Dx(128 * 8, 128 * 4, 0.1, seed=seed+10),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            MTSNDNoiseLayer2Dx(128 * 4, 128 * 2, 0.1, seed=seed+20),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            MTSNDNoiseLayer2Dx(128 * 2, 128 * 1, 0.1, seed=seed+30),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(32, 32) -> (64, 64)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed+40),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(64, 64) -> (128, 128)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed+50),
            TransConvLayer(128 * 1, 128 * 1),             #(128, 128) -> (256, 256)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed+60),
            TransConvLayer(128 * 1, 128 * 1),             #(256, 256) -> (512, 512)
            MTSNDNoiseLayer2Dx(128 * 1, channels, 0.1, seed=seed+70),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class NoiseConvGenerator2Dv2_512(nn.Module):
    def __init__(self, opt, seed=None):
        super(NoiseConvGenerator2Dv2_512, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            MTSNDNoiseLayer2Dx(128 * 8, 128 * 8, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            MTSNDNoiseLayer2Dx(128 * 8, 128 * 4, 0.1, seed=seed+10),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            MTSNDNoiseLayer2Dx(128 * 4, 128 * 2, 0.1, seed=seed+20),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            MTSNDNoiseLayer2Dx(128 * 2, 128 * 1, 0.1, seed=seed+30),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(32, 32) -> (64, 64)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed+40),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(64, 64) -> (128, 128)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed+50),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(128, 128) -> (256, 256)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed+60),
            TransConvLayer(128 * 1, 128 * 1),             #(256, 256) -> (512, 512)
            MTSNDNoiseLayer2Dx(128 * 1, channels, 0.1, seed=seed+70),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class NoiseConvGenerator2Dv3_512(nn.Module):
    def __init__(self, opt, seed=None):
        super(NoiseConvGenerator2Dv3_512, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            MTSNDNoiseLayer2Dx(128 * 8, 128 * 8, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            MTSNDNoiseLayer2Dx(128 * 8, 128 * 4, 0.1, seed=seed+10),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            MTSNDNoiseLayer2Dx(128 * 4, 128 * 2, 0.1, seed=seed+20),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            MTSNDNoiseLayer2Dx(128 * 2, 128 * 1, 0.1, seed=seed+30),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(32, 32) -> (64, 64)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed+40),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(64, 64) -> (128, 128)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed+50),
            TransConvLayer(128 * 1, 128 * 1),             #(128, 128) -> (256, 256)
            TransConvLayer(128 * 1, 128 * 1),             #(256, 256) -> (512, 512)
            MTSNDNoiseLayer2Dx(128 * 1, channels, 0.1, seed=seed+70),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class NoiseConvGenerator2Dv4_512(nn.Module):
    def __init__(self, opt, seed=None):
        super(NoiseConvGenerator2Dv4_512, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            MTSNDNoiseLayer2Dx(128 * 8, 128 * 8, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            MTSNDNoiseLayer2Dx(128 * 8, 128 * 4, 0.1, seed=seed+10),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            MTSNDNoiseLayer2Dx(128 * 4, 128 * 2, 0.1, seed=seed+20),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            MTSNDNoiseLayer2Dx(128 * 2, 128 * 1, 0.1, seed=seed+30),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(32, 32) -> (64, 64)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed+40),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(64, 64) -> (128, 128)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed+50),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(128, 128) -> (256, 256)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed+60),
            ConvUpsample(128 * 1, 128 * 1),             #(256, 256) -> (512, 512)
            MTSNDNoiseLayer2Dx(128 * 1, channels, 0.1, seed=seed+70),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class NoiseConvGenerator2Dv5_512(nn.Module):
    def __init__(self, opt, seed=None):
        super(NoiseConvGenerator2Dv5_512, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            MTSNDNoiseLayer2Dx(128 * 8, 128 * 8, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            MTSNDNoiseLayer2Dx(128 * 8, 128 * 4, 0.1, seed=seed+10),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            MTSNDNoiseLayer2Dx(128 * 4, 128 * 2, 0.1, seed=seed+20),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            MTSNDNoiseLayer2Dx(128 * 2, 128 * 1, 0.1, seed=seed+30),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(32, 32) -> (64, 64)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed+40),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(64, 64) -> (128, 128)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed+50),
            ConvUpsample(128 * 1, 128 * 1),  #(128, 128) -> (256, 256)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed+60),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(256, 256) -> (512, 512)
            MTSNDNoiseLayer2Dx(128 * 1, channels, 0.1, seed=seed+70),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class NoiseConvGenerator2Dv6_512(nn.Module):
    def __init__(self, opt, seed=None):
        super(NoiseConvGenerator2Dv6_512, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            MTSNDNoiseLayer2Dx(128 * 8, 128 * 8, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            MTSNDNoiseLayer2Dx(128 * 8, 128 * 4, 0.1, seed=seed+10),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(8, 8) -> (16, 16)
            MTSNDNoiseLayer2Dx(128 * 4, 128 * 2, 0.1, seed=seed+20),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            MTSNDNoiseLayer2Dx(128 * 2, 128 * 1, 0.1, seed=seed+30),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(32, 32) -> (64, 64)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed+40),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(64, 64) -> (128, 128)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed+50),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(128, 128) -> (256, 256)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed+60),
            ConvUpsample(128 * 1, 128 * 1),             #(256, 256) -> (512, 512)
            MTSNDNoiseLayer2Dx(128 * 1, channels, 0.1, seed=seed+70),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class NoiseConvGenerator2Dv7_512(nn.Module):
    def __init__(self, opt, seed=None):
        super(NoiseConvGenerator2Dv7_512, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            MTSNDNoiseLayer2Dx(128 * 8, 128 * 8, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            MTSNDNoiseLayer2Dx(128 * 8, 128 * 4, 0.1, seed=seed+10),
            ConvUpsample(128 * 4, 128 * 4),  #(8, 8) -> (16, 16)
            MTSNDNoiseLayer2Dx(128 * 4, 128 * 2, 0.1, seed=seed+20),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            MTSNDNoiseLayer2Dx(128 * 2, 128 * 1, 0.1, seed=seed+30),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(32, 32) -> (64, 64)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed+40),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(64, 64) -> (128, 128)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed+50),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(128, 128) -> (256, 256)
            MTSNDNoiseLayer2Dx(128 * 1, 128 * 1, 0.1, seed=seed+60),
            ConvUpsample(128 * 1, 128 * 1),             #(256, 256) -> (512, 512)
            MTSNDNoiseLayer2Dx(128 * 1, channels, 0.1, seed=seed+70),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class NoiseConvGenerator2Dv8_512(nn.Module):
    def __init__(self, opt, seed=None):
        super(NoiseConvGenerator2Dv8_512, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            MTSNDNoiseLayer2Dx(128 * 8, 128 * 8, 0.1, seed=seed),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(4, 4) -> (8, 8)
            MTSNDNoiseLayer2Dx(128 * 8, 128 * 4, 0.1, seed=seed+10),
            ConvUpsample(128 * 4, 128 * 4),  #(8, 8) -> (16, 16)
            MTSNDNoiseLayer2Dx(128 * 4, 128 * 2, 0.1, seed=seed+20),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            MTSNDNoiseLayer2Dx(128 * 2, 128 * 1, 0.1, seed=seed+30),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(32, 32) -> (64, 64)
            MTSNDNoiseBasicBlock2Dx(128 * 1, 128 * 1, 0.1, seed=seed+40),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(64, 64) -> (128, 128)
            MTSNDNoiseBasicBlock2Dx(128 * 1, 128 * 1, 0.1, seed=seed+50),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(128, 128) -> (256, 256)
            MTSNDNoiseBasicBlock2Dx(128 * 1, 128 * 1, 0.1, seed=seed+60),
            ConvUpsample(128 * 1, 128 * 1),             #(256, 256) -> (512, 512)
            MTSNDNoiseLayer2Dx(128 * 1, channels, 0.1, seed=seed+70),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class NoiseConvGenerator2Dv9_512(nn.Module):
    def __init__(self, opt, seed=None):
        super(NoiseConvGenerator2Dv9_512, self).__init__()
        channels = 1 if opt.dataset == 'mnist' or opt.dataset == 'fashion' else 3
        self.img_shape = (channels, opt.img_size, opt.img_size)

        self.pre_layer = nn.Linear(opt.latent_dim, 128 * 8 * 4 * 4)

        self.model = nn.Sequential(
            MTSNDNoiseLayer2Dx(128 * 8, 128 * 8, 0.1, seed=seed),
            ConvUpsample(128 * 8, 128 * 8), #(4, 4) -> (8, 8)
            MTSNDNoiseLayer2Dx(128 * 8, 128 * 4, 0.1, seed=seed+10),
            nn.Upsample(scale_factor=2, mode='bilinear'),  #(8, 8) -> (16, 16)
            MTSNDNoiseLayer2Dx(128 * 4, 128 * 2, 0.1, seed=seed+20),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(16, 16) -> (32, 32)
            MTSNDNoiseLayer2Dx(128 * 2, 128 * 1, 0.1, seed=seed+30),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(32, 32) -> (64, 64)
            MTSNDNoiseBasicBlock2Dx(128 * 1, 128 * 1, 0.1, seed=seed+40),
            ConvUpsample(128 * 1, 128 * 1), #(64, 64) -> (128, 128)
            MTSNDNoiseBasicBlock2Dx(128 * 1, 128 * 1, 0.1, seed=seed+50),
            nn.Upsample(scale_factor=2, mode='bilinear'), #(128, 128) -> (256, 256)
            MTSNDNoiseBasicBlock2Dx(128 * 1, 128 * 1, 0.1, seed=seed+60),
            ConvUpsample(128 * 1, 128 * 1),             #(256, 256) -> (512, 512)
            MTSNDNoiseLayer2Dx(128 * 1, channels, 0.1, seed=seed+70),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.pre_layer(z)
        img = self.model(x.view(-1, 128 * 8, 4, 4))
        return img

class NoiseGenerator(nn.Module):
    def __init__(self, opt):
        super(NoiseGenerator, self).__init__()
        if opt.dataset == 'mnist' or opt.dataset == 'fashion':
          channels = 1
        else:
          channels = 3
        self.img_shape = (channels, opt.img_size, opt.img_size)
        def block(in_feat, out_feat, level, normalize=True):
            layers = [NoiseLayer(in_feat, out_feat, level, normalize)]
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, 0.1, normalize=False),
            *block(128, 256, 0.1),
            *block(256, 512, 0.1),
            *block(512, 1024, 0.1),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *self.img_shape)
        return img
