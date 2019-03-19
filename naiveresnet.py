# naiveresnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

import noise_layers


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class NoiseLayer(nn.Module):
    def __init__(self, in_planes, out_planes, level):
        super(NoiseLayer, self).__init__()
        self.noise = nn.Parameter(torch.Tensor(0), requires_grad=False).to(device)
        self.level = level
        self.layers = nn.Sequential(
            nn.ReLU(True),
            nn.BatchNorm2d(in_planes),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
        )

    def forward(self, x):
        if self.noise.numel() == 0:
            self.noise.resize_(x.data[0].shape).uniform_()
            self.noise = (2 * self.noise - 1) * self.level

        y = torch.add(x, self.noise)
        z = self.layers(y)
        return z

class NoiseBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, shortcut=None, level=0.2):
        super(NoiseBasicBlock, self).__init__()
        self.layers = nn.Sequential(
            NoiseLayer(in_planes, planes, level),
            nn.MaxPool2d(stride, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            NoiseLayer(planes, planes, level),
            nn.BatchNorm2d(planes),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y

class MTNoiseBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, shortcut=None, seed=0, level=0.2):
        super(MTNoiseBasicBlock, self).__init__()
        self.layers = nn.Sequential(
            noise_layers.MTNoiseLayer2D(in_planes, planes, level, seed),
            nn.MaxPool2d(stride, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            noise_layers.MTNoiseLayer2D(planes, planes, level, seed+1),
            nn.BatchNorm2d(planes),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y

class MTSNDNoiseBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, shortcut=None, seed=0, level=0.2):
        super(MTSNDNoiseBasicBlock, self).__init__()
        self.layers = nn.Sequential(
            noise_layers.MTSNDNoiseLayer2D(in_planes, planes, level, seed),
            nn.MaxPool2d(stride, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            noise_layers.MTSNDNoiseLayer2D(planes, planes, level, seed+1),
            nn.BatchNorm2d(planes),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y

class LCGNoiseBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, shortcut=None, seed=0, level=0.2):
        super(LCGNoiseBasicBlock, self).__init__()
        self.layers = nn.Sequential(
            noise_layers.LCGNoiseLayer2D(in_planes, planes, level, seed),
            nn.MaxPool2d(stride, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            noise_layers.LCGNoiseLayer2D(planes, planes, level, seed+1),
            nn.BatchNorm2d(planes),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y

class LCGNoiseBasicBlock_(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, shortcut=None, seed=0, level=0.2, size=[128, 1, 1]):
        super(LCGNoiseBasicBlock_, self).__init__()
        self.layers = nn.Sequential(
            noise_layers.LCGNoiseLayer2D_(in_planes, planes, level, seed, size),
            nn.MaxPool2d(stride, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            noise_layers.LCGNoiseLayer2D_(planes, planes, level, seed+1, size=[size[0]*stride, int(size[1]/stride), int(size[2]/stride)]),
            nn.BatchNorm2d(planes),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y

class ArgNoiseBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, shortcut=None, seed=0, level=0.2):
        super(ArgNoiseBasicBlock, self).__init__()
        self.layers = nn.Sequential(
            noise_layers.AlgorithmicNoiseLayer(in_planes, planes, seed, level),
            nn.MaxPool2d(stride, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            noise_layers.AlgorithmicNoiseLayer(planes, planes, seed+1, level),
            nn.BatchNorm2d(planes),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y

class NoiseBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, shortcut=None, level=0.2):
        super(NoiseBottleneck, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            NoiseLayer(planes, planes, level),
            nn.MaxPool2d(stride, stride),
            nn.BatchNorm2d(planes),
            nn.ReLU(True),
            nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes * 4),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = F.relu(y)
        return y

class NoiseResNet(nn.Module):
    def __init__(self, block, nblocks, nchannels, nfilters, nclasses, pool, level):
        super(NoiseResNet, self).__init__()
        self.in_planes = nfilters
        self.pre_layers = nn.Sequential(
            nn.Conv2d(nchannels,nfilters,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(nfilters),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.layer1 = self._make_layer(block, 1*nfilters, nblocks[0], level=level)
        self.layer2 = self._make_layer(block, 2*nfilters, nblocks[1], stride=2, level=level)
        self.layer3 = self._make_layer(block, 4*nfilters, nblocks[2], stride=2, level=level)
        self.layer4 = self._make_layer(block, 8*nfilters, nblocks[3], stride=2, level=level)
        self.avgpool = nn.AvgPool2d(pool, stride=1)
        self.linear = nn.Linear(8*nfilters*block.expansion, nclasses)

    def _make_layer(self, block, planes, nblocks, stride=1, level=0.2):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut, level=level))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, level=level))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.pre_layers(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.avgpool(x5)
        x7 = x6.view(x6.size(0), -1)
        x8 = self.linear(x7)
        return x8

class NoiseResNet32(nn.Module):
    def __init__(self, block, nblocks, nchannels, nfilters, nclasses, pool, level):
        super(NoiseResNet32, self).__init__()
        self.in_planes = nfilters
        self.pre_layers = nn.Sequential(
            nn.Conv2d(nchannels,nfilters,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(nfilters),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.layer1 = self._make_layer(block, 1*nfilters, nblocks[0], level=level)
        self.layer2 = self._make_layer(block, 2*nfilters, nblocks[1], stride=2, level=level)
        self.layer3 = self._make_layer(block, 4*nfilters, nblocks[2], stride=2, level=level)
        #self.layer4 = self._make_layer(block, 8*nfilters, nblocks[3], stride=2, level=level)
        self.avgpool = nn.AvgPool2d(pool, stride=1)
        self.linear = nn.Linear(4*nfilters*block.expansion, nclasses)

    def _make_layer(self, block, planes, nblocks, stride=1, level=0.2):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut, level=level))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, level=level))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.pre_layers(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x5 = self.layer3(x3)
        x6 = self.avgpool(x5)
        x7 = x6.view(x6.size(0), -1)
        x8 = self.linear(x7)
        return x8

class NoiseResNet512(nn.Module):
    def __init__(self, block, nblocks, nchannels, nfilters, nclasses, pool, level):
        super(NoiseResNet512, self).__init__()
        self.in_planes = nfilters
        self.pre_layers = nn.Sequential(
            nn.Conv2d(nchannels,nfilters,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(nfilters),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.layer1 = self._make_layer(block, 1*nfilters, nblocks[0], level=level)
        self.layer2 = self._make_layer(block, 2*nfilters, nblocks[1], stride=2, level=level)
        self.layer3 = self._make_layer(block, 2*nfilters, nblocks[2], stride=2, level=level)
        self.layer4 = self._make_layer(block, 4*nfilters, nblocks[3], stride=2, level=level)
        self.layer5 = self._make_layer(block, 4*nfilters, nblocks[4], stride=2, level=level)
        self.layer6 = self._make_layer(block, 8*nfilters, nblocks[5], stride=2, level=level)
        self.layer7 = self._make_layer(block, 8*nfilters, nblocks[6], stride=2, level=level)
        self.avgpool = nn.AvgPool2d(pool, stride=1)
        self.linear = nn.Linear(8*nfilters*block.expansion, nclasses)

    def _make_layer(self, block, planes, nblocks, stride=1, level=0.2):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut, level=level))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, level=level))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.pre_layers(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        x7 = self.layer6(x6)
        x8 = self.layer7(x7)
        x9 = self.avgpool(x8)
        x10 = x9.view(x9.size(0), -1)
        x11 = self.linear(x10)
        return x11

class MTNoiseResNet32(nn.Module):
    def __init__(self, block, nblocks, nchannels, nfilters, nclasses, pool, level, seeds):
        super(MTNoiseResNet32, self).__init__()
        self.in_planes = nfilters
        self.pre_layers = nn.Sequential(
            nn.Conv2d(nchannels,nfilters,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(nfilters),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.layer1 = self._make_layer(block, 1*nfilters, nblocks[0], level=level, seed=seeds[0])
        self.layer2 = self._make_layer(block, 2*nfilters, nblocks[1], stride=2, level=level, seed=seeds[1])
        self.layer3 = self._make_layer(block, 4*nfilters, nblocks[2], stride=2, level=level, seed=seeds[2])
        #self.layer4 = self._make_layer(block, 8*nfilters, nblocks[3], stride=2, level=level)
        self.avgpool = nn.AvgPool2d(pool, stride=1)
        self.linear = nn.Linear(4*nfilters*block.expansion, nclasses)

    def _make_layer(self, block, planes, nblocks, stride=1, level=0.2, seed=0):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut, level=level, seed=seed))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, level=level, seed=seed))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.pre_layers(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x5 = self.layer3(x3)
        x6 = self.avgpool(x5)
        x7 = x6.view(x6.size(0), -1)
        x8 = self.linear(x7)
        return x8

class MTNoiseResNet512(nn.Module):
    def __init__(self, block, nblocks, nchannels, nfilters, nclasses, pool, level, seed):
        super(MTNoiseResNet512, self).__init__()
        self.in_planes = nfilters
        self.pre_layers = nn.Sequential(
            nn.Conv2d(nchannels,nfilters,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(nfilters),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.layer1 = self._make_layer(block, 1*nfilters, nblocks[0], level=level, seed=seed)
        self.layer2 = self._make_layer(block, 2*nfilters, nblocks[1], stride=2, level=level, seed=seed+nblocks[0])
        self.layer3 = self._make_layer(block, 2*nfilters, nblocks[2], stride=2, level=level, seed=seed+nblocks[1])
        self.layer4 = self._make_layer(block, 4*nfilters, nblocks[3], stride=2, level=level, seed=seed+nblocks[2])
        self.layer5 = self._make_layer(block, 4*nfilters, nblocks[4], stride=2, level=level, seed=seed+nblocks[3])
        self.layer6 = self._make_layer(block, 8*nfilters, nblocks[5], stride=2, level=level, seed=seed+nblocks[4])
        self.layer7 = self._make_layer(block, 8*nfilters, nblocks[6], stride=2, level=level, seed=seed+nblocks[5])
        self.avgpool = nn.AvgPool2d(pool, stride=1)
        self.linear = nn.Linear(8*nfilters*block.expansion, nclasses)

    def _make_layer(self, block, planes, nblocks, stride=1, level=0.2, seed=0):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut, level=level, seed=seed))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, level=level, seed=seed))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.pre_layers(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        x7 = self.layer6(x6)
        x8 = self.layer7(x7)
        x9 = self.avgpool(x8)
        x10 = x9.view(x9.size(0), -1)
        x11 = self.linear(x10)
        return x11

class MTNoiseResNet1024(nn.Module):
    def __init__(self, block, nblocks, nchannels, nfilters, nclasses, pool, level, seed):
        super(MTNoiseResNet1024, self).__init__()
        self.in_planes = nfilters
        self.pre_layers = nn.Sequential(
            nn.Conv2d(nchannels,nfilters,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(nfilters),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.layer1 = self._make_layer(block, 1*nfilters, nblocks[0], level=level, seed=seed)
        self.layer2 = self._make_layer(block, 2*nfilters, nblocks[1], stride=2, level=level, seed=seed+nblocks[0])
        self.layer3 = self._make_layer(block, 2*nfilters, nblocks[2], stride=2, level=level, seed=seed+nblocks[1])
        self.layer4 = self._make_layer(block, 2*nfilters, nblocks[3], stride=2, level=level, seed=seed+nblocks[2])
        self.layer5 = self._make_layer(block, 4*nfilters, nblocks[4], stride=2, level=level, seed=seed+nblocks[3])
        self.layer6 = self._make_layer(block, 4*nfilters, nblocks[5], stride=2, level=level, seed=seed+nblocks[4])
        self.layer7 = self._make_layer(block, 8*nfilters, nblocks[6], stride=2, level=level, seed=seed+nblocks[5])
        self.layer8 = self._make_layer(block, 8*nfilters, nblocks[7], stride=2, level=level, seed=seed+nblocks[6])
        self.avgpool = nn.AvgPool2d(pool, stride=1)
        self.linear = nn.Linear(8*nfilters*block.expansion, nclasses)

    def _make_layer(self, block, planes, nblocks, stride=1, level=0.2, seed=0):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut, level=level, seed=seed))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, level=level, seed=seed))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.pre_layers(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        x7 = self.layer6(x6)
        x8 = self.layer7(x7)
        x9 = self.layer8(x8)
        x10 = self.avgpool(x9)
        x11 = x10.view(x10.size(0), -1)
        x12 = self.linear(x11)
        return x12

class MTNoiseResNet2048(nn.Module):
    def __init__(self, block, nblocks, nchannels, nfilters, nclasses, pool, level, seed):
        super(MTNoiseResNet2048, self).__init__()
        self.in_planes = nfilters
        self.pre_layers = nn.Sequential(
            nn.Conv2d(nchannels,nfilters,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(nfilters),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.layer1 = self._make_layer(block, 1*nfilters, nblocks[0], level=level, seed=seed)
        self.layer2 = self._make_layer(block, 2*nfilters, nblocks[1], stride=2, level=level, seed=seed+nblocks[0])
        self.layer3 = self._make_layer(block, 2*nfilters, nblocks[2], stride=2, level=level, seed=seed+nblocks[1])
        self.layer4 = self._make_layer(block, 2*nfilters, nblocks[3], stride=2, level=level, seed=seed+nblocks[2])
        self.layer5 = self._make_layer(block, 4*nfilters, nblocks[4], stride=2, level=level, seed=seed+nblocks[3])
        self.layer6 = self._make_layer(block, 4*nfilters, nblocks[5], stride=2, level=level, seed=seed+nblocks[4])
        self.layer7 = self._make_layer(block, 4*nfilters, nblocks[6], stride=2, level=level, seed=seed+nblocks[5])
        self.layer8 = self._make_layer(block, 8*nfilters, nblocks[7], stride=2, level=level, seed=seed+nblocks[6])
        self.layer9 = self._make_layer(block, 8*nfilters, nblocks[8], stride=2, level=level, seed=seed+nblocks[7])
        self.avgpool = nn.AvgPool2d(pool, stride=1)
        self.linear = nn.Linear(8*nfilters*block.expansion, nclasses)

    def _make_layer(self, block, planes, nblocks, stride=1, level=0.2, seed=0):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut, level=level, seed=seed))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, level=level, seed=seed))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.pre_layers(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = self.layer5(x5)
        x7 = self.layer6(x6)
        x8 = self.layer7(x7)
        x9 = self.layer8(x8)
        x10 = self.layer9(x9)
        x11 = self.avgpool(x10)
        x12 = x11.view(x11.size(0), -1)
        x13 = self.linear(x12)
        return x13

class LCGNoiseResNet32(nn.Module):
    def __init__(self, block, nblocks, nchannels, nfilters, nclasses, pool, level, seeds):
        super(LCGNoiseResNet32, self).__init__()
        self.in_planes = nfilters
        self.pre_layers = nn.Sequential(
            nn.Conv2d(nchannels,nfilters,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(nfilters),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.layer1 = self._make_layer(block, 1*nfilters, nblocks[0], level=level, seed=seeds[0])
        self.layer2 = self._make_layer(block, 2*nfilters, nblocks[1], stride=2, level=level, seed=seeds[1])
        self.layer3 = self._make_layer(block, 4*nfilters, nblocks[2], stride=2, level=level, seed=seeds[2])
        #self.layer4 = self._make_layer(block, 8*nfilters, nblocks[3], stride=2, level=level)
        self.avgpool = nn.AvgPool2d(pool, stride=1)
        self.linear = nn.Linear(4*nfilters*block.expansion, nclasses)

    def _make_layer(self, block, planes, nblocks, stride=1, level=0.2, seed=0):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut, level=level, seed=seed))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, level=level, seed=seed))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.pre_layers(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x5 = self.layer3(x3)
        x6 = self.avgpool(x5)
        x7 = x6.view(x6.size(0), -1)
        x8 = self.linear(x7)
        return x8

class LCGNoiseResNet32_(nn.Module):
    def __init__(self, block, nblocks, nchannels, nfilters, nclasses, pool, level, seeds, sizes):
        super(LCGNoiseResNet32_, self).__init__()
        self.in_planes = nfilters
        self.pre_layers = nn.Sequential(
            nn.Conv2d(nchannels,nfilters,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(nfilters),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.layer1 = self._make_layer(block, 1*nfilters, nblocks[0], level=level, seed=seeds[0], size=sizes[0])
        self.layer2 = self._make_layer(block, 2*nfilters, nblocks[1], stride=2, level=level, seed=seeds[1], size=sizes[1])
        self.layer3 = self._make_layer(block, 4*nfilters, nblocks[2], stride=2, level=level, seed=seeds[2], size=sizes[2])
        #self.layer4 = self._make_layer(block, 8*nfilters, nblocks[3], stride=2, level=level)
        self.avgpool = nn.AvgPool2d(pool, stride=1)
        self.linear = nn.Linear(4*nfilters*block.expansion, nclasses)

    def _make_layer(self, block, planes, nblocks, stride=1, level=0.2, seed=0, size=1):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut, level=level, seed=seed, size=[self.in_planes, size, size]))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, level=level, seed=seed, size=[self.in_planes, int(size/stride), int(size/stride)]))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.pre_layers(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x5 = self.layer3(x3)
        x6 = self.avgpool(x5)
        x7 = x6.view(x6.size(0), -1)
        x8 = self.linear(x7)
        return x8

class ArgNoiseResNet32(nn.Module):
    def __init__(self, block, nblocks, nchannels, nfilters, nclasses, pool, seeds, level):
        super(ArgNoiseResNet32, self).__init__()
        self.in_planes = nfilters
        self.pre_layers = nn.Sequential(
            nn.Conv2d(nchannels,nfilters,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(nfilters),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.layer1 = self._make_layer(block, 1*nfilters, nblocks[0], seed=seeds[0], level=level)
        self.layer2 = self._make_layer(block, 2*nfilters, nblocks[1], stride=2, seed=seeds[1], level=level)
        self.layer3 = self._make_layer(block, 4*nfilters, nblocks[2], stride=2, seed=seeds[2], level=level)
        #self.layer4 = self._make_layer(block, 8*nfilters, nblocks[3], stride=2, level=level)
        self.avgpool = nn.AvgPool2d(pool, stride=1)
        self.linear = nn.Linear(4*nfilters*block.expansion, nclasses)

    def _make_layer(self, block, planes, nblocks, stride=1, seed=0, level=0.2):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut, seed=seed, level=level))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, seed = seed, level=level))
        return nn.Sequential(*layers)

    def forward(self, x):
        print (x.size())
        x1 = self.pre_layers(x)
        print (x1.size())
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x5 = self.layer3(x3)
        x6 = self.avgpool(x5)
        x7 = x6.view(x6.size(0), -1)
        x8 = self.linear(x7)
        return x8

class NoiseResNetEco32(nn.Module):
    def __init__(self, block, nblocks, nchannels, nfilters, nclasses, pool, level):
        super(NoiseResNetEco32, self).__init__()
        self.in_planes = nfilters
        self.pre_layers = nn.Sequential(
            nn.Conv2d(nchannels,nfilters,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(nfilters),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.layer1 = self._make_layer(block, 1*nfilters, nblocks[0], level=level)
        self.layer2 = self._make_layer(block, 2*nfilters, nblocks[1], stride=2, level=level)
        self.layer3 = self._make_layer(block, 4*nfilters, nblocks[2], stride=2, level=level)
        self.layer4 = self._make_layer(block, nclasses, nblocks[3], stride=2, level=level)
        #self.layer4 = self._make_layer(block, 8*nfilters, nblocks[3], stride=2, level=level)
        self.avgpool = nn.AvgPool2d(pool, stride=1)
        self.linear = nn.Linear(4*nfilters*block.expansion, nclasses)

    def _make_layer(self, block, planes, nblocks, stride=1, level=0.2):
        shortcut = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            shortcut = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_planes, planes, stride, shortcut, level=level))
        self.in_planes = planes * block.expansion
        for i in range(1, nblocks):
            layers.append(block(self.in_planes, planes, level=level))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.pre_layers(x) #([64, 3, 32, 32]) => ([64, 128, 8, 8])
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        return x5.view(x5.size(0), -1)

def noiseresnet18(nchannels, nfilters, nclasses, pool=7, level=0.1):
    return NoiseResNet(NoiseBasicBlock, [2,2,2,2], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses, pool=pool, level=level)

def noiseresnet18_32(nchannels, nfilters, nclasses, pool=7, level=0.1):
    return NoiseResNet32(NoiseBasicBlock, [2,2,2,2], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses, pool=pool, level=level)

def argnoiseresnet18_32(nchannels, nfilters, nclasses, pool=7, level=0.1):
    return NoiseResNet32(NoiseBasicBlock, [2,2,2,2], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses, pool=pool, seeds=[0,2,4,6], level=level)

def noiseresnet34(nchannels, nfilters, nclasses, pool=7, level=0.1):
    return NoiseResNet(NoiseBasicBlock, [3,4,6,3], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses, pool=pool, level=level)

def noiseresnet50(nchannels, nfilters, nclasses, pool=7, level=0.1):
    return NoiseResNet(NoiseBottleneck, [3,4,6,3], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses, pool=pool, level=level)

def noiseresnet101(nchannels, nfilters, nclasses, pool=7, level=0.1):
    return NoiseResNet(NoiseBottleneck, [3,4,23,3], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses, pool=pool, level=level)

def noiseresnet152(nchannels, nfilters, nclasses, pool=7, level=0.1):
    return NoiseResNet(NoiseBottleneck, [3,8,36,3], nchannels=nchannels, nfilters=nfilters, nclasses=nclasses, pool=pool, level=level)
