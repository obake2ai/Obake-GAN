import torch
import torch.nn as nn

import itertools

#Simple Noise Layer from PNN
class NoiseLayer(nn.Module):
    def __init__(self, in_planes, out_planes, level, normalize = True):
        super(NoiseLayer, self).__init__()
        self.noise = nn.Parameter(torch.Tensor(0), requires_grad=False).cuda()
        self.level = level
        if normalize:
          self.pre_layers = nn.Sequential(
              nn.ReLU(True),
              nn.BatchNorm1d(in_planes, 0.8),
          )
        else:
          self.pre_layers = nn.Sequential(
              nn.ReLU(True),
          )
        self.post_layers = nn.Sequential(
            nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1),
        )

    def forward(self, x):
        if self.noise.numel() == 0:
            self.noise.resize_(x.data[0].shape).uniform_()
            self.noise = (2 * self.noise - 1) * self.level

        x1 = torch.add(x, self.noise)
        resized_x1 = x1.view(x1.size(0), x1.size()[1], 1)
        x2 = self.pre_layers(resized_x1)

        z = self.post_layers(x2)
        return z.view(z.size(0), z.size(1))

class NoiseLayer2D(nn.Module):
    def __init__(self, in_planes, out_planes, level, normalize=True, seed=None):
        super(NoiseLayer2D, self).__init__()

        if seed is not None: torch.manual_seed(seed)
        self.noise = torch.randn(1,in_planes,1,1).cuda()
        self.level = level
        if normalize:
            self.layers = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
            )
        else:
            self.layers = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
            )

    def forward(self, x):
        tmp1 = x.data.shape
        tmp2 = self.noise.shape

        if (tmp1[1] != tmp2[1]) or (tmp1[2] != tmp2[2]) or (tmp1[3] != tmp2[3]):
            self.noise = (2*torch.rand(x.data.shape)-1)*self.level
            self.noise = self.noise.cuda()

        if tmp1[0] < tmp2[0]: x.data = x.data + self.noise[:tmp1[0]]
        else: x.data = x.data + self.noise

        x = self.layers(x)
        return x

RAND_MAX = 0xffffffff #2^32
M = 65539 #http://www.geocities.jp/m_hiroi/light/index.html#cite
import random

class LCG:
    def __init__(self, seed):
        assert seed != None, 'set seed'
        self.seed = seed

    def irand(self):
        self.seed = (M * self.seed + 1) & RAND_MAX
        return self.seed / (RAND_MAX / 10) / 10

class PoolLCG:
    def __init__(self, gen, seed, dim, pool_size = 255):
        assert seed != None, 'set seed'
        self.gen = gen(seed)
        self.pool_size = pool_size
        self.pool = [self.gen.irand() for _ in range(self.pool_size)]
        self.next = self.pool_size - 1
        self.dim = dim
        random.shuffle(self.pool)

    def irand(self):
        self.next = int(self.pool[self.next] % self.pool_size)
        x = self.pool[self.next : self.next+self.dim]
        #self.pool[self.next] = self.gen.irand()
        return x

class FitLCG:
    def __init__(self, gen, seed, dim):
        assert seed != None, 'set seed'
        self.gen = gen(seed)
        self.pool_size = dim
        self.pool = [self.gen.irand() for _ in range(self.pool_size)]
        random.shuffle(self.pool)

    def irand(self):
        return self.pool

    def shuffle(self):
        random.shuffle(self.pool)
        return self.pool

class MaskLCG:
    def __init__(self, gen, seed):
        assert seed != None, 'set seed'
        self.gen = gen(seed)

    def make(self, size, level):
        assert len(size) == 3, 'LCG Mask Dimention Error'
        mask = torch.zeros(*size)
        for i, j, k in itertools.product(range(size[0]), range(size[1]), range(size[2])):
          mask.data[i, j, k] = self.gen.irand() * level
        return mask

class AlgorithmicNoiseLayer(nn.Module):
    def __init__(self, in_planes, out_planes, noise_seed, level, normalize = True):
        super(AlgorithmicNoiseLayer, self).__init__()
        self.seed = noise_seed
        self.out_planes = out_planes
        self.level = level
        if normalize:
          self.pre_layers = nn.Sequential(
              nn.ReLU(True),
              nn.BatchNorm1d(in_planes, 0.8),
          )
        else:
          self.pre_layers = nn.Sequential(
              nn.ReLU(True),
          )
        self.post_layers = nn.Sequential(
            nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=1),
        )

    def forward(self, x):
        noiseAdder = FitLCG(LCG, self.seed, x.size()[1])
        x1 = torch.add(x, torch.Tensor(noiseAdder.irand()).cuda() * self.level)
        x2 = self.pre_layers(x1.view(x.size()[0], x1.size()[1], 1))
        z = self.post_layers(x2)
        return z.view(z.size()[0], z.size()[1])

class MTNoiseLayer2D(nn.Module):
    def __init__(self, in_planes, out_planes, level, seed, normalize=True):
        super(MTNoiseLayer2D, self).__init__()

        self.level = level
        self.seed = seed
        if normalize:
            self.layers = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_planes),
            )
        else:
            self.layers = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
            )

    def forward(self, x):
        torch.manual_seed(self.seed)
        x2 = torch.add(x, torch.rand(x.size(1), x.size(2), x.size(3)).cuda() * self.level)

        z = self.layers(x2)
        return z

class MTSNDNoiseLayer2D(nn.Module):
    def __init__(self, in_planes, out_planes, level, seed, normalize=True):
        super(MTSNDNoiseLayer2D, self).__init__()

        self.level = level
        self.seed = seed
        if normalize:
            self.layers = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_planes),
            )
        else:
            self.layers = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
            )

    def forward(self, x):
        torch.manual_seed(self.seed)
        x2 = torch.add(x, torch.randn(x.size(1), x.size(2), x.size(3)).cuda() * self.level)

        z = self.layers(x2)
        return z

class MTSNDNoiseLayer2D_x4(nn.Module):
    def __init__(self, in_planes, out_planes, level, seed, normalize=True):
        super(MTSNDNoiseLayer2D_x4, self).__init__()

        self.level = level
        self.seed = seed
        if normalize:
            self.layers = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_planes),
            )
        else:
            self.layers = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
            )

    def forward(self, x):
        torch.manual_seed(self.seed)
        if x.size(2) >= 512:
            bar = int(x.size(2)/2)
            x2 = x
            x2[:,:,:bar,:bar] = torch.add(x[:,:,:bar,:bar], torch.randn(x.size(1), bar, bar).cuda() * self.level)
            torch.manual_seed(self.seed+1)
            x2[:,:,bar:bar*2,:bar] = torch.add(x[:,:,bar:bar*2,:bar], torch.randn(x.size(1), bar, bar).cuda() * self.level)
            torch.manual_seed(self.seed+2)
            x2[:,:,:bar,bar:bar*2] = torch.add(x[:,:,:bar,bar:bar*2], torch.randn(x.size(1), bar, bar).cuda() * self.level)
            torch.manual_seed(self.seed+3)
            x2[:,:,bar:bar*2,bar:bar*2] = torch.add(x[:,:,bar:bar*2,bar:bar*2], torch.randn(x.size(1), bar, bar).cuda() * self.level)
        else:
            x2 = torch.add(x, torch.randn(x.size(1), x.size(2), x.size(3)).cuda() * self.level)

        z = self.layers(x2)
        return z

class MTSNDNoiseLayer2Dx(nn.Module):
    def __init__(self, in_planes, out_planes, level, seed, normalize=True):
        super(MTSNDNoiseLayer2Dx, self).__init__()

        self.level = level
        self.seed = seed
        if normalize:
            self.layers = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_planes),
            )
        else:
            self.layers = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
            )

    def forward(self, x):
        torch.manual_seed(self.seed)
        if x.size(2) >= 512 and x.size(2) < 2048:
            split = 4 if x.size(2) < 2048 else 8
            bar = int(x.size(2)/split)
            x2 = x
            k = 0
            for i, j in itertools.product(range(split), range(split)):
                torch.manual_seed(self.seed+k)
                x2[:,:,bar*j:bar*(j+1),bar*i:bar*(i+1)] = torch.add(x[:,:,bar*j:bar*(j+1),bar*i:bar*(i+1)], torch.randn(x.size(1), bar, bar).cuda() * self.level)
                k += 1
        else:
            x2 = torch.add(x, torch.randn(x.size(1), x.size(2), x.size(3)).cuda() * self.level)

        z = self.layers(x2)
        return z

class LCGNoiseLayer2D(nn.Module):
    def __init__(self, in_planes, out_planes, level, seed, normalize=True):
        super(LCGNoiseLayer2D, self).__init__()

        self.level = level
        self.seed = seed
        if normalize:
            self.layers = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_planes),
            )
        else:
            self.layers = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
            )

    def forward(self, x):
        noiseMaker = MaskLCG(LCG, self.seed)
        noise = noiseMaker.make([x.size(1), x.size(2), x.size(3)], self.level)
        x2 = torch.add(x, noise.cuda())
        z = self.layers(x2)
        return z

class LCGNoiseLayer2D_(nn.Module):
    def __init__(self, in_planes, out_planes, level, seed, size, normalize=True):
        super(LCGNoiseLayer2D_, self).__init__()
        noiseMaker = MaskLCG(LCG, seed)
        self.noise = noiseMaker.make(size, level).cuda()
        if normalize:
            self.layers = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_planes),
            )
        else:
            self.layers = nn.Sequential(
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1),
            )

    def forward(self, x):
        x2 = torch.add(x, self.noise)
        z = self.layers(x2)
        return z

class NoiseBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, shortcut=None, level=0.2, normalize=True):
        super(NoiseBasicBlock, self).__init__()
        self.layers = nn.Sequential(
            NoiseLayer(in_planes, out_planes, level, normalize),
            NoiseLayer(out_planes, out_planes, level),
        )
        self.shortcut = shortcut
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = self.relu(y)
        return y

class NoiseBasicBlock2D(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, shortcut=None, level=0.2, normalize=True, seed=None):
        super(NoiseBasicBlock2D, self).__init__()
        self.layers = nn.Sequential(
            NoiseLayer2D(in_planes, out_planes, level, normalize, seed),
            NoiseLayer2D(out_planes, out_planes, level, True, seed),
        )
        self.shortcut = shortcut
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = self.relu(y)
        return y

class NoiseBasicBlock2Dv2(nn.Module):
    expansion = 1
    def __init__(self, in_planes, out_planes, scale=1, shortcut=None, level=0.2, normalize=True):
        super(NoiseBasicBlock2Dv2, self).__init__()
        self.layers = nn.Sequential(
            NoiseLayer2D(in_planes, out_planes, level, normalize),
            nn.Upsample(scale_factor=scale, mode='bilinear'),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(),
            NoiseLayer2D(out_planes, out_planes, level),
            nn.BatchNorm2d(out_planes),
        )
        self.shortcut = shortcut
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = self.relu(y)
        return y

class MTNoiseBasicBlock2D(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, shortcut=None, level=0.2, seed=0, normalize=True):
        super(MTNoiseBasicBlock2D, self).__init__()
        self.layers = nn.Sequential(
            MTNoiseLayer2D(in_planes, out_planes, level, seed, normalize),
            MTNoiseLayer2D(out_planes, out_planes, level, seed+1),
        )
        self.shortcut = shortcut
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = self.relu(y)
        return y

class MTSNDNoiseBasicBlock2D_x4(nn.Module):
    expansion = 1
    def __init__(self, in_planes, out_planes, stride=1, shortcut=None, level=0.2, seed=0, normalize=True):
        super(MTSNDNoiseBasicBlock2D_x4, self).__init__()
        self.layers = nn.Sequential(
            MTSNDNoiseLayer2D_x4(in_planes, out_planes, level, seed, normalize),
            MTSNDNoiseLayer2D_x4(out_planes, out_planes, level, seed+1),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        return y

class MTSNDNoiseBasicBlock2Dx(nn.Module):
    expansion = 1
    def __init__(self, in_planes, out_planes, stride=1, shortcut=None, level=0.1, seed=0, normalize=True):
        super(MTSNDNoiseBasicBlock2Dx, self).__init__()
        self.layers = nn.Sequential(
            MTSNDNoiseLayer2Dx(in_planes, out_planes, level, seed, normalize),
            MTSNDNoiseLayer2Dx(out_planes, out_planes, level, seed+1),
        )
        self.shortcut = shortcut

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        return y

class LCGNoiseBasicBlock2D(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, shortcut=None, level=0.2, seed=0, normalize=True):
        super(LCGNoiseBasicBlock2D, self).__init__()
        self.layers = nn.Sequential(
            LCGNoiseLayer2D(in_planes, out_planes, level, seed, normalize),
            LCGNoiseLayer2D(out_planes, out_planes, level, seed+1),
        )
        self.shortcut = shortcut
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = self.relu(y)
        return y

class LCGNoiseBasicBlock2D_(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, shortcut=None, level=0.2, seed=0, size=[128, 1, 1], normalize=True):
        super(LCGNoiseBasicBlock2D_, self).__init__()
        self.layers = nn.Sequential(
            LCGNoiseLayer2D_(in_planes, out_planes, level, seed, size, normalize),
            LCGNoiseLayer2D_(out_planes, out_planes, level, seed+1, size),
        )
        self.shortcut = shortcut
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = self.relu(y)
        return y

class ArgNoiseBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, seed, stride=1, shortcut=None, level=0.2, normalize=True):
        super(ArgNoiseBasicBlock, self).__init__()
        self.layers = nn.Sequential(
            AlgorithmicNoiseLayer(in_planes, out_planes, seed, level, normalize),
            AlgorithmicNoiseLayer(out_planes, out_planes, seed*2, level),
        )
        self.shortcut = shortcut
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        y = self.layers(x)
        if self.shortcut:
            residual = self.shortcut(x)
        y += residual
        y = self.relu(y)
        return y
