import models
import dataset
from main import train
from otherGANs import past_models
import naiveresnet
import noise_layers

import easydict

opt = easydict.EasyDict({
    'n_epochs': 200,
    'batch_size': 64,
    'lr': 0.0002,
    'b1': 0.5,
    'b2': 0.999,
    'n_cpu': 8,
    'latent_dim': 128,
    'img_size': 32,
    'n_critic': 1,
    'clip_value': 0.01,
    'sample_interval': 100,
    'log_interval': 100,
    'modelsave_interval': 1, #per epoch
    'dataset': 'mnist',
    'num_filters': 128,
    'saveDir' : None,
    'resume' : None,
    'logIS' : True,
    'loadDir' : None
})

_, dataloader = dataset.makeDataloader(opt)

# Initialize generator and discriminator
#generator = models.NoiseGenerator2Dv6(opt)
#generator = models.MTNoiseGenerator2Dv6(opt)
generator = models.NoiseResGenerator2Dv1(opt, noise_layers.NoiseBasicBlock2Dv2, [2,2,2,2,1], level=0.1)
discriminator = naiveresnet.NoiseResNet32(naiveresnet.NoiseBasicBlock, [2,2,2,2], nchannels=1, nfilters=opt.num_filters, nclasses=1, pool=2, level=0.1)
#discriminator = naiveresnet.LCGNoiseResNet32_(naiveresnet.LCGNoiseBasicBlock_, [2,2,2,2], nchannels=3, nfilters=opt.num_filters, nclasses=1, pool=2, level=0.1, seeds=[100, 200, 300], sizes=[8, 8, 4])
#discriminator = naiveresnet.MTNoiseResNet32(naiveresnet.MTNoiseBasicBlock, [2,2,2,2], nchannels=3, nfilters=opt.num_filters, nclasses=1, pool=2, level=0.1, seeds=[100, 200, 300])
#discriminator = naiveresnet.NoiseResNet32(naiveresnet.NoiseBasicBlock, [2,2,2,2], nchannels=3, nfilters=opt.num_filters, nclasses=1, pool=2, level=0.1)
#discriminator = past_models.WGANDiscriminator32_(opt)
#discriminator = past_models.DCGANDiscriminator32_(opt)

train(generator, discriminator, dataloader, opt)
