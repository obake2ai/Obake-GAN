import models
import dataset
from main import train
from otherGANs import past_models
import naiveresnet
import noise_layers
from generateVideo import generate_interpolation_video

import easydict

opt = easydict.EasyDict({
    'n_epochs': 2000,
    'batch_size': 16,
    'lr': 0.0002,
    'b1': 0.5,
    'b2': 0.999,
    'n_cpu': 8,
    'latent_dim': 100,
    'img_size': 512,
    'n_critic': 1,
    'clip_value': 0.01,
    'sample_interval': 100,
    'modelsave_interval': 1000,
    'log_interval': 10,
    'dataset': 'imagenet',
    'num_filters': 128, #for CNN Discriminator and Generator
    'saveDir' : None,
    'resume' : 0, #0 = train from scratch
    'logIS' : False,
    'loadDir' : False
})

_, dataloader = dataset.makeDataloader(opt)

# Initialize generator and discriminator
#generator = models.NoiseGenerator2Dv6(opt)
#generator = models.MTNoiseGenerator2Dv6SND512(opt, seed=0)
generator = models.NoiseConvGenerator2Dv6_512(opt, seed=0)
#generator = models.NoiseConvGenerator2Dv2_512(opt, seed=0)
#generator = models.MTNoiseGenerator2Dv6SND1024_x4v4(opt, seed=0)
#generator = models.MTNoiseGenerator2Dv6SND2048x(opt, seed=0)
#generator = models.MTNoiseGenerator2Dv6(opt, seed=40)
#generator = models.LCGNoiseGenerator2Dv6_(opt, seed=6)
#generator = models.NoiseResGenerator2Dv1(opt, noise_layers.NoiseBasicBlock2Dv2, [2,2,2,2,1], level=0.1)
#discriminator = naiveresnet.NoiseResNet32(naiveresnet.NoiseBasicBlock, [2,2,2,2], nchannels=3, nfilters=opt.num_filters, nclasses=1, pool=2, level=0.1)
#discriminator = naiveresnet.MTNoiseResNet1024(naiveresnet.MTSNDNoiseBasicBlock, [1,1,1,1,1,1,1,1], nchannels=3, nfilters=opt.num_filters, nclasses=1, pool=2, level=0.1, seed=0)
discriminator = naiveresnet.MTNoiseResNet512(naiveresnet.MTSNDNoiseBasicBlock, [1,1,1,1,1,1,1,1], nchannels=3, nfilters=opt.num_filters, nclasses=1, pool=2, level=0.1, seed=0)
#discriminator = naiveresnet.LCGNoiseResNet32_(naiveresnet.LCGNoiseBasicBlock_, [2,2,2,2], nchannels=3, nfilters=opt.num_filters, nclasses=1, pool=2, level=0.1, seeds=[100, 200, 300], sizes=[8, 8, 4])
#discriminator = naiveresnet.MTNoiseResNet32(naiveresnet.MTNoiseBasicBlock, [2,2,2,2], nchannels=3, nfilters=opt.num_filters, nclasses=1, pool=2, level=0.1, seeds=[10, 20,90])
#discriminator = naiveresnet.LCGNoiseResNet32_(naiveresnet.LCGNoiseBasicBlock_, [2,2,2,2], nchannels=3, nfilters=opt.num_filters, nclasses=1, pool=2, level=0.1, seeds=[7, 8, 9], sizes=[8, 8, 4])
#discriminator = naiveresnet.NoiseResNet32(naiveresnet.NoiseBasicBlock, [2,2,2,2], nchannels=3, nfilters=opt.num_filters, nclasses=1, pool=2, level=0.1)
#discriminator = past_models.WGANDiscriminator32_(opt)
#discriminator = past_models.DCGANDiscriminator32_(opt)

train(generator, discriminator, dataloader, opt)
#generate_interpolation_video(generator, opt, image_shrink=1, image_zoom=1, duration_sec=180.0, smoothing_sec=1.0, mp4=None, mp4_fps=30, mp4_codec='libx265', mp4_bitrate='16M', random_seed=1000, minibatch_size=8)
