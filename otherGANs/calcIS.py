import torch
import glob
import os
import numpy as np

import past_models
from inception_score import inception_score
from dataset import makeDataloader

import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision import datasets
import torchvision.transforms as transforms

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

import logging

import easydict

opt = easydict.EasyDict({
    'n_epochs': 200,
    'batch_size': 128,
    'lr': 0.0002,
    'b1': 0.5,
    'b2': 0.999,
    'n_cpu': 8,
    'latent_dim': 128,
    'img_size': 32,
    'n_critic': 1,
    'clip_value': 0.01,
    'sample_interval': 100,
    'log_interval': 10,
    'dataset': 'cifar10',
    'num_filters': 128, #for CNN Discriminator and Generator
    'saveDir' : None,
    'resume' : None,
    'loadDir' : './otherGANs/1035:181227_WGAN-GP_DCGANGenerator32_DCGANDiscriminator32_mnist'
})

class IgnoreLabelDataset(torch.utils.data.Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)

def calcurateInceptionScore(opt, generator, idx):
    num4eval = 1024
    assert num4eval % opt.batch_size == 0, 'num4eval:%d % opt.batch_size:%d != 0' % (num4eval, opt.batch_size)

    saveDir = os.path.join(opt.loadDir, 'fake_%s' % idx)
    os.makedirs(saveDir, exist_ok = True)
    os.makedirs(os.path.join(saveDir, 'img'), exist_ok = True)

    for j in range(int(num4eval / opt.batch_size)):
        z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

        if 'Noise' in generator.__class__.__name__ or 'WGAN' in generator.__class__.__name__:
            fake_imgs = generator(z)
        else:
            fake_imgs = generator(z.view(*z.size(), 1, 1))

        for i in range(fake_imgs.size(0)):
            vutils.save_image(fake_imgs.data[i], (os.path.join(saveDir, 'img', "fake_%s.png")) % str(i+j*opt.batch_size).zfill(4), normalize=True)

    dataset = datasets.ImageFolder(root=saveDir,
                            transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]))

    IgnoreLabelDataset(dataset)

    if opt.dataset == 'mnist':
        calcIS = inception_score(IgnoreLabelDataset(dataset), cuda=cuda, batch_size=32, resize=True, expand=True)
    else:
        calcIS = inception_score(IgnoreLabelDataset(dataset), cuda=cuda, batch_size=32, resize=True)

    return calcIS


def main(opt, generator):
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    handler1 = logging.StreamHandler()
    handler1.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))
    logger.addHandler(handler1)
    handler2 = logging.FileHandler(filename=os.path.join(opt.loadDir, "is.log"))
    handler2.setLevel(logging.INFO)
    handler2.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler2)
    logger.info(opt)

    np.random.seed(seed=0)

    #calcurate real dataset IS
    if opt.calcRealIs:
        dataset, _ = makeDataloader(opt)
        IgnoreLabelDataset(dataset)
        if opt.dataset == 'mnist':
            calcIS = inception_score(IgnoreLabelDataset(dataset), cuda=cuda, batch_size=32, resize=True, expand=True)
        else:
            calcIS = inception_score(IgnoreLabelDataset(dataset), cuda=cuda, batch_size=32, resize=True)
        logger.info('real, ' + str(calcIS[0]))

    #calcurate fake dataset init IS
    calcIS = calcurateInceptionScore(opt, generator.cuda(), str(0))
    logger.info('0, ' + str(calcIS[0]))

    #calcurate fake dataset IS per iter
    for model_path in sorted(glob.glob(os.path.join(opt.loadDir, 'generator_*'))):
        name = os.path.basename(model_path)
        idx = name.replace('generator_model_', '')

        calcG = generator.cuda()
        calcG.load_state_dict(torch.load(model_path))

        if int(idx) > 0:
            calcIS = calcurateInceptionScore(opt, calcG, idx)
            logger.info(str(int(idx)) + ', ' + str(calcIS[0]))
