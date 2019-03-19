import torch
import argparse
import os
import numpy as np
import math
import sys
import time
import datetime

import logging
logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
handler1 = logging.StreamHandler()
handler1.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))
logger.addHandler(handler1)

from past_models import weights_init

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.utils as vutils
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

def train(generator, discriminator, dataloader, opt):
    gName = generator.__class__.__name__
    dName = discriminator.__class__.__name__
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()
        discriminator.cuda()
        #generator = torch.nn.DataParallel(generator) # make parallel
        #discriminator = torch.nn.DataParallel(discriminator) # make parallel
        torch.backends.cudnn.benchmark = True

    if opt.resume != None:
        generator.load_state_dict(torch.load(os.path.join(loadDir, "generator_model__%s") % str(batches_done).zfill(8)))
        discriminator.load_state_dict(torch.load(os.path.join(loadDir, "discriminator_model__%s") % str(batches_done).zfill(8)))

    datasetName = opt.dataset
    date = datetime.datetime.now()
    dateInfo =  str(date.hour).zfill(2) + str(date.minute).zfill(2) + ':' + str(date.year).replace('20','') + str(date.month).zfill(2) + str(date.day).zfill(2)
    saveDir = dateInfo + '_MLP_' + gName + '_' + dName + '_' + datasetName
    os.makedirs(saveDir, exist_ok = True)

    handler2 = logging.FileHandler(filename=os.path.join(saveDir, "train.log"))
    handler2.setLevel(logging.INFO)
    handler2.setFormatter(logging.Formatter("%(asctime)s :%(message)s"))
    logger.addHandler(handler2)

    logger.info(opt)
    #logger.info(gName)
    logger.info(generator)
    #logger.info(dName)
    logger.info(discriminator)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    batches_done = 0
    start = time.time()

    real_label = 1
    fake_label = 0

    criterion = nn.BCELoss()

    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            fake_imgs = generator(z)

            label = torch.full((opt.batch_size,), real_label).cuda()

            # Real images
            real_output = discriminator(real_imgs)


            # Adversarial loss
            d_loss = criterion(real_output, label)
            d_loss.backward()
            optimizer_D.step()

            # Fake images
            fake_output = discriminator(fake_imgs)

            optimizer_G.zero_grad()

            # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                # Generate a batch of images
                fake_imgs = generator(z)
                # Loss measures generator's ability to fool the discriminator
                # Train on fake images
                label.fill_(fake_label)

                fake_output = discriminator(fake_imgs)
                g_loss = criterion(fake_output, label)

                g_loss.backward()
                optimizer_G.step()

                if batches_done % opt.log_interval == 0:
                    elapsed_time = time.time() - start
                    logger.info(
                        "[Epoch: %d/%d] [Batch: %d/%d] [D loss: %f] [G loss: %f] [ElapsedTime: %s]"
                        % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), "{0:.2f}".format(elapsed_time) + " [sec]")
                    )

                if batches_done % opt.sample_interval == 0:
                    if batches_done == 0:
                        vutils.save_image(real_imgs.data[:49], (os.path.join(saveDir, opt.dataset + "_real.png")), nrow=7, normalize=True)
                        vutils.save_image(fake_imgs.data[:49], (os.path.join(saveDir, opt.dataset + "_fake_%s.png")) % str(batches_done).zfill(8), nrow=7, normalize=True)
                    else:
                        vutils.save_image(fake_imgs.data[:49], (os.path.join(saveDir, opt.dataset + "_fake_%s.png")) % str(batches_done).zfill(8), nrow=7, normalize=True)
                        torch.save(generator.state_dict(), os.path.join(saveDir, "generator_model_%s") % str(batches_done).zfill(8))
                        torch.save(discriminator.state_dict(), os.path.join(saveDir, "discriminator_model_%s") % str(batches_done).zfill(8))

                batches_done += opt.n_critic
