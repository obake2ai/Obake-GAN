import torch
import argparse
import os
import numpy as np
import math
import sys
import time
import datetime
import shutil

import logging
logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
handler1 = logging.StreamHandler()
handler1.setFormatter(logging.Formatter("%(asctime)s %(levelname)8s %(message)s"))
logger.addHandler(handler1)

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.utils as vutils
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from calcIS import calcurateInceptionScore

def compute_gradient_penalty(D, real_samples, fake_samples, Tensor):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def logInceptionScore(logger, opt, generator, epoch, batches_done, loadDir, maxIS):
    opt.loadDir = loadDir
    idx = str(batches_done)
    score = calcurateInceptionScore(opt, generator, idx.zfill(8))
    maxIS = score[0] if score[0] > maxIS else maxIS
    logger.info(
        "[Epoch: %d/%d] [Iteration: %d] [Inception Score: %s] [Max Score Ever: %s]"
        % (epoch, opt.n_epochs,  batches_done, "{0:.2f}".format(score[0]), "{0:.2f}".format(maxIS))
    )
    return maxIS

def train(generator, discriminator, dataloader, opt):
    gName = generator.__class__.__name__
    dName = discriminator.__class__.__name__
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        generator.cuda()
        discriminator.cuda()
        generator = torch.nn.DataParallel(generator) # make parallel
        discriminator = torch.nn.DataParallel(discriminator) # make parallel
        torch.backends.cudnn.benchmark = True

    datasetName = opt.dataset
    date = datetime.datetime.now()
    dateInfo =  str(date.hour).zfill(2) + str(date.minute).zfill(2) + ':' + str(date.year).replace('20','') + str(date.month).zfill(2) + str(date.day).zfill(2)
    saveDir = dateInfo + '_' + gName + '_' + dName + '_' + datasetName
    os.makedirs(saveDir, exist_ok = True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    # deepPG vs simpleCD Training
    # ----------

    # Loss weight for gradient penalty
    lambda_gp = 10
    batches_done = 0
    epoch_done = 0
    start = time.time()
    maxIS = 0
    num_eval = 50 if opt.img_size <= 512 else 10
    fixed_z = Variable(Tensor(np.random.normal(0, 1, (num_eval, opt.latent_dim))))

    if opt.resume != 0:
        batches_done += opt.resume
        epoch_done += int(opt.resume/len(dataloader))
        generator.load_state_dict(torch.load(os.path.join(opt.loadDir, "generator_model_%s") % str(opt.resume - opt.resume%opt.modelsave_interval).zfill(8)))
        discriminator.load_state_dict(torch.load(os.path.join(opt.loadDir, "discriminator_model_%s") % str(opt.resume - opt.resume%opt.modelsave_interval).zfill(8)))
        shutil.rmtree(saveDir)
        saveDir = opt.loadDir
        handler2 = logging.FileHandler(filename=os.path.join(saveDir, "train.log"))
        handler2.setLevel(logging.INFO)
        handler2.setFormatter(logging.Formatter("%(asctime)s :%(message)s"))
        logger.addHandler(handler2)

        logger.info('resume: loaded models')
        logger.info("generator_model_%s" % str(epoch_done-1).zfill(4))
        logger.info("discriminator_model_%s" % str(epoch_done-1).zfill(4))
    else:
        handler2 = logging.FileHandler(filename=os.path.join(saveDir, "train.log"))
        handler2.setLevel(logging.INFO)
        handler2.setFormatter(logging.Formatter("%(asctime)s :%(message)s"))
        logger.addHandler(handler2)

        logger.info(opt)
        #logger.info(gName)
        logger.info(generator)
        #logger.info(dName)
        logger.info(discriminator)

    for epoch in range(opt.n_epochs):
        if epoch == 0 and epoch_done != 0: epoch += epoch_done
        for i, (imgs, _) in enumerate(dataloader, batches_done % len(dataloader)):
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

            # Real images
            real_validity = discriminator(real_imgs)

            # Fake images
            fake_validity = discriminator(fake_imgs)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data, Tensor)

            # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()

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
                fake_validity = discriminator(fake_imgs)
                g_loss = -torch.mean(fake_validity)
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
                        vutils.save_image(real_imgs.data[:num_eval-1], (os.path.join(saveDir, opt.dataset + "_real.png")), nrow=int(math.sqrt(num_eval - 1)), normalize=True)
                        vutils.save_image(generator(fixed_z).data[:num_eval-1], (os.path.join(saveDir, opt.dataset + "_fake_%s.png")) % str(batches_done).zfill(8), nrow=int(math.sqrt(num_eval - 1)), normalize=True)
                    else:
                        vutils.save_image(generator(fixed_z).data[:num_eval-1], (os.path.join(saveDir, opt.dataset + "_fake_%s.png")) % str(batches_done).zfill(8), nrow=int(math.sqrt(num_eval - 1)), normalize=True)

                if batches_done % opt.modelsave_interval == 0:
                    torch.save(generator.state_dict(), os.path.join(saveDir, "generator_model_%s") % str(batches_done).zfill(8))
                    torch.save(discriminator.state_dict(), os.path.join(saveDir, "discriminator_model_%s") % str(batches_done).zfill(8))
                    if opt.logIS: maxIS = logInceptionScore(logger, opt, generator, epoch, batches_done, saveDir, maxIS)

                batches_done += opt.n_critic
