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

class Dataset(nn.Module):
    def __init__(self, opt):
        os.makedirs(os.path.join("data", opt.dataset), exist_ok=True)

    def mnist(self, opt):
        dataset = datasets.MNIST(
                        "./otherGANs/data/mnist",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                            transforms.Resize(opt.img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]),
                    )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
        )
        return dataset, dataloader

    def cifar10(self, opt):
        dataset = datasets.CIFAR10(
                        "./otherGANs/data/cifar10",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                            transforms.Resize(opt.img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]),
                    )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
        )
        return dataset, dataloader

    def fashion(self, opt):
        dataset = datasets.FashionMNIST(
                        "./otherGANs/data/FashionMNIST",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                            transforms.Resize(opt.img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                            ]),
                    )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
        )
        return dataset, dataloader

    def lsun(self, opt):
        dataset = datasets.LSUN(
                        root="./otherGANs/data/lsun",
                        classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(opt.img_size),
                            transforms.CenterCrop(opt.img_size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]),
                    )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True
            )
        return dataset, dataloader

    def celeba(self, opt):
	    dataset = datasets.ImageFolder(root="./otherGANs/data/celeba/",
	                        transform=transforms.Compose([
	                            transforms.Resize(opt.img_size),
	                            transforms.CenterCrop(opt.img_size),
	                            transforms.ToTensor(),
	                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	                        ]))
	    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
	    return dataset, dataloader

    def imagenet(self, opt):
	    dataset = datasets.ImageFolder(root="./otherGANs/data/imagenet/train",
	                        transform=transforms.Compose([
	                            transforms.Resize(opt.img_size),
	                            transforms.CenterCrop(opt.img_size),
	                            transforms.ToTensor(),
	                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	                        ]))
	    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
	    return dataset, dataloader

    def maico2kiku(self, opt):
	    dataset = datasets.ImageFolder(root="./otherGANs/data/maico2kiku/",
	                        transform=transforms.Compose([
	                            transforms.Resize(opt.img_size),
	                            transforms.CenterCrop(opt.img_size),
	                            transforms.ToTensor(),
	                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	                        ]))
	    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=8, shuffle=True)
	    return dataset, dataloader

    def shunga(self, opt):
	    dataset = datasets.ImageFolder(root="./otherGANs/data/shunga/",
	                        transform=transforms.Compose([
	                            transforms.Resize(opt.img_size),
	                            transforms.CenterCrop(opt.img_size),
	                            transforms.ToTensor(),
	                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	                        ]))
	    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
	    return dataset, dataloader

def makeDataloader(opt):
    dataset = Dataset(opt)
    if opt.dataset == 'mnist': return dataset.mnist(opt)
    if opt.dataset == 'cifar10': return dataset.cifar10(opt)
    if opt.dataset == 'fashion': return dataset.fashion(opt)
    if opt.dataset == 'lsun': return dataset.lsun(opt)
    if opt.dataset == 'celeba': return dataset.celeba(opt)
    if opt.dataset == 'imagenet': return dataset.imagenet(opt)
    if opt.dataset == 'maico2kiku': return dataset.maico2kiku(opt)
    if opt.dataset == 'shunga': return dataset.shunga(opt)
