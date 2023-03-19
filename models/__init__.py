import torch
from .neuralnetworks import *
from .utils import *


def getDiscriminator(pretrained=True, **kwargs):
    netD = Discriminator(**kwargs)
    if pretrained:
        try:
            netD.load_state_dict(torch.load('saved_models/flower_discriminator_weights.pth'))
        except RuntimeError:
            raise RuntimeError("Network shape does not match last recorded weights at flower_discriminator_weights.pth")

    return netD


def getGenerator(pretrained=True, **kwargs):
    netG = Generator(**kwargs)
    if pretrained:
        try:
            netG.load_state_dict(torch.load('saved_models/flower_generator_weights.pth'))
        except RuntimeError:
            raise RuntimeError("Network shape does not match last recorded weights at flower_generator_weights.pth")

    return netG
