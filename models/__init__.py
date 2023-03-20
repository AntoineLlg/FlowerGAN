import torch
from .neuralnetworks import *
from .utils import *


def getDiscriminator(filename=None, **kwargs):
    netD = Discriminator(**kwargs)
    if filename is None:
        pass
    else:
        try:
            netD.load_state_dict(torch.load(filename))
        except RuntimeError:
            raise RuntimeError("Network shape does not match last recorded weights at flower_discriminator_weights.pth")

    return netD


def getGenerator(filename=None, **kwargs):
    netG = Generator(**kwargs)
    if filename is None:
        pass
    else:
        try:
            netG.load_state_dict(torch.load('./saved_models/flower_generator_weights.pth'))
        except RuntimeError:
            raise RuntimeError("Network shape does not match last recorded weights at flower_generator_weights.pth")

    return netG
