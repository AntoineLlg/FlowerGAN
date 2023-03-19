import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self, img_size=64, ndf=64, nc=3):
        """

        :param img_size: size of the images of input (square images expected) (default=64)
        :param ndf: number of feature maps in the first layer (default=64)
        :param nc: number of color channels of input (default=3)
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1,
                      bias=False),  # each image has shape (nc=3, img_size/2, img_size/2)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, int(img_size / 16), 1, 0, bias=False),

            nn.Sigmoid())

    def forward(self, X):
        return self.main(X)

    def save(self):
        torch.save(self.state_dict(), "saved_models/flower_discriminator_weights.pth")


class Generator(nn.Module):
    def __init__(self, nz=50, ngf=128, img_size=64, nc=3):
        """

        :param nz: dimension of latent space of input (default=50)
        :param img_size: size of the images of input (square images expected) (default=64)
        :param ngf: number of feature maps in the second to last layer (default=128)
        :param nc: number of color channels of input (default=3)
        """
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz,
                               ngf * 4,
                               int(img_size / 8),
                               1,
                               0,
                               bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, X):
        return self.main(X)

    def save(self):
        torch.save(self.state_dict(), "saved_models/flower_generator_weights.pth")
