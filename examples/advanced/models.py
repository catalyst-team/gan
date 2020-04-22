# flake8: noqa
import numpy as np

import torch
import torch.nn as nn


from catalyst.dl import registry


@registry.Module
class Reshape(nn.Module):
    def __init__(self, shape=None, batch_shape=None):
        super().__init__()
        assert (shape is None) ^ (batch_shape is None)
        self.shape = shape
        self.batch_shape = batch_shape

    def forward(self, x):
        if self.shape is not None:
            return x.reshape(x.size(0), *self.shape)
        else:
            return x.reshape(*self.batch_shape)


def fc_generator(n_in, n_out=28*28, n_hidden=100, hidden_multiplier=1):
    c, k = n_hidden, hidden_multiplier
    return nn.Sequential(
        nn.Linear(n_in, c),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(c, c * k),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(c * k, c * k ** 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(c * k ** 2, n_out),
        nn.Tanh()
    )


def fc_discriminator(n_in=28*28, n_out=1, n_hidden=100, hidden_multiplier=1):
    c, k = n_hidden, hidden_multiplier
    return nn.Sequential(
        nn.Linear(n_in, c * k ** 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(c * k ** 2, c * k),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(c * k, c),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(c, n_out)
    )


def conv_generator(n_in, ch_out=1, n_hidden=32, hidden_multiplier=2):
    c, k = n_hidden, hidden_multiplier
    return nn.Sequential(
        nn.Linear(n_in, c*k**3 * 4*4),
        nn.ReLU(inplace=True),
        Reshape(shape=(c*k**3, 4, 4)),
        nn.ConvTranspose2d(c*k**3, c*k**2, 3, 2, 1),
        nn.BatchNorm2d(c*k**2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(c*k**2, c*k, 4, 2, 1),
        nn.BatchNorm2d(c*k),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(c*k, c, 4, 2, 1),
        nn.BatchNorm2d(c),
        nn.ReLU(inplace=True),
        nn.Conv2d(c, ch_out, 1, 1, 0),
        nn.Tanh()
    )


def conv_discriminator(n_in, ch_out=1, n_hidden=32, hidden_multiplier=2):
    c, k = n_hidden, hidden_multiplier
    return nn.Sequential(
        nn.Conv2d(n_in, c, 4, 2, 1),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(c, c * k, 4, 2, 1),
        nn.BatchNorm2d(c * k),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(c * k, c * k**2, 3, 2, 1),#4
        nn.BatchNorm2d(c * k**2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(c * k**2, c * k**3, 4, 1, 0),
        nn.BatchNorm2d(c * k**3),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(c * k**3, ch_out, 1, 1, 0),
        Reshape(shape=(ch_out, ))
    )


class SimpleGenerator(nn.Module):
    """Simple fully-connected generator"""
    def __init__(
        self,
        noise_dim=10,
        image_resolution=(28, 28),
        channels=1,
        conv_mode=True,
        n_hidden=32,
        hidden_multiplier=2
    ):
        """

        :param noise_dim:
        :param image_resolution:
        :param channels:
        :param conv_mode:
        """
        super().__init__()
        self.noise_dim = noise_dim
        self.image_resolution = image_resolution
        self.channels = channels
        self.is_mode_conv = conv_mode

        if self.is_mode_conv:
            self.net = conv_generator(
                n_in=self.noise_dim,
                ch_out=channels,
                n_hidden=n_hidden,
                hidden_multiplier=hidden_multiplier
            )
        else:
            self.net = fc_generator(
                n_in=self.noise_dim,
                n_out=int(channels * np.prod(image_resolution)),
                n_hidden=n_hidden,
                hidden_multiplier=hidden_multiplier
            )

    def forward(self, x):
        x = self.net(x)
        x = x.reshape(x.size(0), self.channels, *self.image_resolution)
        return x


class SimpleDiscriminator(nn.Module):
    def __init__(self, image_resolution=(28, 28), channels=1, conv_mode=True,
                 n_hidden=32, hidden_multiplier=2):
        super().__init__()
        self.image_resolution = image_resolution
        self.channels = channels
        self.is_mode_conv = conv_mode

        if self.is_mode_conv:
            self.net = conv_discriminator(
                n_in=channels,
                ch_out=1,
                n_hidden=n_hidden,
                hidden_multiplier=hidden_multiplier
            )
        else:
            self.net = fc_discriminator(
                n_in=int(np.prod(image_resolution) * channels),
                n_out=1,
                n_hidden=n_hidden,
                hidden_multiplier=hidden_multiplier
            )

    def forward(self, x):
        if not self.is_mode_conv:
            x = x.reshape(x.size(0), -1)
        x = self.net(x)
        return x


class SimpleCGenerator(SimpleGenerator):
    def __init__(
        self,
        noise_dim=10,
        num_classes=10,
        image_resolution=(28, 28),
        channels=1,
        conv_mode=True,
        n_hidden=32,
        hidden_multiplier=2
    ):
        super().__init__(
            noise_dim=noise_dim + num_classes,
            image_resolution=image_resolution,
            channels=channels,
            conv_mode=conv_mode,
            n_hidden=n_hidden,
            hidden_multiplier=hidden_multiplier
        )
        self.num_classes = num_classes

    def forward(self, z, c_one_hot):
        x = torch.cat((z, c_one_hot.float()), dim=1)
        return super().forward(x)


class SimpleCDiscriminator(nn.Module):
    def __init__(
        self,
        num_classes=10,
        image_resolution=(28, 28),
        channels=1,
        conv_mode=True,
        n_hidden=32,
        hidden_multiplier=2
    ):
        super().__init__()
        self.image_resolution = image_resolution
        self.channels = channels

        self.is_mode_conv = conv_mode

        if self.is_mode_conv:
            n_emb = 64
            self.net = conv_discriminator(
                n_in=channels,
                ch_out=n_emb,
                n_hidden=n_hidden,
                hidden_multiplier=hidden_multiplier
            )
            self.classifier = nn.Sequential(
                nn.Linear(n_emb + num_classes, 64),
                nn.LeakyReLU(0.2),
                nn.Linear(64, 1)
            )
        else:
            self.net = fc_discriminator(
                n_in=int(np.prod(image_resolution) * channels + num_classes),
                n_out=1,
                n_hidden=n_hidden,
                hidden_multiplier=hidden_multiplier
            )
            self.classifier = None

    def forward(self, x, c_one_hot):
        if self.is_mode_conv:
            x = self.net(x)
            x = x.reshape(x.size(0), -1)
            x = torch.cat((x, c_one_hot.float()), dim=1)
            x = self.classifier(x)
        else:
            x = x.reshape(x.size(0), -1)
            x = torch.cat((x, c_one_hot.float()), dim=1)
            x = self.net(x)
        return x


# scitator models https://colab.research.google.com/github/catalyst-team/catalyst/blob/master/examples/notebooks/demo.ipynb


class ScitatorGenerator(nn.Module):
    def __init__(self, noise_dim, image_resolution=(28, 28), channels=1):
        super().__init__()
        self.img_shape = image_resolution
        self.channels = 1

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(noise_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(image_resolution))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), self.channels, *self.img_shape)
        return img


class ScitatorDiscriminator(nn.Module):
    def __init__(self, image_resolution=(28, 28)):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(image_resolution)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity


class ScitatorCGenerator(ScitatorGenerator):
    def __init__(
        self,
        noise_dim=10,
        num_classes=10,
        image_resolution=(28, 28)
    ):
        super().__init__(
            noise_dim + num_classes, image_resolution
        )
        self.num_classes = num_classes

    def forward(self, z, c_one_hot):
        x = torch.cat((z, c_one_hot.float()), dim=1)
        return super().forward(x)


class ScitatorCDiscriminator(nn.Module):
    def __init__(
        self,
        num_classes=10,
        image_resolution=(28, 28),
        channels=1,
        hidden_dim=100
    ):
        super().__init__()
        self.image_resolution = image_resolution
        self.channels = channels

        self.embedder = nn.Sequential(
            nn.Linear(int(np.prod(image_resolution)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, hidden_dim)
        )
        self.classifier = nn.Linear(hidden_dim + num_classes, 1)

    def forward(self, x, c_one_hot):
        x = self.embedder(x.reshape(x.size(0), -1))
        x = self.classifier(torch.cat((x, c_one_hot.float()), dim=1))
        return x


class DCGanGenerator(nn.Module):
    def __init__(self, noise_dim=100,
                 image_resolution=(64, 64),
                 channels=3,
                 hidden_dim=64,
                 ):
        super().__init__()
        assert tuple(image_resolution) == (64, 64)
        self.main = nn.Sequential(
            Reshape(shape=(noise_dim, 1, 1)),
            # input is Z, going into a convolution
            nn.ConvTranspose2d(noise_dim, hidden_dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(hidden_dim, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, x):
        return self.main(x)


class DCGanDiscriminator(nn.Module):
    def __init__(self, image_resolution=(64, 64),
                 channels=3,
                 hidden_dim=64):
        super().__init__()
        assert tuple(image_resolution) == (64, 64)
        ndf = hidden_dim
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
            Reshape(shape=(1,))
        )

    def forward(self, x):
        return self.main(x)


__all__ = [
    "SimpleGenerator", "SimpleDiscriminator", "SimpleCGenerator",
    "SimpleCDiscriminator", "ScitatorGenerator", "ScitatorDiscriminator",
    "ScitatorCGenerator", "ScitatorCDiscriminator",
    "DCGanDiscriminator", "DCGanGenerator"
]
