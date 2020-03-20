# flake8: noqa
import numpy as np

import torch
import torch.nn as nn

# TODO: add conv models


class SimpleGenerator(nn.Module):
    """Simple fully-connected generator"""
    def __init__(
        self,
        noise_dim=10,
        hidden_dim=256,
        image_resolution=(28, 28),
        channels=1
    ):
        """

        :param noise_dim:
        :param hidden_dim:
        :param image_resolution:
        :param channels:
        """
        super().__init__()
        self.noise_dim = noise_dim
        self.image_resolution = image_resolution
        self.channels = channels

        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim), nn.LeakyReLU(0.05),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.05),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.05),
            nn.Linear(hidden_dim, np.prod(image_resolution)), nn.Tanh()
        )

    def forward(self, x):
        x = self.net(x)
        x = x.reshape(x.size(0), self.channels, *self.image_resolution)
        return x


class SimpleDiscriminator(nn.Module):
    def __init__(self, image_resolution=(28, 28), channels=1, hidden_dim=100):
        super().__init__()
        self.image_resolution = image_resolution
        self.channels = channels

        self.net = nn.Sequential(
            nn.Linear(channels * np.prod(image_resolution), hidden_dim),
            nn.LeakyReLU(0.05), nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.05), nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.05), nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.05), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x = self.net(x.reshape(x.size(0), -1))
        return x


class SimpleCGenerator(SimpleGenerator):
    def __init__(
        self,
        noise_dim=10,
        num_classes=10,
        hidden_dim=256,
        image_resolution=(28, 28),
        channels=1
    ):
        super().__init__(
            noise_dim + num_classes, hidden_dim, image_resolution, channels
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
        hidden_dim=100
    ):
        super().__init__()
        self.image_resolution = image_resolution
        self.channels = channels

        self.embedder = nn.Sequential(
            nn.Linear(channels * np.prod(image_resolution), hidden_dim),
            nn.LeakyReLU(0.05), nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.05), nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.05), nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.05)
        )
        self.classifier = nn.Linear(hidden_dim + num_classes, 1)

    def forward(self, x, c_one_hot):
        x = self.embedder(x.reshape(x.size(0), -1))
        x = self.classifier(torch.cat((x, c_one_hot.float()), dim=1))
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


__all__ = [
    "SimpleGenerator", "SimpleDiscriminator", "SimpleCGenerator",
    "SimpleCDiscriminator", "ScitatorGenerator", "ScitatorDiscriminator",
    "ScitatorCGenerator", "ScitatorCDiscriminator"
]
