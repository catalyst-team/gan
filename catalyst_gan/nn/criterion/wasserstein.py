from torch import nn


# Wasserstein losses


class WassersteinLossGenerator(nn.Module):

    def forward(self, fake_logits):
        return -fake_logits.mean()


class WassersteinLossDiscriminator(nn.Module):

    def forward(self, fake_logits, real_logits):
        return fake_logits.mean() - real_logits.mean()


class WassersteinLossDiscriminatorReal(nn.Module):

    def forward(self, real_logits):
        return -real_logits.mean()


class WassersteinLossDiscriminatorFake(nn.Module):

    def forward(self, fake_logits):
        return fake_logits.mean()


class WassersteinDistance(nn.Module):

    def forward(self, fake_validity, real_validity):
        return real_validity.mean() - fake_validity.mean()
