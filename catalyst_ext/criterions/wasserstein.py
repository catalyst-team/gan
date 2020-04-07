from torch import nn


# Wasserstein losses


class WassersteinLossGenerator(nn.Module):

    def forward(self, fake_logits):
        return -fake_logits.mean()


class WassersteinLossDiscriminator(nn.Module):

    def forward(self, fake_logits, real_logits):
        return fake_logits.mean() - real_logits.mean()
