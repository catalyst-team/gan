import torch
from torch import nn


# Hinge losses


class HingeLossGenerator(nn.Module):

    def forward(self, fake_logits):
        return -fake_logits.mean()


class HingeLossDiscriminator(nn.Module):

    def forward(self, fake_logits, real_logits):
        loss = torch.relu(1.0 - real_logits).mean() + \
               torch.relu(1.0 + fake_logits).mean()
        return loss


class HingeLossDiscriminatorReal(nn.Module):

    def forward(self, real_logits):
        loss = torch.relu(1.0 - real_logits).mean()
        return loss


class HingeLossDiscriminatorFake(nn.Module):

    def forward(self, fake_logits):
        loss = torch.relu(1.0 + fake_logits).mean()
        return loss
