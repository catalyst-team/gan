import torch
from torch import nn


# BCE losses


class BCELossGenerator(nn.BCEWithLogitsLoss):

    def __init__(self, target=1.0, **kwargs):
        assert 0 <= target <= 1
        self.target = target
        super().__init__(**kwargs)

    def forward(self, fake_logits):
        target = self.target * torch.ones_like(fake_logits)
        return super().forward(fake_logits, target)


class BCELossDiscriminator(nn.BCEWithLogitsLoss):

    def __init__(self, target_fake=0, target_real=1, **kwargs):
        assert 0 <= target_real <= 1
        assert 0 <= target_fake <= 1
        super().__init__(**kwargs)
        self.target_real = target_real
        self.target_fake = target_fake

    def forward(self, fake_logits, real_logits):
        fake_target = torch.ones_like(fake_logits) * self.target_fake
        real_target = torch.ones_like(real_logits) * self.target_real

        real_loss = super().forward(fake_logits, fake_target)
        fake_loss = super().forward(real_logits, real_target)
        return fake_loss + real_loss
