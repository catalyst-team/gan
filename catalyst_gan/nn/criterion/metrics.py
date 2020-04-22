import torch
from torch import nn


# Metrics


class AverageValue(nn.Module):

    def __init__(self, activation=None, activation_params=None):
        super().__init__()
        self.activation_params = activation_params or {}
        assert isinstance(self.activation_params, dict)
        if activation is not None:
            if isinstance(activation, str):
                activation = getattr(torch, activation)
        self.activation = activation

    def forward(self, tensor):
        if self.activation is not None:
            tensor = self.activation(tensor, **self.activation_params)
        return tensor.mean()


class AverageProbability(AverageValue):

    def __init__(self, activation='sigmoid',
                 activation_params=None):
        super().__init__(activation=activation,
                         activation_params=activation_params)
