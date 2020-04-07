from .mnist import mnist
from .inception import InceptionV3

from torch_mimicry.nets import sngan, ssgan, infomax_gan
from catalyst.dl import registry
registry.MODELS.add_from_module(sngan, prefix='sngan.')
registry.MODELS.add_from_module(ssgan, prefix='ssgan.')
registry.MODELS.add_from_module(infomax_gan, prefix='infomax_gan.')
