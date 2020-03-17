# flake8: noqa
from catalyst_ext import *
from catalyst.dl import registry
# from catalyst.dl.runner import GanRunner as Runner
from .runners import FullGanRunner as Runner
from . import callbacks, models, transforms
from .experiment import MnistGanExperiment as Experiment
from .phase_callbacks import SmartPhaseManagerCallback

#
from torchvision.models import resnet18
registry.Model(resnet18)

registry.CALLBACKS.add_from_module(callbacks)
registry.MODELS.add_from_module(models)
