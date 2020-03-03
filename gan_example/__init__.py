# flake8: noqa
from catalyst_ext import *
from catalyst.dl import registry
from catalyst.dl.runner import GanRunner as Runner
from . import callbacks, models, transforms
from .experiment import MnistGanExperiment as Experiment

registry.CALLBACKS.add_from_module(callbacks)
registry.MODELS.add_from_module(models)
