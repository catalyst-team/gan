# flake8: noqa
import sys
sys.path.insert(-1, '.')  # TODO: remove this terrible workaround
from catalyst.dl import registry
from catalyst_gan.core.runner import GanRunner as Runner
from . import callbacks, models, transforms
from .experiment import MnistGanExperiment as Experiment

registry.CALLBACKS.add_from_module(callbacks)
registry.MODELS.add_from_module(models)
