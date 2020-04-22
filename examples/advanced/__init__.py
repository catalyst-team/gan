# flake8: noqa
import sys
sys.path.insert(-1, '.')  # TODO: remove this terrible workaround
import catalyst_gan
from catalyst.dl import registry
# from catalyst.dl.runner import GanRunner as Runner
from .runners import FullGanRunner as Runner
from . import callbacks, models
from .experiment import Experiment as Experiment
from .phase_callbacks import SmartPhaseManagerCallback

registry.CALLBACKS.add_from_module(callbacks)
registry.MODELS.add_from_module(models)
