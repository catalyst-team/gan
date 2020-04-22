import copy
from typing import Dict, List  # isort:skip
from collections import OrderedDict

from catalyst.core import State
from catalyst.dl import registry

from catalyst_gan.callbacks import PhaseManagerCallback


# TODO: remove copy-paste from catalyst.dl.callback.phase
class Phase:
    """
    Class for storing information about certain phase, including
    - phase name
    - number of steps (batches) in phase before next phase is chosen
    - how many steps (batches) are done already
    """
    def __init__(self, name: str = None, steps: int = None,
                 batch_metric_key: str = None,
                 threshold: float = None,
                 alpha: float = 1.0,
                 greater_is_good: bool = None,
                 do_abs_metric: bool = False):
        self.steps = int(steps) if steps is not None else None
        assert 1e-9 < alpha <= 1
        self.alpha = alpha
        self.curr_step = 0
        self.name = name

        self.batch_metric_key = batch_metric_key
        self.threshold = threshold
        self.greater_is_good = greater_is_good

        self.do_abs_metric = do_abs_metric

        self._prev_metric_value = None

    def step(self, state: State):
        metric_value = state.prev_batch_metrics.get(self.batch_metric_key, None)
        if metric_value is None:
            return False
        if self.do_abs_metric:
            metric_value = abs(metric_value)
        if self._prev_metric_value is not None:
            metric_value = (
                self._prev_metric_value * (1 - self.alpha)
                + metric_value * self.alpha
            )
        self._prev_metric_value = metric_value

        is_greater = metric_value > self.threshold
        do_step = (not is_greater) and (not self.greater_is_good)
        do_step = do_step or (is_greater and self.greater_is_good)
        if do_step:
            self.curr_step = (self.curr_step + 1) % self.steps
            phase_finished = self.curr_step == 0
            if phase_finished:
                self._prev_metric_value = None
            return phase_finished
        else:
            return False


# TODO: remove copy-paste from catalyst.dl.callback.phase
class PhaseManager:
    """
    Class for storing & managing all phases in experiment configuration

    Stores separately current phases in train & validation modes

    By calling `.step(...)` method current phase is updated by step-size
    and if current phase is finished, the next phase becomes current
    """
    def __init__(self, train_phases: List[Phase], valid_phases: List[Phase]):
        self.train_phases = train_phases
        self.valid_phases = valid_phases

        self.train_index = 0
        self.valid_index = 0

    def step(self, state: State, step_size: int = 1):
        assert step_size == 1
        if state.is_train_loader:
            if len(self.train_phases) > 1:
                need_change_phase = self.train_phases[self.train_index].step(state)
                if need_change_phase:
                    self.train_index = \
                        (self.train_index + 1) % len(self.train_phases)
        else:
            if len(self.valid_phases) > 1:
                need_change_phase = self.valid_phases[self.valid_index].step(state)
                if need_change_phase:
                    self.valid_index = \
                        (self.valid_index + 1) % len(self.valid_phases)

    def get_phase_name(self, state: State):
        if state.is_train_loader:
            return self.train_phases[self.train_index].name
        return self.valid_phases[self.valid_index].name


@registry.Callback
class SmartPhaseManagerCallback(PhaseManagerCallback):

    def __init__(self, train_phases: "OrderedDict[str, int]" = None,
                 valid_phases: "OrderedDict[str, int]" = None,
                 valid_mode: str = None):
        super().__init__(train_phases, valid_phases, valid_mode)
        self._curr_phase_steps = 0

    def _get_phase_manager(
            self,
            train_phases: "OrderedDict[str, Dict]" = None,
            valid_phases: "OrderedDict[str, Dict]" = None,
            valid_mode: str = None
    ):
        assert (valid_phases is None) ^ (valid_mode is None), \
            "Exactly one of them must be specified"

        if train_phases is None:
            train_phases = [Phase(name=None, steps=None)]
        else:
            train_phases = [
                Phase(name=name, **params)
                for name, params in train_phases.items()
            ]

        if valid_phases is None:
            if valid_mode == self.VALIDATION_MODE_ALL:
                valid_phases = [Phase(name=None, steps=None)]
            elif valid_mode == self.VALIDATION_MODE_SAME:
                valid_phases = copy.deepcopy(train_phases)
            else:
                raise ValueError(
                    f"Unsupported validation_mode, should be one of "
                    f"{self.allowed_valid_modes}"
                )

        return PhaseManager(
            train_phases=train_phases, valid_phases=valid_phases
        )

    def on_batch_start(self, state: State):
        super().on_batch_start(state)

    def on_batch_end(self, state: State):
        super().on_batch_end(state)
        if state.is_train_loader:
            self._curr_phase_steps += 1
            if state.phase != self.phase_manager.get_phase_name(state):
                state.batch_metrics[f"phase_steps/{state.phase}"] = \
                    self._curr_phase_steps
                self._curr_phase_steps = 0
