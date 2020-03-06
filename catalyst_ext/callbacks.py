import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Dict, List, Union

import torch
from catalyst import utils
from catalyst.core import _State, Callback, CallbackOrder
from catalyst.dl import registry

from metrics import METRICS

logger = logging.getLogger(__name__)


# TODO: remove after upd to newer catalyst
class _MetricCallback(ABC, Callback):
    def __init__(
            self,
            prefix: str,
            input_key: Union[str, List[str], Dict[str, str]] = "targets",
            output_key: Union[str, List[str], Dict[str, str]] = "logits",
            multiplier: float = 1.0,
            **metrics_kwargs,
    ):
        super().__init__(order=CallbackOrder.Metric)
        self.prefix = prefix
        self.input_key = input_key
        self.output_key = output_key
        self.multiplier = multiplier
        self.metrics_kwargs = metrics_kwargs

        self._get_input = utils.get_dictkey_auto_fn(self.input_key)
        self._get_output = utils.get_dictkey_auto_fn(self.output_key)
        kv_types = (dict, tuple, list, type(None))

        is_value_input = \
            isinstance(self.input_key, str) and self.input_key != "__all__"
        is_value_output = \
            isinstance(self.output_key, str) and self.output_key != "__all__"
        is_kv_input = \
            isinstance(self.input_key, kv_types) or self.input_key == "__all__"
        is_kv_output = (
                isinstance(self.output_key, kv_types)
                or self.output_key == "__all__"
        )

        # @TODO: fix to only KV usage
        if hasattr(self, "_compute_metric"):
            pass  # overridden in descendants
        elif is_value_input and is_value_output:
            self._compute_metric = self._compute_metric_value
        elif is_kv_input and is_kv_output:
            self._compute_metric = self._compute_metric_key_value
        else:
            raise NotImplementedError()

    @property
    @abstractmethod
    def metric_fn(self):
        pass

    def _compute_metric_value(self, state: _State):
        output = self._get_output(state.output, self.output_key)
        input = self._get_input(state.input, self.input_key)

        metric = self.metric_fn(output, input, **self.metrics_kwargs)
        return metric

    def _compute_metric_key_value(self, state: _State):
        output = self._get_output(state.output, self.output_key)
        input = self._get_input(state.input, self.input_key)

        metric = self.metric_fn(**output, **input, **self.metrics_kwargs)
        return metric

    def on_batch_end(self, state: _State):
        """
        Computes the metric and add it to batch metrics
        """

        metric = self._compute_metric(state) * self.multiplier
        state.metric_manager.add_batch_value(name=[self.prefix], value=metric)


# TODO: remove after upd to newer catalyst
class MetricCallback(_MetricCallback):
    """
    A callback that returns single metric on `state.on_batch_end`
    """

    def __init__(
            self,
            prefix: str,
            metric_fn: Callable,
            input_key: str = "targets",
            output_key: str = "logits",
            multiplier: float = 1.0,
            **metric_kwargs,
    ):
        super().__init__(
            prefix=prefix,
            input_key=input_key,
            output_key=output_key,
            multiplier=multiplier,
            **metric_kwargs,
        )
        self.metric = metric_fn

    @property
    def metric_fn(self):
        return self.metric


@registry.Callback
class MemoryMetricCallback(MetricCallback):

    def __init__(self, prefix: str, metric_fn_params: Dict[str, str],
                 input_key: str = "targets", output_key: str = "logits",
                 multiplier: float = 1.0, **metric_kwargs):
        metric_fn = METRICS.get(**metric_fn_params)()  # TODO: why () needed? how to avoid?

        super().__init__(prefix=prefix, metric_fn=metric_fn,
                         input_key=input_key, output_key=output_key,
                         multiplier=multiplier,
                         **metric_kwargs)

    def _compute_metric_value(self, state: _State):
        output = self._get_output(state.memory.output, self.output_key)
        input = self._get_input(state.memory.input, self.input_key)

        metric = self.metric_fn(output, input, **self.metrics_kwargs)
        return metric

    def _compute_metric_key_value(self, state: _State):
        output = self._get_output(state.memory.output, self.output_key)
        input = self._get_input(state.memory.input, self.input_key)

        metric = self.metric_fn(**output, **input, **self.metrics_kwargs)
        return metric

    def on_batch_end(self, state: _State):
        pass  # do nothing!!!

    def on_loader_end(self, state: _State):
        metric = self._compute_metric(state) * self.multiplier
        state.metric_manager.epoch_values[state.loader_name][self.prefix] = metric


@registry.Callback
class MemoryAccumulatorCallback(Callback):
    class Memory:  # TODO: move somewhere
        def __init__(self):
            self.input = defaultdict(list)
            self.output = defaultdict(list)

    def __init__(self, input_key=None, output_key=None, memory_size=2000):
        super().__init__(order=CallbackOrder.Metric - 1)
        if not isinstance(input_key, str):
            raise NotImplementedError()
        if not isinstance(output_key, str):
            raise NotImplementedError()
        self.input_key = input_key
        self.output_key = output_key
        self.memory_size = memory_size

    def on_loader_start(self, state: _State):
        state.memory = MemoryAccumulatorCallback.Memory()  # empty memory

    def on_loader_end(self, state: _State):
        for memory in (state.memory.input, state.memory.output):
            for key, values_list in memory.items():
                assert len(values_list) > 0
                assert isinstance(values_list[0], torch.Tensor)
                memory[key] = torch.stack(values_list, dim=0)

    def on_batch_end(self, state: _State):
        self.add_to_memory(
            memory=state.memory.input[self.input_key],
            items=state.input[self.input_key],
            max_memory_size=self.memory_size
        )
        self.add_to_memory(
            memory=state.memory.output[self.output_key],
            items=state.output[self.output_key],
            max_memory_size=self.memory_size
        )

    @staticmethod
    def add_to_memory(memory, items, max_memory_size=2000):
        # TODO: save last/random N items (now first items are saved only)

        if not isinstance(items, (list, tuple, torch.Tensor)):
            # must be iterable
            raise NotImplementedError()

        if len(memory) < max_memory_size:
            if len(memory) + len(items) > max_memory_size:
                items = items[:(max_memory_size - len(memory))]
            memory.extend(items)
