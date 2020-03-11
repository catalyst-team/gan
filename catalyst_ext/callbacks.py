import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Dict, List, Union

import torch
import torch.nn.functional as F
import numpy as np
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

    def __init__(self, prefix: str,
                 metric: Union[str, Dict[str, str]],
                 memory_key: Union[str, List[str], Dict[str, str]],
                 multiplier: float = 1.0, **metric_kwargs):
        if isinstance(metric, str):
            metric = {"metric": metric}
        elif isinstance(metric, dict):
            pass
        else:
            raise NotImplementedError()
        metric_fn = METRICS.get_from_params(**metric)

        self.memory_key = memory_key
        self._get_memory = utils.get_dictkey_auto_fn(self.memory_key)
        if isinstance(self.memory_key, str):
            self._compute_metric = self._compute_metric_value
        elif isinstance(self.memory_key, (list, tuple, dict)):
            self._compute_metric = self._compute_metric_key_value
        else:
            raise NotImplementedError()

        super().__init__(prefix=prefix, metric_fn=metric_fn,
                         multiplier=multiplier,
                         **metric_kwargs)

    def _compute_metric_value(self, state: _State):
        output = self._get_memory(state.memory, self.memory_key)

        metric = self.metric_fn(output, **self.metrics_kwargs)
        return metric

    def _compute_metric_key_value(self, state: _State):
        output = self._get_memory(state.memory, self.memory_key)

        metric = self.metric_fn(**output, **self.metrics_kwargs)
        return metric

    def on_batch_end(self, state: _State):
        pass  # do nothing (override parent method which does something)

    def on_loader_end(self, state: _State):
        with torch.no_grad():
            metric = self._compute_metric(state) * self.multiplier
            if isinstance(metric, torch.Tensor):
                metric = metric.item()
        state.metric_manager.epoch_values[state.loader_name][self.prefix] = metric


@registry.Callback
class MemoryAccumulatorCallback(Callback):

    SAVE_FIRST_MODE = "first"
    SAVE_LAST_MODE = "last"
    SAVE_RANDOM_MODE = "random"

    def __init__(self, input_key: Dict[str, str], output_key: Dict[str, str],
                 mode: str = SAVE_LAST_MODE,
                 memory_size: int = 2000):
        super().__init__(order=CallbackOrder.Metric)  # todo
        if not isinstance(input_key, dict):
            raise NotImplementedError()
        if not isinstance(output_key, dict):
            raise NotImplementedError()
        self.input_key = input_key
        self.output_key = output_key
        self.memory_size = memory_size

        if mode not in (
                self.SAVE_FIRST_MODE,
                self.SAVE_LAST_MODE,
                self.SAVE_RANDOM_MODE):
            raise ValueError(f"Unsupported mode '{mode}'")
        self.mode = mode

        self._start_idx = None

    def on_loader_start(self, state: _State):
        state.memory = defaultdict(list)  # empty memory

        # self._start_idx[key] - index in state.memory[key]
        # where to update memory next
        self._start_idx = defaultdict(int)  # default value 0

    def on_loader_end(self, state: _State):
        for key, values_list in state.memory.items():
            assert len(values_list) > 0
            assert isinstance(values_list[0], torch.Tensor)
            state.memory[key] = torch.stack(values_list, dim=0)

    def on_batch_end(self, state: _State):
        for input_key, memory_key in self.input_key.items():
            self.add_to_memory(
                memory=state.memory[memory_key],
                items=state.input[input_key],
                memory_key=memory_key,
                max_memory_size=self.memory_size
            )
        for output_key, memory_key in self.output_key.items():
            self.add_to_memory(
                memory=state.memory[memory_key],
                items=state.output[output_key],
                memory_key=memory_key,
                max_memory_size=self.memory_size
            )

    def add_to_memory(self, memory, items, memory_key,
                      max_memory_size=2000):
        if not isinstance(items, (list, tuple, torch.Tensor)):
            # must be iterable
            raise NotImplementedError()

        if len(memory) < max_memory_size:
            n_remaining_items = len(memory) + len(items) - max_memory_size
            if n_remaining_items > 0:
                memory.extend(items[:-n_remaining_items])
                items = items[-n_remaining_items:]
            else:
                memory.extend(items)
                return  # everything added

        if self.mode == self.SAVE_FIRST_MODE:
            pass
        elif self.mode == self.SAVE_LAST_MODE:
            self._start_idx[memory_key] = self._store_all(
                memory, items, start_index=self._start_idx[memory_key]
            )
        elif self.mode == self.SAVE_RANDOM_MODE:
            memory_indices = np.random.choice(len(memory), size=len(items))
            for idx, item in zip(memory_indices, items):
                memory[idx] = item
        else:
            raise ValueError(f"Unsupported mode '{self.mode}'")

    @staticmethod
    def _store_all(memory, items, start_index=0):
        index = start_index
        for item in items:
            memory[index] = item
            index = (index + 1) % len(memory)
        return index


@registry.Callback
class MemoryFeatureExtractorCallback(Callback):

    def __init__(
        self,
        memory_key: str,
        model_key: str,
        layer_key: Dict[str, str] = None,
        batch_size: int = 64,
        channels: int = 3,
        target_size: int = None,
        interpolation_mode: str = 'nearest',
        align_corners: bool = None
    ):
        super().__init__(order=CallbackOrder.Internal)
        self.memory_key = memory_key

        self.model_key = model_key

        assert isinstance(layer_key, dict)
        self.layer_key = layer_key

        self.batch_size = batch_size
        assert channels in (1, 3), "suspicious number of image channels"
        self.channels = channels

        self.need_resize = target_size is not None
        self.resize_params = {
            "size": target_size,
            "mode": interpolation_mode,
            "align_corners": align_corners
        }

        self._extracted_features = {}

    def on_loader_end(self, state: _State):
        # st.memory[out_key[s]] = model(st.memory[memory_key])
        # 1. register hooks
        model = state.model[self.model_key]
        hook_handles = self._wrap_model(model)

        # 2. apply model to extract features
        model_training_old = model.training
        model.eval()
        with torch.no_grad():
            data = state.memory[self.memory_key]
            for start_idx in range(0, len(data), self.batch_size):
                batch = data[start_idx:start_idx + self.batch_size]

                self._process_batch(
                    model=model,
                    batch=batch,
                    memory=state.memory
                )
        model.train(mode=model_training_old)
        # 3. remove handles
        for handle in hook_handles:
            handle.remove()

    def _prepare_batch_input(self, batch):
        if isinstance(batch, list):
            batch = torch.stack(batch, dim=0)
        elif isinstance(batch, torch.Tensor):
            pass
        else:
            raise NotImplementedError()

        channels = batch.size(1)
        if channels != self.channels:
            if channels == 1:
                batch = torch.cat([batch] * self.channels, dim=1)
            elif self.channels == 1:
                batch = torch.mean(batch, dim=1, keepdim=True)
            else:
                raise NotImplementedError()

        if self.need_resize:
            batch = F.interpolate(batch, **self.resize_params)
        return batch

    def _process_batch(self, model, batch, memory):
        batch = self._prepare_batch_input(batch=batch)

        model(batch)  # now data is in self._extracted_features

        for feature_name, features in self._extracted_features.items():
            assert features.size(0) == batch.size(0)
            memory[feature_name].extend(
                features
            )
        # sanity check & clean up
        self._extracted_features = {}

    def _wrap_model(self, model):
        handles = []
        for module_key, output_params in self.layer_key.items():
            module = self._get_module(model, module_key)
            output_key, activation, activation_params = \
                self._parse_layer_output_params(output_params)

            def fwd_hook(module, input, output,
                         output_key=output_key,
                         activation=activation,
                         activation_params=activation_params):
                self._extracted_features[output_key] = \
                    activation(output, **activation_params)

            handle = module.register_forward_hook(fwd_hook)
            handles.append(handle)
        return handles

    @staticmethod
    def _get_module(model, path):
        if path == '':
            return model

        curr = model
        for attrib_name in path.split('.'):
            # prev = curr
            curr = getattr(curr, attrib_name)
        return curr

    @staticmethod
    def _parse_layer_output_params(params):
        # return output_key, activation, activation_params
        if isinstance(params, str):
            output_key = params
            activation = lambda x: x
            activation_params = {}
        elif isinstance(params, dict):
            assert len(params) <= 2
            output_key = params.get("memory_out_key", None)
            assert output_key is not None

            activation_params = params.get("activation", None).copy()

            activation = activation_params.pop("name")
            if not hasattr(torch, activation):
                raise ValueError(f"unknown activation '{activation}'")
            activation = getattr(torch, activation)
        else:
            raise NotImplementedError()
        return output_key, activation, activation_params
