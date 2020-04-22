import logging
from collections import defaultdict
from typing import Dict, List, Union, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from catalyst.core import State, Callback, CallbackOrder, MetricCallback
from catalyst.dl import registry
from catalyst.utils import get_dictkey_auto_fn

from catalyst_gan.utils import get_metric, get_batch_transform, get_module
from catalyst_gan import utils

logger = logging.getLogger(__name__)


def _stack_memory_lists_to_tensors(state: State):
    for key, values_list in state.memory.items():
        assert len(values_list) > 0
        if isinstance(values_list, (list, tuple)):
            assert isinstance(values_list[0], torch.Tensor)
            state.memory[key] = torch.stack(values_list, dim=0)


class MemoryMetricCallback(MetricCallback):

    def __init__(self, prefix: str,
                 metric: Union[str, Dict[str, str]],
                 memory_key: Union[str, List[str], Dict[str, str]],
                 multiplier: float = 1.0, **metric_kwargs):
        metric_fn = get_metric(metric)

        self.memory_key = memory_key
        self._get_memory = get_dictkey_auto_fn(self.memory_key)
        if isinstance(self.memory_key, str):
            self._compute_metric = self._compute_metric_value
        elif isinstance(self.memory_key, (list, tuple, dict)):
            self._compute_metric = self._compute_metric_key_value
        else:
            raise NotImplementedError()

        super().__init__(prefix=prefix, metric_fn=metric_fn,
                         multiplier=multiplier,
                         **metric_kwargs)

    def _compute_metric_value(self, state: State):
        output = self._get_memory(state.memory, self.memory_key)

        metric = self.metric_fn(output, **self.metrics_kwargs)
        return metric

    def _compute_metric_key_value(self, state: State):
        output = self._get_memory(state.memory, self.memory_key)

        metric = self.metric_fn(**output, **self.metrics_kwargs)
        return metric

    def on_batch_end(self, state: State):
        pass  # do nothing (override parent method which does something)

    def on_loader_end(self, state: State):
        with torch.no_grad():
            metric = self._compute_metric(state) * self.multiplier
            if isinstance(metric, torch.Tensor):
                metric = metric.item()
        state.loader_metrics[self.prefix] = metric


class MemoryMultiMetricCallback(MemoryMetricCallback):

    def __init__(self, prefix: str, metric: Union[str, Dict[str, str]],
                 suffixes: List,
                 memory_key: Union[str, List[str], Dict[str, str]],
                 multiplier: float = 1.0, **metric_kwargs):
        super().__init__(prefix, metric, memory_key, multiplier,
                         **metric_kwargs)
        self.suffixes = suffixes

    def on_loader_end(self, state: State):
        with torch.no_grad():
            metrics_ = self._compute_metric(state)
            if isinstance(metrics_, torch.Tensor):
                metrics_ = metrics_.detach().cpu().numpy()

        for arg, metric in zip(self.suffixes, metrics_):
            if isinstance(arg, int):
                key = f"{self.prefix}{arg:02}"
            else:
                key = f"{self.prefix}_{arg}"
            state.loader_metrics[key] = metric * self.multiplier


class MemoryAccumulatorCallback(Callback):
    SAVE_FIRST_MODE = "first"
    SAVE_LAST_MODE = "last"
    SAVE_RANDOM_MODE = "random"

    def __init__(self, input_key: Dict[str, str], output_key: Dict[str, str],
                 mode: str = SAVE_LAST_MODE,
                 memory_size: int = 2000):
        super().__init__(order=CallbackOrder.Internal)
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

    def on_loader_start(self, state: State):
        state.memory = defaultdict(list)  # empty memory

        # self._start_idx[key] - index in state.memory[key]
        # where to update memory next
        self._start_idx = defaultdict(int)  # default value 0

    def on_loader_end(self, state: State):
        _stack_memory_lists_to_tensors(state)

    def on_batch_end(self, state: State):
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
                items=state.output[output_key].detach(),
                memory_key=memory_key,
                max_memory_size=self.memory_size
            )

    def add_to_memory(self,
                      memory: List[torch.Tensor],
                      items: Iterable[torch.Tensor],
                      memory_key: Optional[str] = None,
                      max_memory_size: int = 2000):
        """
        :param memory: list to extend
        :param items: sequence of elements to add to `memory`
        :param memory_key: unique placeholder for `memory` list,
            used only if mode == SAVE_LAST_MODE to update the most outdated
            items (FIFO) if current memory size > max_memory_size
        :param max_memory_size:
        :return:
        """
        if not isinstance(items, (list, tuple, torch.Tensor)):
            raise NotImplementedError()

        if len(memory) < max_memory_size:
            n_remaining_items = len(memory) + len(items) - max_memory_size
            if n_remaining_items > 0:
                memory.extend(items[:-n_remaining_items])
                items = items[-n_remaining_items:]
            else:
                memory.extend(items)
                return  # everything added, memory not full

        # memory is full -> update some items if needed

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

    def on_loader_end(self, state: State):
        # st.memory[out_key[s]] = model(st.memory[memory_key])
        # 1. register hooks
        model = state.model[self.model_key]
        hook_handles = self._wrap_model(model)

        # 2. apply model to extract features
        # TODO: move model.eval() -> deeval() wrapping to external score
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

        # convert memory lists to tensors
        _stack_memory_lists_to_tensors(state)

    def _prepare_batch_input(self, batch):
        batch = utils.as_tensor(batch)

        batch = utils.change_num_image_channels(batch,
                                                channels_num_out=self.channels)

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
            module = get_module(model, module_key)
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
    def _parse_layer_output_params(params):
        """
        Returns tuple (output_key, activation, activation_params)
        output_key - memory output key
        activation - activation function to apply
        activation_params
            - params for activation (activation(x, **activation_params))

        See example inputs below
        ```yaml
            params_example_1: "fake_features"
            params_example_2:
              activation:
                name: "softmax"
                dim: -1
              memory_out_key: "fake_probabilities"
        ```
        """
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


class MemoryTransformCallback(Callback):  # todo: generalize

    def __init__(self,
                 batch_transform: Dict[str, str],
                 transform_in_key: Union[str, List[str], Dict[str, str]],
                 transform_out_key: str = None,
                 suffixes: List[str] = None):
        super().__init__(order=CallbackOrder.Internal)
        self.transform_fn = get_batch_transform(batch_transform)

        self._get_input_key = get_dictkey_auto_fn(transform_in_key)
        self.transform_in_key = transform_in_key
        self.transform_out_key = transform_out_key
        self.suffixes = suffixes or []

    def on_loader_end(self, state: State):
        input = self._get_input_key(state.memory, self.transform_in_key)
        output = self.transform_fn(**input)
        if isinstance(output, torch.Tensor):
            state.memory[self.transform_out_key] = output
        elif isinstance(output, (list, tuple)):
            for key, value in zip(self.suffixes, output):
                assert isinstance(value, torch.Tensor)

                if self.transform_out_key is not None:
                    key = f"{self.transform_out_key}_{key}"
                state.memory[key] = value
        else:
            raise NotImplementedError()


class PerceptualPathLengthCallback(MetricCallback):

    def __init__(self, prefix: str,
                 generator_model_key: str,
                 embedder_model_key: str,
                 noise_shape: Union[int, List[int]],
                 interpolation: str = "spherical",
                 num_samples: int = 100_000,
                 eps: float = 1e-4,
                 batch_size: int = 32,
                 multiplier: float = 1.0,
                 **metric_kwargs):
        super().__init__(prefix=prefix,
                         metric_fn=self._metric_fn,
                         multiplier=multiplier,
                         num_samples=num_samples,
                         eps=eps,
                         batch_size=batch_size,
                         **metric_kwargs)
        self.generator_model_key = generator_model_key
        self.embedder_model_key = embedder_model_key
        if interpolation == "spherical":
            self.interpolate = utils.slerp
        elif interpolation == "linear":
            self.interpolate = torch.lerp
        else:
            raise ValueError(f"Unknown interpolation '{interpolation}'")
        if isinstance(noise_shape, int):
            noise_shape = [noise_shape, ]
        self.noise_shape = tuple(noise_shape)

    def _compute_metric(self, state: State):
        generator = state.model[self.generator_model_key]
        embedder = state.model[self.embedder_model_key]

        g_training = generator.training
        e_training = embedder.training
        generator.train(False)
        embedder.train(False)

        metric_value = self._metric_fn(
            generator=generator,
            embedder=embedder,
            **self.metrics_kwargs
        )
        generator.train(g_training)
        embedder.train(e_training)
        return metric_value

    def _metric_fn(self, generator, embedder,
                   batch_size=1,
                   num_samples=100_000, eps=1e-4):
        # todo weighted sum perceptual
        dist = lambda x, y: ((x-y)**2).sum(1)

        device = next(generator.parameters()).device  # todo remove this hack
        get_z = lambda: self._get_z(batch_size=batch_size, device=device)
        get_t = lambda: torch.rand(size=(batch_size,), device=device)
        get_c = lambda: self._get_generator_condition_inputs(batch_size,
                                                             device=device)

        scores = []
        for _ in range(num_samples // batch_size):  # todo: do not ignore last incomplete batch
            z1 = get_z()
            z2 = get_z()
            t = get_t()
            c_args = get_c()
            e1 = embedder(generator(self.interpolate(z1, z2, t), *c_args))
            e2 = embedder(generator(self.interpolate(z1, z2, t + eps), *c_args))
            d = dist(e1, e2).mean().item() / eps ** 2
            scores.append(d)
        return np.mean(scores)  # todo (see above todo): then fixed this line will be formally incorrect

    def _get_z(self, batch_size, device=None):
        return torch.normal(0, 1,
                            size=(batch_size, ) + self.noise_shape,
                            device=device)

    def _get_generator_condition_inputs(self, batch_size, device=None):
        return ()

    def on_batch_end(self, state: State):
        pass  # do nothing

    def on_loader_end(self, state: State):
        with torch.no_grad():
            metric = self._compute_metric(state) * self.multiplier
        state.loader_metrics[self.prefix] = metric


class CGanPerceptualPathLengthCallback(PerceptualPathLengthCallback):

    def __init__(self, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def _get_generator_condition_inputs(self, batch_size, device=None):
        c_flat = torch.randint(0, self.num_classes, size=(batch_size,))
        c_one_hot = torch.zeros((batch_size, self.num_classes))
        c_one_hot[torch.arange(batch_size), c_flat] = 1
        c_one_hot = c_one_hot.to(device)
        return (c_one_hot, )
