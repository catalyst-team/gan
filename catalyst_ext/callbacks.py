from copy import deepcopy
from collections import defaultdict, OrderedDict

import torch

from catalyst.dl import registry
from catalyst.dl import State
from catalyst.dl.core import Callback, CallbackOrder

from metrics import METRICS


class ArgsTransformer:
    IN_KEY = "in_key"
    IN_STATE_KEY = "in_state_key"
    OUT_KEY = "out_key"

    KEYS = (IN_KEY, IN_STATE_KEY, OUT_KEY)

    def __init__(self, args_params: list, kwargs_params: dict):
        assert isinstance(args_params, (list, tuple))
        assert isinstance(kwargs_params, dict)
        assert all(self._is_valid_arg(p) for p in args_params)
        assert all(self._is_valid_arg(p) for p in kwargs_params.values())

        self.args_params = args_params
        self.kwargs_params = kwargs_params

    def transform(self, state_input, state_output):
        args = []
        kwargs = {}
        for arg_params in self.args_params:
            args.append(self._get_key(
                params=arg_params,
                state_input=state_input,
                state_output=state_output
            ))
        for out_key, kwarg_params in self.kwargs_params.items():
            if ArgsTransformer.OUT_KEY in kwarg_params:
                out_key = kwarg_params[ArgsTransformer.OUT_KEY]

            kwargs[out_key] = self._get_key(
                params=kwarg_params,
                state_input=state_input,
                state_output=state_output
            )
        return args, kwargs

    @staticmethod
    def _is_valid_arg(arg_params):
        if not isinstance(arg_params, dict):
            return False
        return (
            ArgsTransformer.IN_KEY in arg_params
            and ArgsTransformer.IN_STATE_KEY in arg_params
            and (
                len(arg_params) == 2
                or (
                    len(arg_params) == 3
                    and ArgsTransformer.OUT_KEY in arg_params
                )
            )
        )

    @staticmethod
    def _get_key(params, state_input, state_output):
        in_state_key = params[ArgsTransformer.IN_STATE_KEY]
        if in_state_key == "input":
            dict_ = state_input
        elif in_state_key == "output":
            dict_ = state_output
        else:
            raise ValueError("Unsupported in_state_key")
        in_key = params["in_key"]
        if in_key not in dict_:
            raise ValueError(f"Key '{in_key}' is missing "
                             f"in state.{in_state_key}")

        return dict_[in_key]


@registry.Callback
class SampleBasedMetricCallback(Callback):
    def __init__(self, metrics,
                 n_samples: int = 2000,
                 feature_extractor_model_key=None,
                 real_data_input_key: str = "image",
                 fake_data_output_key: str = "fake_image"):
        super().__init__(order=CallbackOrder.Metric)
        self.metric_fns = self._get_metric_functions(deepcopy(metrics))
        self.n_samples = n_samples

        self.real_data_input_key = real_data_input_key
        self.fake_data_output_key = fake_data_output_key

        self.feature_extractor_model_key = feature_extractor_model_key

        self._real_data_images = [None] * self.n_samples
        self._fake_data_images = [None] * self.n_samples
        self._real_data_index = 0
        self._fake_data_index = 0

    @staticmethod
    def _get_metric_functions(metrics_params):
        metric_functions = OrderedDict()

        for prefix, metric_params in metrics_params.items():
            # change default metric key (prefix) if specified
            prefix = metric_params.pop("prefix", prefix)
            # todo: use
            feature_extractor = \
                metric_params.pop("feature_extractor_model_key", None)

            metric_args = metric_params.pop("metric_args", [])
            metric_kwargs = metric_params.pop("metric_kwargs", {})
            args_transformer = ArgsTransformer(
                args_params=metric_args,
                kwargs_params=metric_kwargs
            )
            metric_fn_ = METRICS.get(**metric_params)()  # todo: why not ititialized?

            def metric_fn(outputs, inputs,
                          args_transformer=args_transformer,
                          metric_fn_=metric_fn_):
                args, kwargs = args_transformer.transform(
                    state_input=inputs,
                    state_output=outputs
                )
                return metric_fn_(*args, **kwargs)

            metric_functions[prefix] = metric_fn
        return metric_functions

    def on_loader_start(self, state: State):
        # TODO: remove duplicated code
        self._real_data_images = [None] * self.n_samples
        self._fake_data_images = [None] * self.n_samples
        self._real_data_index = 0
        self._fake_data_index = 0

    def on_batch_end(self, state: State):
        # remember last samples
        real_data_batch = state.input[self.real_data_input_key]
        fake_data_batch = state.output[self.fake_data_output_key]

        for image in real_data_batch:
            self._real_data_images[self._real_data_index] = image
            self._real_data_index = (self._real_data_index + 1) % self.n_samples

        for image in fake_data_batch:
            self._fake_data_images[self._fake_data_index] = image
            self._fake_data_index = (self._fake_data_index + 1) % self.n_samples

    def on_loader_end(self, state: State):
        assert self._real_data_images[self._real_data_index] is not None, \
            f"Not enough images collected, ({self._real_data_index} " \
            f"images per epoch < {self.n_samples} images to evaluate)"

        assert self._fake_data_images[self._fake_data_index] is not None, \
            f"Not enough images collected, ({self._fake_data_index} " \
            f"images per epoch < {self.n_samples} images to evaluate)"

        self._real_data_images = torch.stack(self._real_data_images, dim=0)
        self._fake_data_images = torch.stack(self._fake_data_images, dim=0)

        # compute metrics
        for metric_name, metric_fn in self.metric_fns.items():
            # todo: feature extractor
            metric_value = metric_fn(state.output, state.input)
            # TODO: add method `add_epoch_value` to MetricManager
            state.metric_manager.epoch_values[state.loader_name][metric_name] = metric_value
