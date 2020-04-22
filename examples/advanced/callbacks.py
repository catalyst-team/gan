# flake8: noqa
# isort: skip_file
import copy
import torch
import torchvision.utils

from catalyst.dl import Callback, CallbackOrder, State
from catalyst.dl.callbacks import MetricManagerCallback
from catalyst.contrib.utils.tools.tensorboard import SummaryWriter

from catalyst_gan.callbacks import PhaseManagerCallback


class VisualizationCallback(Callback):
    TENSORBOARD_LOGGER_KEY = "_tensorboard"

    def __init__(
        self,
        input_keys=None,
        output_keys=None,
        batch_frequency=25,
        concat_images=True,
        max_images=20,
        num_rows=1,
        denorm="default"
    ):
        super().__init__(CallbackOrder.External)
        if input_keys is None:
            self.input_keys = []
        elif isinstance(input_keys, str):
            self.input_keys = [input_keys]
        elif isinstance(input_keys, (tuple, list)):
            assert all(isinstance(k, str) for k in input_keys)
            self.input_keys = list(input_keys)
        else:
            raise ValueError(
                f"Unexpected format of 'input_keys' "
                f"argument: must be string or list/tuple"
            )

        if output_keys is None:
            self.output_keys = []
        elif isinstance(output_keys, str):
            self.output_keys = [output_keys]
        elif isinstance(output_keys, (tuple, list)):
            assert all(isinstance(k, str) for k in output_keys)
            self.output_keys = list(output_keys)
        else:
            raise ValueError(
                f"Unexpected format of 'output_keys' "
                f"argument: must be string or list/tuple"
            )

        if len(self.input_keys) + len(self.output_keys) == 0:
            raise ValueError("Useless visualizer: pass at least one image key")

        self.batch_frequency = int(batch_frequency)
        assert self.batch_frequency > 0

        self.concat_images = concat_images
        self.max_images = max_images
        if denorm.lower() == "default":
            # normalization from [-1, 1] to [0, 1] (the latter is valid for tb)
            self.denorm = lambda x: x / 2 + .5
        elif denorm is None or denorm.lower() == "none":
            self.denorm = lambda x: x
        else:
            raise ValueError("unknown denorm fn")
        self._num_rows = num_rows
        self._reset()

    def _reset(self):
        self._loader_batch_count = 0
        self._loader_visualized_in_current_epoch = False

    @staticmethod
    def _get_tensorboard_logger(state: State) -> SummaryWriter:
        tb_key = VisualizationCallback.TENSORBOARD_LOGGER_KEY
        if (
            tb_key in state.callbacks
            and state.loader_name in state.callbacks[tb_key].loggers
        ):
            return state.callbacks[tb_key].loggers[state.loader_name]
        raise RuntimeError(
            f"Cannot find Tensorboard logger for loader {state.loader_name}"
        )

    def compute_visualizations(self, state):
        input_tensors = [
            state.input[input_key] for input_key in self.input_keys
        ]
        output_tensors = [
            state.batch_out[output_key] for output_key in self.output_keys
        ]
        visualizations = dict()
        if self.concat_images:
            viz_name = "|".join(self.input_keys + self.output_keys)
            viz_tensor = self.denorm(
                # concat by width
                torch.cat(input_tensors + output_tensors, dim=3)
            ).detach().cpu()
            visualizations[viz_name] = viz_tensor
        else:
            visualizations = dict(
                (k, self.denorm(v)) for k, v in zip(
                    self.input_keys + self.output_keys, input_tensors +
                    output_tensors
                )
            )
        return visualizations

    def save_visualizations(self, state: State, visualizations):
        tb_logger = self._get_tensorboard_logger(state)
        for key, batch_images in visualizations.items():
            batch_images = batch_images[:self.max_images]
            image = torchvision.utils.make_grid(
                batch_images, nrow=self._num_rows
            )
            tb_logger.add_image(key, image, global_step=state.global_step)

    def visualize(self, state):
        visualizations = self.compute_visualizations(state)
        self.save_visualizations(state, visualizations)
        self._loader_visualized_in_current_epoch = True

    def on_loader_start(self, state: State):
        self._reset()

    def on_loader_end(self, state: State):
        if not self._loader_visualized_in_current_epoch:
            self.visualize(state)

    def on_batch_end(self, state: State):
        self._loader_batch_count += 1
        if self._loader_batch_count % self.batch_frequency:
            self.visualize(state)


class ConstNoiseVisualizerCalback(Callback):
    TENSORBOARD_LOGGER_KEY = "_tensorboard"

    def __init__(self, noise_dim, visualization_key="fixed_noise",
                 rows=5, cols=5, n_classes=None, only_valid=True):
        super().__init__(order=CallbackOrder.Metric)
        self.visualization_key = visualization_key
        self.rows = rows
        # normalization from [-1, 1] to [0, 1] (the latter is valid for tb)
        self.denorm = lambda x: x / 2 + .5

        self.noise = torch.normal(0, 1, size=(rows * cols, noise_dim))
        self.n_classes = n_classes
        self.only_valid = only_valid
        if n_classes is None:
            self.generator_inputs = (self.noise, )
        else:
            classes = torch.arange(rows * cols) % n_classes
            one_hot_classes = torch.zeros((rows * cols, n_classes))
            one_hot_classes[torch.arange(rows*cols), classes] = 1
            self.generator_inputs = (self.noise, one_hot_classes)

    def on_epoch_end(self, state: "State"):
        if not state.is_train_loader or not self.only_valid:
            inputs = (tensor.to(state.device) for tensor in self.generator_inputs)
            model = state.model["generator"]
            outputs = model(*inputs)
            image = torchvision.utils.make_grid(
                self.denorm(outputs), nrow=self.rows
            )
            tb_logger = self._get_tensorboard_logger(state)
            tb_logger.add_image(self.visualization_key, image,
                                global_step=state.global_step)

    @staticmethod
    def _get_tensorboard_logger(state: State) -> SummaryWriter:
        tb_key = ConstNoiseVisualizerCalback.TENSORBOARD_LOGGER_KEY
        if (
                tb_key in state.callbacks
                and state.loader_name in state.callbacks[tb_key].loggers
        ):
            return state.callbacks[tb_key].loggers[state.loader_name]
        raise RuntimeError(
            f"Cannot find Tensorboard logger for loader {state.loader_name}"
        )


class TrickyMetricManagerCallback(MetricManagerCallback):

    def on_batch_start(self, state: State):
        state.prev_batch_metrics = state.batch_metrics
        super().on_batch_start(state)


class HParamsTbSaverCallback(Callback):

    _metrics_dict = {
        "metrics/FID": "FID"
    }

    def __init__(self, hparams_dict, metrics_dict=None,
                 last=True, best=True):
        super().__init__(order=CallbackOrder.Metric)
        self.hparams_dict = dict(hparams_dict)
        # name_in -> name_out
        self.metrics_dict = metrics_dict or self._metrics_dict
        self.last = last
        self.best = best

    def on_stage_end(self, state: State):
        if state.is_infer_stage:
            return
        tb_logger = self._get_tensorboard_logger(state)
        hparams_dict = dict(self.hparams_dict)
        if self.last:
            hparams_dict["record_kind"] = "last"
            tb_logger.add_hparams(
                hparam_dict=hparams_dict,
                metric_dict={
                    v: state.valid_metrics[k]
                    for k, v in self.metrics_dict.items()
                }
            )
        if self.best:
            hparams_dict["record_kind"] = "best"
            tb_logger.add_hparams(
                hparam_dict=hparams_dict,
                metric_dict={
                    v: state.best_valid_metrics[k]
                    for k, v in self.metrics_dict.items()
                }
            )

    @staticmethod
    def _get_tensorboard_logger(state: State) -> SummaryWriter:
        tb_key = ConstNoiseVisualizerCalback.TENSORBOARD_LOGGER_KEY
        if (
                tb_key in state.callbacks
                and state.loader_name in state.callbacks[tb_key].loggers
        ):  # Note: gets valid_loader(!)
            return state.callbacks[tb_key].loggers[state.valid_loader]
        raise RuntimeError(
            f"Cannot find Tensorboard logger for loader {state.loader_name}"
        )


__all__ = ["VisualizationCallback", "TrickyMetricManagerCallback",
           "ConstNoiseVisualizerCalback", "HParamsTbSaverCallback"]
