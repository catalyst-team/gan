from collections import OrderedDict

from torch import nn

from catalyst.dl import ConfigExperiment
from catalyst.dl.registry import MODELS
from catalyst_gan.utils import get_dataset
from .callbacks import HParamsTbSaverCallback


# custom weights initialization called on netG and netD
def dcgan_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# data loaders & transforms
class Experiment(ConfigExperiment):
    """
    Simple experiment
    """
    def get_datasets(
        self, stage: str,
        datasets: dict = None,
        **dataset_params_shared
    ):
        """

        :param stage:
        :param datasets:
        :param dataset_params_shared:
        :return:
        """
        datasets_ = OrderedDict()

        for dataset_name, dataset_params in datasets.items():
            transform = self.get_transforms(stage=stage, dataset=dataset_name)
            datasets_[dataset_name] = get_dataset(
                transform=transform,
                **dataset_params_shared, **dataset_params
            )

        return datasets_

    @staticmethod
    def _get_model(**params):
        key_value_flag = params.pop("_key_value", False)
        # todo registry init
        dc_init_flag = params.pop("_dcgan_initialize", False)

        if key_value_flag:
            model = {}
            for key, params_ in params.items():
                model[key] = Experiment._get_model(**params_)
            model = nn.ModuleDict(model)
        else:
            model = MODELS.get_from_params(**params)
        if dc_init_flag:
            model.apply(dcgan_weights_init)
        return model

    def get_callbacks(self, stage: str):
        # more convenient to add this callback from here
        # as otherwise it would require to add extra lines to all configs
        callbacks = super().get_callbacks(stage)
        callbacks["hparams_saver"] = HParamsTbSaverCallback(
            hparams_dict=dict(self._config.get("hparams", {})),
            metrics_dict=None
        )
        return callbacks
