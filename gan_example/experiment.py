from collections import OrderedDict

from catalyst.dl import ConfigExperiment
from catalyst_ext import DATASETS


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
            datasets_[dataset_name] = DATASETS.get_from_params(
                transform=transform,
                **dataset_params_shared, **dataset_params
            )

        return datasets_
