import inspect
import functools
import torch.utils.data
import numpy as np
from catalyst.dl import registry


def _modules_loader(r: registry.Registry):
    from catalyst_ext import nn
    r.add_from_module(nn)


registry.MODULES.late_add(_modules_loader)


def _criterions_loader(r: registry.Registry):
    from catalyst_ext.nn import criterion
    r.add_from_module(criterion)


registry.CRITERIONS.late_add(_criterions_loader)


def _models_loader(r: registry.Registry):
    from catalyst_ext import models
    r.add_from_module(models)

    try:
        from torch_mimicry.nets import sngan, ssgan, infomax_gan
        r.add_from_module(sngan, prefix=['tm.', 'tm.sngan.'])
        r.add_from_module(ssgan, prefix=['tm.', 'tm.ssgan.'])
        r.add_from_module(infomax_gan, prefix=['tm.', 'tm.infomax_gan.'])
    except ImportError:
        pass  # TODO: warning?


registry.MODELS.late_add(_models_loader)


def _callbacks_loader(r: registry.Registry):
    from catalyst_ext import callbacks
    r.add_from_module(callbacks)


registry.CALLBACKS.late_add(_callbacks_loader)


def _transforms_loader(r: registry.Registry):
    from catalyst_ext import transforms
    r.add_from_module(transforms)


registry.TRANSFORMS.late_add(_transforms_loader)


def _batch_transforms_loader(r: registry.Registry):
    from catalyst_ext import batch_transforms
    r.add_from_module(batch_transforms)


BATCH_TRANSFORMS = registry.Registry("batch_transform")
BatchTransform = BATCH_TRANSFORMS.add
BATCH_TRANSFORMS.late_add(_batch_transforms_loader)


def _metrics_loader(r: registry.Registry):
    from catalyst_ext import metrics
    r.add_from_module(metrics)


METRICS = registry.Registry("metric")
Metric = METRICS.add
METRICS.late_add(_metrics_loader)


class _TorchvisionDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, base_dataset,
                 transform=None,
                 base_dataset_transform=None,
                 image_key="image",
                 target_key="target",
                 **kwargs):
        self.base_dataset = base_dataset(transform=base_dataset_transform,
                                         **kwargs)
        self.image_key = image_key
        self.target_key = target_key
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        out = self.base_dataset[idx]
        if self.target_key is not None:
            assert len(out) == 2
            dict_ = {
                self.image_key: out[0],
                self.target_key: out[1]
            }
        else:
            if isinstance(out, tuple):
                out = out[0]
            dict_ = {
                self.image_key: out
            }
        if self.transform is not None:
            dict_ = self.transform(dict_)
        return dict_


def _get_module_classes(m):
    factories = {
        k: v
        for k, v in m.__dict__.items()
        if inspect.isclass(v)
    }

    # Filter by __all__ if present
    names_to_add = getattr(m, "__all__", list(factories.keys()))
    return {k: factories[k] for k in names_to_add}


def _datasets_loader(r: registry.Registry):
    # TODO add original torchvision and wrapped datasets (with kv dataset[i])
    from catalyst_ext import datasets as ext_datasets
    r.add_from_module(ext_datasets)

    try:
        from torchvision import datasets
        r.add_from_module(datasets, prefix=["tv.", "torchvision."])
        # wrapped datasets
        datasets_to_add = _get_module_classes(datasets)
        prefixes = ["tv.kv.", "torchvision.keyvalue."]
        for name, cls in datasets_to_add.items():
            cls_w = functools.partial(_TorchvisionDatasetWrapper,
                                      base_dataset=cls)
            r.add(**{f"{p}{name}": cls_w for p in prefixes})
    except ImportError:
        pass


DATASETS = registry.Registry("dataset")
Dataset = DATASETS.add
DATASETS.late_add(_datasets_loader)
