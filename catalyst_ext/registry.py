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


def _datasets_loader(r: registry.Registry):
    # TODO add original torchvision and wrapped datasets (with kv dataset[i])
    from catalyst_ext import datasets
    r.add_from_module(datasets)


DATASETS = registry.Registry("dataset")
Dataset = DATASETS.add
DATASETS.late_add(_datasets_loader)
