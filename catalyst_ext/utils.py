from typing import Dict, Union

import torch

from batch_transforms import BATCH_TRANSFORMS
from metrics import METRICS


# config parsing


def get_metric(params: Union[str, Dict[str, str]]):
    if isinstance(params, str):
        params = {"metric": params}
    if not isinstance(params, dict):
        raise NotImplementedError()
    metric_fn = METRICS.get_from_params(**params)
    return metric_fn


def get_batch_transform(params: Union[str, Dict[str, str]]):
    if isinstance(params, str):
        params = {"batch_transform": params}
    if not isinstance(params, dict):
        raise NotImplementedError()
    transform_fn = BATCH_TRANSFORMS.get_from_params(**params)
    return transform_fn


# torch models


def get_module(model, path):
    if path == '':
        return model

    curr = model
    for attrib_name in path.split('.'):
        # prev = curr
        curr = getattr(curr, attrib_name)
    return curr


# preprocessing


def as_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        return torch.stack(x, dim=0)
    raise NotImplementedError()


def change_num_image_channels(x: torch.Tensor,
                              channels_num_out: int,
                              channels_dim: int = 1):
    assert x.ndim > channels_dim
    channels_num_in = x.size(channels_dim)
    if channels_num_out != channels_num_in:
        if channels_num_in == 1:
            x = torch.cat([x] * channels_num_out, dim=channels_dim)
        elif channels_num_out == 1:
            x = torch.mean(x, dim=channels_dim, keepdim=True)
        else:
            raise NotImplementedError()
    return x


# interpolation


def slerp(x1, x2, t):
    """spherical interpolation"""
    x1_norm = x1 / torch.norm(x1, dim=1, keepdim=True)
    x2_norm = x2 / torch.norm(x2, dim=1, keepdim=True)
    omega = torch.acos((x1_norm * x2_norm).sum(1))
    sin_omega = torch.sin(omega)
    return (
            (torch.sin((1 - t) * omega) / sin_omega).unsqueeze(1) * x1
            + (torch.sin(t * omega) / sin_omega).unsqueeze(1) * x2
    )
