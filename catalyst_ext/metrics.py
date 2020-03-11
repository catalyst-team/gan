import torch
import numpy as np
import scipy.linalg

from catalyst.dl import registry

METRICS = registry.Registry("metric")
Metric = METRICS.add


def inception_score(samples, eps=1e-20):
    X = samples
    kl = X * ((X + eps).log() - (X.mean(0)+eps).log().expand_as(X))
    score = kl.sum(1).mean().exp()

    return score


def mode_score(samples, samples_real, eps=1e-20):
    X = samples
    Y = samples_real
    kl1 = X * ((X + eps).log() - (X.mean(0) + eps).log().expand_as(X))
    kl2 = X.mean(0) * ((X.mean(0) + eps).log() - (Y.mean(0) + eps).log())
    score = (kl1.sum(1).mean() - kl2.sum()).exp()

    return score


# def torch_covariance(x):
#     # x.shape == (n_samples, n_features);
#     # returns covariance matrix of shape (n_features, n_features)
#     # along axis 1 (!)
#     assert x.ndim == 2
#     # assert x.ndim > 1
#     # if x.ndim != 2:
#     #     x = x.reshape(x.size(0), -1)
#
#     exp_xy = x[:, None] * x[:, :, None]
#     exp_x = x.mean(0)
#     exp_x_exp_y = exp_x[None] * exp_x[:, None]
#     cov = exp_xy.mean(0) - exp_x_exp_y
#     return cov


def frechet_inception_distance(samples_a, samples_b):
    if isinstance(samples_a, torch.Tensor):
        samples_a = samples_a.detach().cpu().numpy()
    if isinstance(samples_b, torch.Tensor):
        samples_b = samples_b.detach().cpu().numpy()
    assert isinstance(samples_a, np.ndarray)
    assert isinstance(samples_b, np.ndarray)

    if samples_a.ndim > 2:
        samples_a = samples_a.reshape((samples_a.shape[0], -1))
    if samples_b.ndim > 2:
        samples_b = samples_b.reshape((samples_b.shape[0], -1))
    assert samples_a.ndim == 2
    assert samples_b.ndim == 2

    mu_a = samples_a.mean(0)
    mu_b = samples_b.mean(0)

    cov_a = np.cov(samples_a.T)
    cov_b = np.cov(samples_b.T)

    w1_dist = mu_a @ mu_a + mu_b @ mu_b - 2 * mu_a @ mu_b
    w2_dist = w1_dist + np.trace(
        cov_a + cov_b - 2 * scipy.linalg.sqrtm(cov_a @ cov_b).real
    )

    return w2_dist


@Metric
class MetricModule:
    def __init__(self, metric_fn):
        self.metric_fn = metric_fn

    def __call__(self, *args, **kwargs):
        return self.metric_fn(*args, **kwargs)


@Metric
class FrechetInceptionDistance(MetricModule):
    def __init__(self):
        super().__init__(metric_fn=frechet_inception_distance)


@Metric
class InceptionScore(MetricModule):
    def __init__(self):
        super().__init__(metric_fn=inception_score)


@Metric
class ModeScore(MetricModule):
    def __init__(self):
        super().__init__(metric_fn=mode_score)
