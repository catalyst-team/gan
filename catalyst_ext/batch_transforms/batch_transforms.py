import torch


def distance(X, Y, kind='l2'):
    if kind != 'l2':
        raise NotImplementedError()
    nx = X.size(0)
    ny = Y.size(0)
    X = X.view(nx, -1)
    X2 = (X * X).sum(1)
    Y = Y.view(ny, -1)
    Y2 = (Y * Y).sum(1)

    dist = X2[:, None] + Y2[None] - 2 * torch.mm(X, Y.T)

    # if sqrt:
    #     M = ((M + M.abs()) / 2).sqrt()
    dist = dist.sqrt()
    return dist


def real_fake_distances(X_real, X_fake, kind='l2'):
    D_rr = distance(X=X_real, Y=X_real, kind=kind)
    D_rf = distance(X=X_real, Y=X_fake, kind=kind)
    D_ff = distance(X=X_fake, Y=X_fake, kind=kind)

    return D_rr, D_rf, D_ff


class BaseBatchTransform:
    def __init__(self, transform_fn, **transform_kwargs):
        self.transform_fn = transform_fn
        self.transform_kwargs = transform_kwargs

    def __call__(self, *args, **kwargs):
        return self.transform_fn(*args, **kwargs, **self.transform_kwargs)


class DistanceBatchTransform(BaseBatchTransform):
    def __init__(self, **transform_kwargs):
        super().__init__(transform_fn=distance, **transform_kwargs)


class RealFakeDistanceBatchTransform(BaseBatchTransform):
    def __init__(self, **transform_kwargs):
        super().__init__(transform_fn=real_fake_distances, **transform_kwargs)
