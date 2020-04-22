from .bce_gan import (
    BCELossDiscriminator, BCELossGenerator,
    BCELossDiscriminatorFake, BCELossDiscriminatorReal
)
from .hinge import (
    HingeLossDiscriminator, HingeLossGenerator,
    HingeLossDiscriminatorFake, HingeLossDiscriminatorReal
)
from .wasserstein import (
    WassersteinLossDiscriminator, WassersteinLossGenerator,
    WassersteinLossDiscriminatorFake, WassersteinLossDiscriminatorReal,
    WassersteinDistance
)
from .metrics import AverageValue, AverageProbability
