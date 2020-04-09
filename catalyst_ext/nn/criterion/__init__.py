from .bce_gan import BCELossDiscriminator, BCELossGenerator
from .hinge import (
    HingeLossDiscriminator, HingeLossGenerator,
    HingeLossDiscriminatorFake, HingeLossDiscriminatorReal
)
from .wasserstein import WassersteinLossDiscriminator, WassersteinLossGenerator