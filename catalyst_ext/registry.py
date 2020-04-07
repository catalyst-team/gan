# TODO: registry everything here in a similar way as catalyst.contrib
from catalyst.dl import registry

import criterions as gan_criterions

registry.MODULES.add_from_module(gan_criterions)
registry.CRITERIONS.add_from_module(gan_criterions)
