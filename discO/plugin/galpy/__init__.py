# -*- coding: utf-8 -*-
# see LICENSE.rst

""":mod:`~galpy` interface."""

__all__ = [
    "GalpyPotentialWrapper",
]


##############################################################################
# IMPORTS

# PROJECT-SPECIFIC
from . import sample
from .sample import *  # noqa: F401, F403
from .wrapper import GalpyPotentialWrapper

# __all__
__all__ += sample.__all__  # flatten

##############################################################################
# END
