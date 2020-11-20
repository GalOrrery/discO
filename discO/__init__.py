# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Disk Expansion Project."""

__author__ = ("Nathaniel Starkman", "Christopher Carr")

__credits__ = ["Jo Bovy"]
__maintainer__ = "Nathaniel Starkman"
__status__ = "In Progress"


__all__ = [
    # data
    "data",
    # configuration
    "conf",
]


##############################################################################
# IMPORTS

# keep this content at the top. (sets the __version__)
from ._astropy_init import *  # noqa  # isort:skip
from ._astropy_init import __version__  # noqa  # isort:skip

# PROJECT-SPECIFIC
from . import extern  # noqa: F401  # just need this to register in classes
from . import core, data
from .config import conf
from .core import *  # noqa: F401, F403

# All
__all__ += core.__all__

##############################################################################
# END
