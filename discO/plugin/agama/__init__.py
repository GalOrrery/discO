# -*- coding: utf-8 -*-
# see LICENSE.rst

"""AGAMA interface.

If using :mod:`~agama`, the units must be set to

.. code-block:: python

    agama.setUnits(mass=1, length=1, velocity=1)


"""

__all__ = [
    "AGAMAPotentialWrapper",
]


##############################################################################
# IMPORTS

# LOCAL
from . import fitter, sample
from .fitter import *  # noqa: F401, F403
from .sample import *  # noqa: F401, F403
from .wrapper import AGAMAPotentialWrapper

# __all__
__all__ += sample.__all__
__all__ += fitter.__all__

##############################################################################
# END
