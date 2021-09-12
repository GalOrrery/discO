# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Set up module."""

##############################################################################
# IMPORTS

from __future__ import absolute_import

# STDLIB
import importlib

__all__ = ["HAS_AGAMA", "HAS_GALA", "HAS_GALPY", "HAS_TQDM"]


##############################################################################
# PARAMETERS

try:
    # THIRD PARTY
    import agama  # noqa: F401
except ImportError:
    HAS_AGAMA = False
else:
    HAS_AGAMA = True

    agama.setUnits(mass=1, length=1, velocity=1)  # FIXME! bad

# /try

# -------------------------------------

try:
    # THIRD PARTY
    import gala  # noqa: F401
except ImportError:
    HAS_GALA = False
else:
    HAS_GALA = True

# /try

# -------------------------------------

try:
    # THIRD PARTY
    import galpy  # noqa: F401
except ImportError:
    HAS_GALPY = False
else:
    HAS_GALPY = True

    importlib.reload(galpy)

    # THIRD PARTY
    from galpy.util import config as galpy_config

    # force configuration
    galpy_config._APY_LOADED = True
    galpy_config.__config__.set("astropy", "astropy-units", "True")
    galpy_config.__config__.set("astropy", "astropy-coords", "True")
    galpy_config.default_configuration["astropy"]["astropy-units"] = True

# /try

# -------------------------------------

try:
    # THIRD PARTY
    from tqdm import tqdm  # noqa: F401
except ImportError:
    HAS_TQDM = False
else:
    HAS_TQDM = True


##############################################################################
# END
