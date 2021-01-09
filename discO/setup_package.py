# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Set up module."""

##############################################################################
# IMPORTS

from __future__ import absolute_import

__all__ = ["HAS_AGAMA", "HAS_GALPY"]


##############################################################################
# PARAMETERS

try:
    import agama  # noqa: F401
except ImportError:
    HAS_AGAMA = False
else:
    HAS_AGAMA = True

# -------------------------------------

try:
    import galpy  # noqa: F401
except ImportError:
    HAS_GALPY = False
else:
    HAS_GALPY = True


##############################################################################
# END
