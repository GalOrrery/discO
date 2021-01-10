# -*- coding: utf-8 -*-
# see LICENSE.rst

"""**DOCSTRING**."""

__all__ = []


##############################################################################
# IMPORTS

# PROJECT-SPECIFIC
from discO.setup_package import HAS_GALPY

if HAS_GALPY:

    # PROJECT-SPECIFIC
    from . import sample
    from .sample import *  # noqa: F401, F403

    __all__ += sample.__all__


##############################################################################
# END
