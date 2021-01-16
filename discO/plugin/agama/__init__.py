# -*- coding: utf-8 -*-
# see LICENSE.rst

"""AGAMA interface."""

__all__ = []


##############################################################################
# IMPORTS

# PROJECT-SPECIFIC
from . import sample
from .sample import *  # noqa: F401, F403

__all__ += sample.__all__


##############################################################################
# END
