# -*- coding: utf-8 -*-
# see LICENSE.rst

"""AGAMA interface."""

__all__ = []


##############################################################################
# IMPORTS

# PROJECT-SPECIFIC
from . import fitter
from .fitter import *  # noqa: F401, F403

__all__ += fitter.__all__


##############################################################################
# END
