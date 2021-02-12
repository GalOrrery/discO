# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Utilities."""

__all__ = [
    "resolve_framelike",
    "resolve_representationlike",
]


##############################################################################
# IMPORTS

# PROJECT-SPECIFIC
from . import vectorfield
from ._framelike import resolve_framelike, resolve_representationlike
from .vectorfield import *  # noqa: F401, F403

__all__ += vectorfield.__all__

##############################################################################
# END
