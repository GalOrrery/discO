# -*- coding: utf-8 -*-

""":mod:`~galpy` type hints.

This project extensively uses `~typing` hints.
Note that this is not (necessarily) static typing.

"""

__all__ = [
    "PotentialType",
]

##############################################################################
# IMPORTS

# BUILT-IN
import typing as T

# THIRD PARTY
from galpy import potential

##############################################################################
# TYPES
##############################################################################

PotentialType = T.TypeVar("Potential", bound=potential.Potential)

##############################################################################
# END
