# -*- coding: utf-8 -*-

""":mod:`~gala` type hints.

This project extensively uses :mod:`~typing` hints.
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
from gala.potential import PotentialBase

##############################################################################
# TYPES
##############################################################################

PotentialType = T.TypeVar("PotentialBase", bound=PotentialBase)
""":class:`~gala.potential.PotentialBase`"""

##############################################################################
# END
