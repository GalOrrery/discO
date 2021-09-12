# -*- coding: utf-8 -*-

""":mod:`~agama` type hints.

This project extensively uses :mod:`~typing` hints.
Note that this is not (necessarily) static typing.

"""

__all__ = [
    "PotentialType",
]

##############################################################################
# IMPORTS

# STDLIB
import typing as T

# THIRD PARTY
from agama import Potential

##############################################################################
# TYPES
##############################################################################

PotentialType = T.TypeVar("Potential", bound=Potential)
""":class:`~agama.Potential`"""

##############################################################################
# END
