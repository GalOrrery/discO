# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.plugin.agama.type_hints`."""

__all__ = [
    "Test_PotentialType",
]


##############################################################################
# IMPORTS

# THIRD PARTY
from agama import Potential

# PROJECT-SPECIFIC
from discO.plugin.agama import type_hints
from discO.tests.helper import TypeVarTests

##############################################################################
# TESTS
##############################################################################


class Test_PotentialType(
    TypeVarTests,
    obj=type_hints.PotentialType,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = Potential

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
