# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.plugin.galpy.type_hints`."""

__all__ = [
    "Test_PotentialType",
]


##############################################################################
# IMPORTS

# THIRD PARTY
from galpy import potential

# PROJECT-SPECIFIC
from discO.plugin.galpy import type_hints
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
        cls.bound = potential.Potential

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
