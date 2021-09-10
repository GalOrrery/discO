# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.plugin.gala.type_hints`."""

__all__ = [
    "Test_PotentialType",
]


##############################################################################
# IMPORTS

# THIRD PARTY
from gala.potential import PotentialBase

# LOCAL
from discO.plugin.gala import type_hints
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
        cls.bound = PotentialBase

    # /def


# /class


##############################################################################
# END
