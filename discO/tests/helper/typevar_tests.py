# -*- coding: utf-8 -*-

"""`~typing.TypeVar` Tests."""

__all__ = [
    "TypeVarTests",
]


##############################################################################
# IMPORTS

# STDLIB
import abc
import typing as T

# LOCAL
from .objecttest import ObjectTest

##############################################################################
# CODE
##############################################################################


class TypeVarTests(ObjectTest, obj=T.TypeVar):
    """Type Testing Framework."""

    @classmethod
    @abc.abstractmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        pass

    # /def

    # -------------------------------

    def test_isTypeVar(self):
        """Test that this is a TypeVar."""
        assert isinstance(self.obj, T.TypeVar)

    # /def

    def test_bound(self):
        """Test TypeVar is correctly bound."""
        assert self.obj.__bound__ is self.bound

    # /def

    def test_name(self):
        """Test that name is [bound]."""
        name: str = self.obj.__name__
        if name.startswith("~"):
            name = name[1:]

        boundname: str = self.bound.__name__

        assert name == f"{boundname}", f"{name} != {boundname}"

    # /def


# /class


# -------------------------------------------------------------------

##############################################################################
# END
