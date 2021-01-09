# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.common`."""

__all__ = [
    "Test_QuantityType",
    "Test_FrameType",
    "Test_SkyCoordType",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u

# PROJECT-SPECIFIC
from discO import common
from discO.tests.helper import TypeVarTests

##############################################################################
# TESTS
##############################################################################


class Test_QuantityType(TypeVarTests, obj=common.QuantityType):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = u.Quantity

    # /def


# /class

# -------------------------------------------------------------------


class Test_FrameType(TypeVarTests, obj=common.FrameType):

    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = coord.BaseCoordinateFrame

    # /def

    def test_name(self):
        """Test that name is {bound}."""
        name: str = self.obj.__name__
        if name.startswith("~"):
            name = name[1:]

        assert name == "CoordinateFrame"

    # /def


# /class

# -------------------------------------------------------------------


class Test_SkyCoordType(TypeVarTests, obj=common.SkyCoordType):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = coord.SkyCoord

    # /def


# /class


##############################################################################
# END
