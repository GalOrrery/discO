# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.type_hints`."""

__all__ = [
    # Astropy types
    # coordinates
    "Test_RepresentationOrDifferentialType",
    "Test_RepresentationType",
    "Test_DifferentialType",
    "Test_FrameType",
    "Test_SkyCoordType",
    "Test_CoordinateType",
    "Test_PositionType",
    "Test_GenericPositionType",
    "Test_FrameLikeType",
    # tables
    "Test_TableType",
    "Test_QTableType",
    # units
    "Test_UnitType",
    "Test_UnitLkeType",
    "Test_QuantityType",
    "Test_QuantityLkeType",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import pytest
from astropy import table

# PROJECT-SPECIFIC
from discO import type_hints
from discO.tests.helper import TypeVarTests

##############################################################################
# TESTS
##############################################################################


class Test_RepresentationOrDifferentialType(
    TypeVarTests, obj=type_hints.RepresentationOrDifferentialType,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = coord.BaseRepresentationOrDifferential

    # /def


# /class

# -------------------------------------------------------------------


class Test_RepresentationType(TypeVarTests, obj=type_hints.RepresentationType):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = coord.BaseRepresentation

    # /def


# /class


# -------------------------------------------------------------------


@pytest.mark.skip("TODO")
class Test_RepresentationLikeType:
    """Test RepresentationLikeType."""


# /class

# -------------------------------------------------------------------


class Test_DifferentialType(TypeVarTests, obj=type_hints.DifferentialType):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = coord.BaseDifferential

    # /def


# /class

# -------------------------------------------------------------------


class Test_FrameType(TypeVarTests, obj=type_hints.FrameType):
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


class Test_SkyCoordType(TypeVarTests, obj=type_hints.SkyCoordType):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = coord.SkyCoord

    # /def


# /class

# -------------------------------------------------------------------


@pytest.mark.skip("TODO")
class Test_CoordinateType:
    """Test CoordinateType."""


# /class

# -------------------------------------------------------------------


@pytest.mark.skip("TODO")
class Test_PositionType:
    """Test PositionType."""


# /class

# -------------------------------------------------------------------


@pytest.mark.skip("TODO")
class Test_GenericPositionType:
    """Test GenericPositionType."""


# /class

# -------------------------------------------------------------------


@pytest.mark.skip("TODO")
class Test_FrameLikeType:
    """Test FrameLikeType."""


# /class

# -------------------------------------------------------------------


class Test_TableType(TypeVarTests, obj=type_hints.TableType):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = table.Table

    # /def


# /class

# -------------------------------------------------------------------


class Test_QTableType(TypeVarTests, obj=type_hints.QTableType):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = table.QTable

    # /def


# /class

# -------------------------------------------------------------------


@pytest.mark.skip("TODO")
class Test_UnitType:
    """Test UnitType."""


# /class


# -------------------------------------------------------------------


@pytest.mark.skip("TODO")
class Test_UnitLkeType:
    """Test UnitLkeType."""


# /class


# -------------------------------------------------------------------


class Test_QuantityType(TypeVarTests, obj=type_hints.QuantityType):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.bound = u.Quantity

    # /def


# /class

# -------------------------------------------------------------------


@pytest.mark.skip("TODO")
class Test_QuantityLkeType:
    """Test QuantityLkeType."""


# /class

# -------------------------------------------------------------------


##############################################################################
# END
