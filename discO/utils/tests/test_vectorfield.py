# -*- coding: utf-8 -*-

"""Testing :mod:`~PACKAGE`."""

__all__ = [
    "Test_BaseVectorField",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import pytest

# PROJECT-SPECIFIC
from discO.tests.helper import ObjectTest
from discO.utils import vectorfield

##############################################################################
# PARAMETERS


##############################################################################
# TESTS
##############################################################################


class Test_BaseVectorField(ObjectTest, obj=vectorfield.BaseVectorField):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        pass

    # /def

    #######################################################
    # Methods Tests

    @pytest.mark.skip("TODO")
    def test___init_subclass__(self):
        """Test method ``__init_subclass__``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test___init__(self):
        """Test method ``__init__``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test_frame(self):
        """Test method ``frame``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test_to_cartesian(self):
        """Test method ``to_cartesian``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test_from_cartesian(self):
        """Test method ``from_cartesian``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test_represent_as(self):
        """Test method ``represent_as``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test_from_field(self):
        """Test method ``from_field``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test__scale_operation(self):
        """Test method ``_scale_operation``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test__combine_operation(self):
        """Test method ``_combine_operation``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test_norm(self):
        """Test method ``norm``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test_unit_vectors(self):
        """Test method ``unit_vectors``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test_scale_factors(self):
        """Test method ``scale_factors``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test___repr__(self):
        """Test method ``__repr__``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test__apply(self):
        """Test method ``_apply``."""
        assert False

    # /def

    #######################################################
    # Usage Tests


# /class


# -------------------------------------------------------------------


class Test_CartesianVectorField(
    Test_BaseVectorField,
    obj=vectorfield.CartesianVectorField,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        pass

    # /def

    #######################################################
    # Methods Tests

    @pytest.mark.skip("TODO")
    def test_attributes(self):
        """Test class attributes."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test___init__(self):
        """Test method ``__init__``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test_get_xyz(self):
        """Test method ``get_xyz``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test_get_vf_xyz(self):
        """Test method ``get_vf_xyz``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test_dot(self):
        """Test method ``dot``."""
        assert False

    # /def

    #######################################################
    # Usage Tests


# /class

# -------------------------------------------------------------------


class Test_CylindricalVectorField(
    Test_BaseVectorField,
    obj=vectorfield.CylindricalVectorField,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        pass

    # /def

    #######################################################
    # Methods Tests

    @pytest.mark.skip("TODO")
    def test_attributes(self):
        """Test class attributes."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test___init__(self):
        """Test method ``__init__``."""
        assert False

    # /def

    #######################################################
    # Usage Tests


# /class

# -------------------------------------------------------------------


class Test_SphericalVectorField(
    Test_BaseVectorField,
    obj=vectorfield.SphericalVectorField,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        pass

    # /def

    #######################################################
    # Methods Tests

    @pytest.mark.skip("TODO")
    def test_attributes(self):
        """Test class attributes."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test___init__(self):
        """Test method ``__init__``."""
        assert False

    # /def

    #######################################################
    # Usage Tests


# /class

# -------------------------------------------------------------------


class Test_PhysicsSphericalVectorField(
    Test_BaseVectorField,
    obj=vectorfield.PhysicsSphericalVectorField,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        pass

    # /def

    #######################################################
    # Methods Tests

    @pytest.mark.skip("TODO")
    def test_attributes(self):
        """Test class attributes."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test___init__(self):
        """Test method ``__init__``."""
        assert False

    # /def

    #######################################################
    # Usage Tests


# /class

##############################################################################
# END
