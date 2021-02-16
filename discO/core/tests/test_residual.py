# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.core.residual`."""

__all__ = ["Test_ResidualMethod", "Test_GridResidual"]


##############################################################################
# IMPORTS

# THIRD PARTY
import pytest

# PROJECT-SPECIFIC
from discO.core import residual
from discO.core.tests.test_common import Test_CommonBase as CommonBase_Test

##############################################################################
# TESTS
##############################################################################


class Test_ResidualMethod(CommonBase_Test, obj=residual.ResidualMethod):
    """Docstring for ClassName."""

    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        pass

    # /def

    @classmethod
    def teardown_class(cls):
        """Tear-down fixtures for testing."""
        pass

    # /def

    # -------------------------------

    @pytest.mark.skip("TODO")
    def test_method(self):
        """Test :class:`PACKAGE.CLASS.METHOD`."""
        assert False

    # /def


# /class


#####################################################################


class Test_GridResidual(Test_ResidualMethod, obj=residual.GridResidual):
    """Docstring for ClassName."""

    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        pass

    # /def

    @classmethod
    def teardown_class(cls):
        """Tear-down fixtures for testing."""
        pass

    # /def

    # -------------------------------

    @pytest.mark.skip("TODO")
    def test_method(self):
        """Test :class:`PACKAGE.CLASS.METHOD`."""
        assert False

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
