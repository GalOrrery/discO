# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.core.core`."""

__all__ = [
    "Test_PotentialBase",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import pytest

# PROJECT-SPECIFIC
from discO.core import core
from discO.tests.helper import ObjectTest

##############################################################################
# TESTS
##############################################################################


class Test_PotentialBase(ObjectTest, obj=core.PotentialBase):
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
