# -*- coding: utf-8 -*-

"""Testing :mod:`~PACKAGE`."""

__all__ = [
    "Test_ClassName",
    "test_function",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import pytest

##############################################################################
# PARAMETERS


##############################################################################
# PYTEST


def setup_module(module):
    """Setup module for testing."""
    pass


# /def


def teardown_module(module):
    """Tear-down module for testing."""
    pass


# /def


##############################################################################
# TESTS
##############################################################################


class Test_ClassName(object):
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


def test_function():
    """Test :func:`PACKAGE.METHOD`."""
    pass


# /def


##############################################################################
# END
