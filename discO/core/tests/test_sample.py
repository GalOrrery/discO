# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.core.sample`."""

__all__ = [
    "Test_PotentialSampler",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import pytest

# PROJECT-SPECIFIC
from discO.core import sample
from discO.core.tests.test_core import Test_PotentialBase

##############################################################################
# TESTS
##############################################################################


class Test_PotentialSampler(Test_PotentialBase, obj=sample.PotentialSampler):
    """Docstring for ClassName."""

    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.potential = object()

    # /def

    # -------------------------------

    def test___init_subclass__(self):
        """Test subclassing."""
        # When package is None
        class SubClasss1(self.obj):
            pass

        assert None not in sample.SAMPLER_REGISTRY

        try:

            class SubClasss2(self.obj, package="pytest"):
                pass

        except Exception:
            pass
        finally:
            sample.SAMPLER_REGISTRY.pop(pytest, None)

    # /def

    # -------------------------------

    def test___init__(self):

        # for object not in registry
        with pytest.raises(ValueError) as e:
            self.obj(self.potential)

        assert (
            "PotentialSampler has no registered sampler for package: "
            "<module 'builtins' (built-in)>"
        ) in str(e.value)

    # /def


# /class


# -------------------------------------------------------------------

##############################################################################
# END
