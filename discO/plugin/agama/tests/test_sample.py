# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.plugin.agama.sample`."""

__all__ = [
    "Test_AGAMAPotentialSampler",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import agama
import pytest

# PROJECT-SPECIFIC
from discO.core.tests.test_sample import (
    Test_PotentialSampler as PotentialSamplerTester,
)
from discO.plugin.agama import sample

##############################################################################
# TESTS
##############################################################################


class Test_AGAMAPotentialSampler(
    PotentialSamplerTester,
    obj=sample.AGAMAPotentialSampler,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        super().setup_class()

        # make potential
        cls.potential = agama.Potential(
            type="Spheroid",
            mass=1e12,
            scaleRadius=10,
            gamma=1,
            alpha=1,
            beta=4,
            cutoffStrength=0,
        )

        cls.inst = cls.obj(cls.potential)

    # /def

    @pytest.mark.skip("TODO")
    def test___call__(self):
        """Test method ``__call__``.

        When Test_MeasurementErrorSampler this calls on the wrapped instance,
        which is GaussianMeasurementErrorSampler.

        """
        # run tests on super
        super().test___call__()

        # --------------------------
        # with c_err

        self.inst(self.c, self.c_err)

        # ---------------
        # without c_err, using from instantiation

        self.inst(self.c)

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
