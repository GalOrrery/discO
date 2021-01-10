# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.plugin.galpy.sample`."""

__all__ = [
    "Test_GalpyPotentialSampler",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import pytest

# PROJECT-SPECIFIC
from discO.core.tests.test_sample import Test_PotentialSampler
from discO.plugin.galpy import sample

##############################################################################
# TESTS
##############################################################################


class Test_GalpyPotentialSampler(
    Test_PotentialSampler, obj=sample.GalpyPotentialSampler
):
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
