# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.plugin.galpy.sample`."""

__all__ = [
    "Test_GalpyPotentialSampler",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest
from galpy.df import isotropicHernquistdf
from galpy.potential import HernquistPotential

# PROJECT-SPECIFIC
from discO.core.tests.test_sample import Test_PotentialSampler
from discO.plugin.galpy import sample

##############################################################################
# TESTS
##############################################################################


class Test_GalpyPotentialSampler(
    Test_PotentialSampler,
    obj=sample.GalpyPotentialSampler,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        super().setup_class()

        # make potential
        cls.mass = 1e12 * u.solMass
        hernquist_pot = HernquistPotential(amp=cls.mass)
        hernquist_pot.turn_physical_on()  # force units
        cls.potential = isotropicHernquistdf(hernquist_pot)
        cls.potential.turn_physical_on()

        cls.inst = cls.obj(cls.potential)

    # /def

    def test___call__(self):
        """Test method ``__call__``.

        When Test_MeasurementErrorSampler this calls on the wrapped instance,
        which is GaussianMeasurementErrorSampler.

        """
        # run tests on super
        super().test___call__()

    # /def

    @pytest.mark.parametrize(
        "n,frame,kwargs",
        [
            (10, None, {}),  # just "n"
            (10, "FK5", {}),  # specifying frame
            (10, "FK5", dict(a=1, b=2)),  # adding kwargs
        ],
    )
    def test_call_parametrize(self, n, frame, kwargs):
        """Parametrized call tests."""
        res = self.inst(n, frame=frame, **kwargs)
        assert res.__class__ == coord.SkyCoord

        assert res.potential == self.potential
        assert len(res.mass) == n

        got = res.mass.sum()
        if hasattr(got, "unit"):
            got = got.to_value(u.solMass)

        expected = self.mass
        if hasattr(expected, "unit"):
            expected = expected.to_value(u.solMass)

        assert np.isclose(got, expected)

        # TODO! value tests when https://github.com/jobovy/galpy/pull/443
        # assert np.allclose(res.ra.deg, [126.10132346, 214.92637031])

    # /def

    @pytest.mark.skip("TODO https://github.com/jobovy/galpy/pull/443")
    def test_specific_call(self):
        assert NotImplementedError("See above.")

    # -------------------------------

    @pytest.mark.parametrize(
        "n,frame,kwargs",
        [
            (10, None, {}),  # just "n"
            (10, "FK5", {}),  # specifying frame
            (10, "FK5", dict(a=1, b=2)),  # adding kwargs
        ],
    )
    def test_sample(self, n, frame, kwargs):
        """Test method ``sample``."""
        res = self.inst.sample(n, frame=frame, **kwargs)
        assert res.__class__ == coord.SkyCoord

        assert res.potential == self.potential
        assert len(res.mass) == n
        # FIXME!
        # assert np.isclose(
        #     res.mass.sum(), self.mass
        # ), f"{res.mass.sum()} != {self.mass}"

        # TODO! value tests when https://github.com/jobovy/galpy/pull/443
        # assert np.allclose(res.ra.deg, [126.10132346, 214.92637031])

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
