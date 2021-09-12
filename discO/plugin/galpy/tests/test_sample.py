# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.plugin.galpy.sample`."""

__all__ = [
    "Test_GalpyPotentialSampler",
    "Test_MeshGridPositionDF",
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

# LOCAL
from discO.core.sample import MeshGridPotentialSampler
from discO.core.tests.test_sample import Test_PotentialSampler as PotentialSampler_Test
from discO.plugin.galpy import GalpyPotentialWrapper, sample
from discO.tests.helper import ObjectTest

##############################################################################
# TESTS
##############################################################################


class Test_GalpyPotentialSampler(
    PotentialSampler_Test,
    obj=sample.GalpyPotentialSampler,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        super().setup_class()

        # make potential
        cls.mass = 1e12 * u.solMass

        hernquist_pot = HernquistPotential(amp=2 * cls.mass, ro=8 * u.kpc, vo=220 * u.km / u.s)
        hernquist_pot.turn_physical_on(ro=8 * u.kpc, vo=220 * u.km / u.s)
        cls.potential = hernquist_pot

        cls.df = isotropicHernquistdf(hernquist_pot)
        cls.df.turn_physical_on(ro=8 * u.kpc, vo=220 * u.km / u.s)

        cls.inst = cls.obj(GalpyPotentialWrapper(cls.potential))

    # /def

    #################################################################
    # Method Tests

    def test_potential(self):
        """Test method ``potential``."""
        assert self.inst.potential is self.inst._wrapper_potential

    # /def

    # -------------------------------

    def test___call__(self):
        """Test method ``__call__``.

        When Test_MeasurementErrorSampler this calls on the wrapped instance,
        which is GaussianMeasurementErrorSampler.

        """
        # run tests on super
        super().test___call__()

    # /def

    @pytest.mark.parametrize(
        "n, frame, representation, random, kwargs",
        [
            (10, None, None, None, {}),  # just "n"
            (10, "FK5", None, None, {}),  # specifying frame
            (10, "FK5", None, None, dict(a=1, b=2)),  # adding kwargs
        ],
    )
    def test_call_parametrize(self, n, frame, representation, random, kwargs):
        """Parametrized call tests."""
        res = self.inst(n, frame=frame, representation_type=representation, **kwargs)
        assert res.__class__ == coord.SkyCoord

        assert res.cache["potential"].__wrapped__ == self.potential
        assert len(res.cache["mass"]) == n

        got = res.cache["mass"].sum()
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
    def test_run(self, n, frame, kwargs):
        """Test method ``run``."""
        res = self.inst.run(n, frame=frame, batch=True, **kwargs)
        assert res.__class__ == coord.SkyCoord

        assert res.cache["potential"].__wrapped__ == self.potential
        assert len(res.cache["mass"]) == n
        # FIXME!
        # assert np.isclose(
        #     res.cache["mass"].sum(), self.mass
        # ), f"{res.cache["mass"].sum()} != {self.mass}"

        # TODO! value tests when https://github.com/jobovy/galpy/pull/443
        # assert np.allclose(res.ra.deg, [126.10132346, 214.92637031])

    # /def


# /class


# ------------------------------------------------------------------------------


class Test_MeshGridPositionDF(
    ObjectTest,
    obj=sample.MeshGridPositionDF,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        # make potential
        cls.mass = 1e12 * u.solMass

        hernquist_pot = HernquistPotential(amp=2 * cls.mass, ro=8 * u.kpc, vo=220 * u.km / u.s)
        hernquist_pot.turn_physical_on(ro=8 * u.kpc, vo=220 * u.km / u.s)
        cls.potential = hernquist_pot

        nx = ny = nz = 76  # must be int and even
        nxr0 = nyr0 = nzr0 = 2.3 * 2

        X, Y, Z = np.array(
            np.meshgrid(
                np.linspace(-nxr0 / 2, nxr0 / 2, nx),
                np.linspace(-nyr0 / 2, nyr0 / 2, ny),
                np.linspace(-nzr0 / 2, nzr0 / 2, nz),
                indexing="ij",
            ),
        )
        XYZ = coord.CartesianRepresentation(X, Y, Z, unit=u.kpc)
        cls.meshgrid = XYZ

        cls.inst = cls.obj(cls.potential, meshgrid=cls.meshgrid)

    # /def

    #################################################################
    # Method Tests

    def test___init__(self):
        inst = self.obj(self.potential, meshgrid=self.meshgrid)

        # THIRD PARTY
        from galpy.df.df import df as DF

        assert isinstance(inst, DF)

    # /def

    def test__pot(self):
        assert self.inst._pot is self.inst._sampler._wrapper_potential

    # /def

    def test_sample(self):
        """Test method ``sample``.

        The method just calls ``MeshGridPotentialSampler``, tested elsewhere.

        """
        comparison_sampler = MeshGridPotentialSampler(
            GalpyPotentialWrapper(self.potential),
            self.meshgrid,
        )

        self.inst.sample(10) == comparison_sampler(10)

    # /def

    #################################################################
    # Usage Tests

    @pytest.mark.skip(reason="TODO")
    def test_use_as_galpy_DF(self):
        assert False

    # /def


# /class


##############################################################################
# END
