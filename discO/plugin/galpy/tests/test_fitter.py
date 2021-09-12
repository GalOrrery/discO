# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.plugin.galpy.fitter`."""

__all__ = [
    "Test_GalpyPotentialFitter",
    "Test_GalpySCFPotentialFitter",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.units as u
import pytest
from galpy import potential as gpot

# LOCAL
from discO.core.tests.test_fitter import Test_PotentialFitter as PotentialFitterTester
from discO.plugin.galpy import GalpyPotentialWrapper, fitter

##############################################################################
# TESTS
##############################################################################


class Test_GalpyPotentialFitter(
    PotentialFitterTester,
    obj=fitter.GalpyPotentialFitter,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.potential = gpot.Potential

        # register a unittest examples
        class SubClassUnitTest(cls.obj, key="unittest"):
            def __init__(
                self,
                potential_cls,
                frame=None,
                **kwargs,
            ):
                super().__init__(potential_cls=potential_cls, frame=frame, **kwargs)

            # /defs

            def __call__(self, c, **kwargs):
                return GalpyPotentialWrapper(
                    gpot.Potential(),
                    frame=self.frame,
                )

            # /def

        cls.SubClassUnitTest = SubClassUnitTest
        # /class

        # make instance. It depends.
        if cls.obj is fitter.GalpyPotentialFitter:
            cls.inst = cls.obj(
                potential_cls=cls.potential,
                key="unittest",
                frame="galactocentric",
            )

    # /def

    #################################################################
    # Method Tests

    def test___new__(self):
        """Test method ``__new__``.

        This is a wrapper class that acts differently when instantiating
        a MeasurementErrorSampler than one of it's subclasses.

        """
        # there are no tests on super
        # super().test___new__()

        # --------------------------
        if self.obj is fitter.GalpyPotentialFitter:

            # --------------------------
            # for object not in registry

            with pytest.raises(ValueError, match="fitter for key: None"):
                self.obj(potential_cls=None, key=None)

            # ---------------
            # as wrapper

            klass = self.obj._registry["unittest"]

            msamp = self.obj(potential_cls=gpot.Potential, key="unittest")

            # test class type
            assert isinstance(msamp, klass)
            assert isinstance(msamp, self.obj)

            # test inputs
            assert msamp._potential_cls == self.potential

            # ---------------
            # key is not None

            with pytest.raises(ValueError, match="Can't specify 'key'"):
                self.obj.__new__(self.SubClassUnitTest, key="not None")

        # --------------------------
        else:  # never hit in Test_PotentialSampler, only in subs

            # ---------------
            # AOK

            msamp = self.obj()

            assert self.obj is not fitter.PotentialFitter
            assert isinstance(msamp, self.obj)
            assert isinstance(msamp, fitter.PotentialFitter)
            assert not hasattr(msamp, "_instance")
            assert msamp._potential_cls == self.potential

    # /def

    # -------------------------------

    def test___call__(self):
        """Test method ``__call__``."""
        # run tests on super
        super().test___call__()

        if self.obj is fitter.GalpyPotentialFitter:

            with pytest.raises(NotImplementedError, match="Implement in sub"):
                self.obj.__call__(self.inst, None)

        # TODO! actually run tests

    # /def


# /class


# -------------------------------------------------------------------


class Test_GalpySCFPotentialFitter(
    Test_GalpyPotentialFitter,
    obj=fitter.GalpySCFPotentialFitter,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        super().setup_class()
        cls.potential = gpot.SCFPotential
        cls.inst = cls.obj(frame="galactocentric", Nmax=4, Lmax=3)

    # /def

    # -------------------------------

    def test___call__(self):
        """Test method ``__call__``."""
        # run tests on super
        super().test___call__()

        # -------------------
        # some errors

        with pytest.raises(AttributeError, match="has no attribute 'cache'"):
            self.inst(None, Nmax=0)

        with pytest.raises(ValueError, match="Nmax & Lmax must be >=0."):
            self.inst(None, Nmax=-1, mass=1 * u.solMass)

        with pytest.raises(ValueError, match="Nmax & Lmax must be >=0."):
            self.inst(None, Lmax=-1, mass=1 * u.solMass)

        with pytest.raises(u.UnitsError, match="length or be dimensionless"):
            self.inst(None, scale_factor=2 * u.Hz, mass=1 * u.solMass)

        with pytest.raises(ValueError, match="scale factor must be a scalar."):
            self.inst(None, scale_factor=[1, 2] * u.km, mass=1 * u.solMass)

        # TODO! actually run tests

    # /def


# /class


##############################################################################
# END
