# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.core.measurement`."""

__all__ = [
    "Test_MeasurementErrorSampler",
    "Test_GaussianMeasurementErrorSampler",
]


##############################################################################
# IMPORTS

# BUILT-IN
from abc import abstractmethod
from types import MappingProxyType

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest

# PROJECT-SPECIFIC
from discO.core import measurement
from discO.core.tests.test_core import Test_PotentialBase

##############################################################################
# TESTS
##############################################################################


class Test_MeasurementErrorSampler(
    Test_PotentialBase,
    obj=measurement.MeasurementErrorSampler,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.c = coord.SkyCoord(
            coord.ICRS(ra=[1, 2] * u.deg, dec=[2, 3] * u.deg),
        )
        cls.c_err = coord.SkyCoord(
            coord.ICRS(ra=[0.1, 0.2] * u.deg, dec=[0.2, 0.3] * u.deg),
        )

        cls.inst = cls.obj(cls.c_err, method="GaussianMeasurementErrorSampler")

    # /def

    #################################################################
    # Method Tests

    def test___init_subclass__(self):
        """Test subclassing."""
        # can't run tests on super b/c doesn't accept "package"
        # super().test___init_subclass__()

        # -------------------------------
        try:
            # registered by name
            class SubClass1(self.obj):
                pass

            assert not hasattr(SubClass1, "_key")
            assert "SubClass1" in measurement.MEASURE_REGISTRY
        except Exception:
            pass
        finally:  # cleanup
            measurement.MEASURE_REGISTRY.pop("SubClass1", None)

        # -------------------------------
        # error when already in registry

        try:
            # registered by name
            class SubClass1(self.obj):
                pass

            # doing it again raises error
            with pytest.raises(KeyError):

                class SubClass1(self.obj):
                    pass

        except Exception:
            pass
        finally:  # cleanup
            measurement.MEASURE_REGISTRY.pop("SubClass1", None)

    # /def

    # -------------------------------

    def test__registry(self):
        """Test method ``_registry``.

        As ``_registry`` is never overwritten in the subclasses, this test
        should carry though.

        """
        # run tests on super
        super().test__registry()

        # -------------------------------
        assert isinstance(self.obj._registry, MappingProxyType)

        # The GaussianMeasurementErrorSampler is already registered, so can
        # test for that.
        assert "GaussianMeasurementErrorSampler" in self.obj._registry
        assert (
            self.obj._registry["GaussianMeasurementErrorSampler"]
            is measurement.GaussianMeasurementErrorSampler
        )

    # /def

    # -------------------------------

    def test___class_getitem__(self):
        """Test method ``__class_getitem__``."""
        # run tests on super
        super().test___class_getitem__()

        # -------------------------------
        # test a specific item in the registry
        assert (
            self.obj["GaussianMeasurementErrorSampler"]
            is measurement.GaussianMeasurementErrorSampler
        )

    # /def

    # -------------------------------

    def test___new__(self):
        """Test method ``__new__``.

        This is a wrapper class that acts differently when instantiating
        a MeasurementErrorSampler than one of it's subclasses.

        """
        # there are no tests on super
        # super().test___new__()

        # --------------------------
        if self.obj is measurement.MeasurementErrorSampler:

            # ---------------
            # Need the "method" argument
            with pytest.raises(ValueError) as e:
                self.obj()

            assert (
                "MeasurementErrorSampler has no "
                "registered measurement resampler"
            ) in str(e.value)

            # ---------------
            # with return_specific_class

            method, klass = tuple(self.obj._registry.items())[0]

            msamp = self.obj(
                c_err=self.c_err,
                method=method,
                return_specific_class=True,
            )

            # test class type
            assert isinstance(msamp, klass)
            assert isinstance(msamp, self.obj)

            # test inputs
            assert all(msamp.c_err == self.c_err)

            # ---------------
            # as wrapper class

            method, klass = tuple(self.obj._registry.items())[0]

            msamp = self.obj(
                c_err=self.c_err,
                method=method,
                return_specific_class=False,
            )

            # test class type
            assert not isinstance(msamp, klass)
            assert isinstance(msamp, self.obj)
            assert isinstance(msamp._instance, klass)

            # test inputs
            assert all(msamp.c_err == self.c_err)

        # --------------------------
        else:  # never hit in Test_MeasurementErrorSampler, only in subs

            # ---------------
            # Can't have the "method" argument

            with pytest.raises(ValueError) as e:
                self.obj(method="not None")

            # ---------------
            # warns on return_specific_class

            with pytest.warns(UserWarning):
                self.obj(method=None, return_specific_class=True)

            # ---------------
            # AOK

            msamp = self.obj(
                c_err=self.c_err,
                method=None,
                return_specific_class=False,
            )

            assert self.obj is not measurement.MeasurementErrorSampler
            assert isinstance(msamp, self.obj)
            assert isinstance(msamp, measurement.MeasurementErrorSampler)
            assert not hasattr(msamp, "_instance")
            assert all(msamp.c_err == self.c_err)

    # /def

    # -------------------------------

    @abstractmethod
    def test___init__(self):
        """Test method ``__init__``."""
        # run tests on super
        super().test___init__()

        # --------------------------
        pass  # for subclasses. The setup_class actually tests this for here.

    # /def

    # -------------------------------

    @abstractmethod
    def test___call__(self):
        """Test method ``__call__``.

        When Test_MeasurementErrorSampler this calls on the wrapped instance,
        which is GaussianMeasurementErrorSampler.

        Subclasses should do real tests on the output. This only tests
        that we can even call the method.

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

    #################################################################
    # Pipeline Tests

    # N/A b/c abstract base-class


# /class


# -------------------------------------------------------------------


class Test_GaussianMeasurementErrorSampler(
    Test_MeasurementErrorSampler,
    obj=measurement.GaussianMeasurementErrorSampler,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.inst = cls.obj(c_err=cls.c_err)

    # /def

    # -------------------------------

    @abstractmethod
    def test___init__(self):
        """Test method ``__init__``."""
        # run tests on super
        super().test___init__()

        # --------------------------
        #  The setup_class actually tests this for here.
        assert hasattr(self.inst, "c_err")

    # /def

    # -------------------------------

    def test___call__(self):
        """Test method ``__call__``."""
        super().test___call__()
        # --------------------------
        # just "c", cannot predict "random"

        res = self.inst(self.c)
        assert res.frame.__class__ == self.c.frame.__class__  # same class

        # --------------------------
        # test "random" by setting the seed
        # doing this here b/c want to control random for all the rest.

        expected_ra = [1.01257302, 2.02098002]
        expected_dec = [1.97357903, 2.83929919]
        expected_dist = [1.64042265, 1.36159505]

        res = self.inst(self.c, random=0)
        assert np.allclose(res.ra.deg, expected_ra)
        assert np.allclose(res.dec.deg, expected_dec)
        assert np.allclose(res.distance, expected_dist)

        res = self.inst(self.c, random=np.random.default_rng(0))
        assert np.allclose(res.ra.deg, expected_ra)
        assert np.allclose(res.dec.deg, expected_dec)
        assert np.allclose(res.distance, expected_dist)

        # --------------------------
        # "c" and "c_err" | random

        res = self.inst(self.c, self.c_err, random=0)
        assert np.allclose(res.ra.deg, expected_ra)
        assert np.allclose(res.dec.deg, expected_dec)
        assert np.allclose(res.distance, expected_dist)

        # --------------------------
        # "c" and "c_err", c_err is BaseCoordinateFrame, not SkyCoord | random

        res = self.inst(self.c, self.c_err.frame, random=0)
        assert np.allclose(res.ra.deg, expected_ra)
        assert np.allclose(res.dec.deg, expected_dec)
        assert np.allclose(res.distance, expected_dist)

        # --------------------------
        # "c" and "c_err", c_err is BaseRepresentation | random

        res = self.inst(
            self.c,
            self.c_err.represent_as(coord.SphericalRepresentation),
            random=0,
        )

        assert np.allclose(res.ra.deg, expected_ra)
        assert np.allclose(res.dec.deg, expected_dec)
        assert np.allclose(res.distance, expected_dist)

        # --------------------------
        # "c" and c_err, c_err is scalar | random

        res = self.inst(self.c, 0.1, random=0)
        assert np.allclose(res.ra.deg, [1.01257302, 2.01049001])
        assert np.allclose(res.dec.deg, [1.98678951, 2.94643306])
        assert np.allclose(res.distance, [1.06404227, 1.03615951])

        # --------------------------
        # "c" and c_err, c_err is callable | random

        res = self.inst(self.c, lambda c: 0.1, random=0)
        assert np.allclose(res.ra.deg, [1.01257302, 2.01049001])
        assert np.allclose(res.dec.deg, [1.98678951, 2.94643306])
        assert np.allclose(res.distance, [1.06404227, 1.03615951])

        # ------------------
        # :fun:`~discO.core.measurement.xpercenterror_factory`

        xpercenterror = measurement.xpercenterror_factory(10 * u.percent)

        res = self.inst(self.c, xpercenterror, random=0)
        assert np.allclose(res.ra.deg, [1.01257302, 2.02098002])
        assert np.allclose(res.dec.deg, [1.97357903, 2.83929919])
        assert np.allclose(res.distance, [1.06404227, 1.03615951])

        # ------------------
        # a raw percent scalar has the same effect

        res = self.inst(self.c, 10 * u.percent, random=0)
        assert np.allclose(res.ra.deg, [1.01257302, 2.02098002])
        assert np.allclose(res.dec.deg, [1.97357903, 2.83929919])
        assert np.allclose(res.distance, [1.06404227, 1.03615951])

        # --------------------------
        # "c" and c_err, c_err is Mapping | random

        with pytest.raises(NotImplementedError):
            self.inst(self.c, {}, random=0)

        # --------------------------
        # "c" and c_err, c_err is none of the above | random

        with pytest.raises(NotImplementedError):

            self.inst(self.c, c_err=Exception(), random=0)

    # /def

    #################################################################
    # Pipeline Tests

    @pytest.mark.skip("TODO")
    def test_Sampler_to_MeasurementSampler(self):
        pass

    # /def


# /class


##############################################################################
# Test Utils


def test_xpercenterror_factory():
    """Test :fun:`~discO.core.measurement.xpercenterror_factory`."""
    rep = coord.SphericalRepresentation(1 * u.rad, 1 * u.rad, 1 * u.kpc)
    crd = coord.ICRS(rep.reshape(-1, 1))

    # --------------------------
    # percent input

    func = measurement.xpercenterror_factory(10 * u.percent)
    res = func(crd)

    assert callable(func)
    assert np.allclose(res, [0.1, 0.1, 0.1])

    # --------------------------
    # fractional error input

    func2 = measurement.xpercenterror_factory(0.1)
    res2 = func2(crd)

    assert callable(func2)
    assert np.allclose(res2, [0.1, 0.1, 0.1])

    # --------------------------
    # caching

    assert measurement.xpercenterror_factory(10 * u.percent) is func
    assert measurement.xpercenterror_factory(0.1) is func2


# /def


##############################################################################
# END
