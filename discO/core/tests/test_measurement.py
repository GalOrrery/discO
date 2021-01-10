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
import pytest

# PROJECT-SPECIFIC
from discO.core import measurement
from discO.core.tests.test_core import Test_PotentialBase

##############################################################################
# TESTS
##############################################################################


class Test_MeasurementErrorSampler(
    Test_PotentialBase, obj=measurement.MeasurementErrorSampler
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.c = coord.ICRS(ra=[1, 2] * u.deg, dec=[2, 3] * u.deg)
        cls.c_err = coord.ICRS(ra=[0.1, 0.2] * u.deg, dec=[0.2, 0.3] * u.deg)

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

            assert not hasattr(SubClass1, "_package")
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
                c_err=self.c_err, method=method, return_specific_class=True
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
                c_err=self.c_err, method=method, return_specific_class=False
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
                c_err=self.c_err, method=None, return_specific_class=False
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

    @pytest.mark.skip("TODO")
    def test___call__(self):
        """Test method ``__call__``."""
        super().test___call__()
        # --------------------------
        # just "c"

        self.inst(self.c)

        # --------------------------
        # test "random"
        # doing this here b/c want to control random for all the rest.

        # --------------------------
        # just "c" | random

        # --------------------------
        # "c" and c_err | random

        # --------------------------
        # "c" and c_err, c_err is BaseCoordinateFrame, not SkyCoord | random

        # --------------------------
        # "c" and c_err, c_err is BaseRepresentation | random

        with pytest.raises(NotImplementedError):

            self.inst(
                self.c, self.c_err.represent_as(coord.SphericalRepresentation)
            )

        # --------------------------
        # "c" and c_err, c_err is scalar | random

        self.inst(self.c, 0.1)

        # --------------------------
        # "c" and c_err, c_err is callable | random

        self.inst(self.c, lambda c: 0.1)

        # --------------------------
        # "c" and c_err, c_err is none of the above | random

        with pytest.raises(NotImplementedError):

            self.inst(self.c, c_err=Exception())

    # /def

    #################################################################
    # Pipeline Tests

    @pytest.mark.skip("TODO")
    def test_Sampler_to_MeasurementSampler(self):
        pass

    # /def


# /class


##############################################################################
# END
