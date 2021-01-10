# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.core.sample`."""

__all__ = [
    "Test_PotentialSampler",
]


##############################################################################
# IMPORTS

# BUILT-IN
import unittest
from abc import abstractmethod
from types import MappingProxyType

# THIRD PARTY
import pytest

# PROJECT-SPECIFIC
from discO.core import sample
from discO.core.tests.test_core import Test_PotentialBase

##############################################################################
# TESTS
##############################################################################


class Test_PotentialSampler(Test_PotentialBase, obj=sample.PotentialSampler):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.potential = object()

        # register a unittest examples
        class SubClassUnitTest(cls.obj, package="unittest"):
            pass

        cls.SubClassUnitTest = SubClassUnitTest

        # make instance. It depends
        if cls.obj is sample.PotentialSampler:
            cls.inst = cls.obj(cls.potential, package="unittest")
        else:
            cls.inst = cls.obj(cls.potential)

    # /def

    @classmethod
    def teardown_class(cls):
        """Teardown fixtures for testing."""
        sample.SAMPLER_REGISTRY.pop(unittest, None)

    # /def

    #################################################################
    # Method Tests

    def test___init_subclass__(self):
        """Test subclassing."""
        # When package is None, it is not registered
        class SubClass1(self.obj):
            pass

        assert None not in sample.SAMPLER_REGISTRY
        assert SubClass1 not in sample.SAMPLER_REGISTRY.values()

        # ------------------------
        # register a new
        try:

            class SubClass1(self.obj, package="pytest"):
                pass

        except Exception:
            pass
        finally:
            sample.SAMPLER_REGISTRY.pop(pytest, None)

        # -------------------------------
        # error when already in registry

        try:
            # registered
            class SubClass1(self.obj, package="pytest"):
                pass

            # doing it again raises error
            with pytest.raises(KeyError):

                class SubClass1(self.obj, package="pytest"):
                    pass

        except Exception:
            pass
        finally:  # cleanup
            sample.SAMPLER_REGISTRY.pop(pytest, None)

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

        # The unittest is already registered, so can
        # test for that.
        assert unittest in self.obj._registry.keys()
        assert self.SubClassUnitTest in self.obj._registry.values()
        assert self.obj._registry[unittest] is self.SubClassUnitTest

    # /def

    # -------------------------------

    def test___class_getitem__(self):
        """Test method ``__class_getitem__``."""
        # run tests on super
        super().test___class_getitem__()

        # -------------------------------
        # test a specific item in the registry
        assert self.obj[unittest] is self.SubClassUnitTest

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
        if self.obj is sample.PotentialSampler:

            # ---------------
            # Need the "potential" argument
            with pytest.raises(TypeError) as e:
                self.obj()

            assert (
                "missing 1 required positional argument: 'potential'"
            ) in str(e.value)

            # --------------------------
            # for object not in registry

            with pytest.raises(ValueError) as e:
                self.obj(self.potential)

            assert (
                "PotentialSampler has no registered sampler for package: "
                "<module 'builtins' (built-in)>"
            ) in str(e.value)

            # ---------------
            # with return_specific_class

            package, klass = tuple(self.obj._registry.items())[0]

            msamp = self.obj(
                self.potential, package=package, return_specific_class=True
            )

            # test class type
            assert isinstance(msamp, klass)
            assert isinstance(msamp, self.obj)

            # test inputs
            assert msamp._sampler == self.potential

            # ---------------
            # as wrapper class

            package, klass = tuple(self.obj._registry.items())[0]

            msamp = self.obj(
                self.potential, package=package, return_specific_class=False
            )

            # test class type
            assert not isinstance(msamp, klass)
            assert isinstance(msamp, self.obj)
            assert isinstance(msamp._instance, klass)

            # test inputs
            assert msamp._sampler == self.potential

        # --------------------------
        else:  # never hit in Test_PotentialSampler, only in subs

            # ---------------
            # Can't have the "package" argument

            with pytest.raises(ValueError) as e:
                self.obj(self.potential, package="not None")

            # ---------------
            # AOK

            msamp = self.obj(self.potential, frame="icrs")

            assert self.obj is not sample.PotentialSampler
            assert isinstance(msamp, self.obj)
            assert isinstance(msamp, sample.PotentialSampler)
            assert not hasattr(msamp, "_instance")
            assert msamp._sampler == self.potential

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

    # -------------------------------

    @pytest.mark.skip("TODO")
    def test_sample(self):
        """Test method ``sample``."""

    # /def

    # -------------------------------

    @pytest.mark.skip("TODO")
    def test_resampler(self):
        """Test method ``resampler``."""

    # /def

    # -------------------------------

    @pytest.mark.skip("TODO")
    def test_resample(self):
        """Test method ``resample``."""

    # /def

    # -------------------------------

    def test__preferred_frame_resolve(self):
        """Test method ``_preferred_frame_resolve``."""
        # None -> own frame
        assert self.inst._preferred_frame_resolve(None) == self.inst._frame

        # own frame is passed through
        assert (
            self.inst._preferred_frame_resolve(self.inst._frame)
            == self.inst._frame
        )

        # "icrs"
        assert self.inst._preferred_frame_resolve("icrs") == self.inst._frame

    # /def


# /class


# -------------------------------------------------------------------

# class PotentialSamplerSubClassTests(Test_PotentialSampler):

#     @classmethod
#     def setup_class(cls):
#         """Setup fixtures for testing."""
#         cls.potential = object()

#         # cls.inst = cls.obj(potential, package="GaussianMeasurementErrorSampler")

#     # /def

#     @classmethod
#     def teardown_class(cls):
#         """Teardown fixtures for testing."""
#         pass

#     # /def

#     #################################################################
#     # Method Tests

# # /def

##############################################################################
# END
