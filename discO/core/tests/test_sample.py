# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.core.sample`."""

__all__ = [
    "Test_PotentialSampler",
]


##############################################################################
# IMPORTS

# BUILT-IN
import itertools
import unittest
from abc import abstractmethod
from types import MappingProxyType

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
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
            def __call__(self, n, *, frame=None, random=None, **kwargs):
                # Get preferred frames
                frame = self._preferred_frame_resolve(frame)

                if random is None:
                    random = np.random
                elif isinstance(random, int):
                    random = np.random.default_rng(random)

                # return
                return coord.SkyCoord(
                    coord.ICRS(
                        ra=random.uniform(size=n) * u.deg,
                        dec=2 * random.uniform(size=n) * u.deg,
                    ),
                ).transform_to(frame)

        cls.SubClassUnitTest = SubClassUnitTest

        # make instance. It depends
        if cls.obj is sample.PotentialSampler:
            cls.inst = cls.obj(cls.potential, package="unittest")

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
                self.potential,
                package=package,
                return_specific_class=True,
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
                self.potential,
                package=package,
                return_specific_class=False,
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

    def test___call__(self):
        """Test method ``__call__``.

        When Test_MeasurementErrorSampler this calls on the wrapped instance,
        which is GaussianMeasurementErrorSampler.

        We can't test the output, but can test that it "works".

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

    # /def

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

    # /def

    # -------------------------------

    @pytest.mark.parametrize(
        "niter, n, frame, sample_axis, random, kwargs",
        [
            (1, 2, None, -1, None, {}),  # basic
            (1, 2, "FK5", -1, None, {}),  # specifying frame
            (1, 2, None, 0, None, {}),  # sample axis
            (1, 2, None, 1, None, {}),  # sample axis
            (1, 2, None, -1, 0, {}),  # random
            (1, 2, None, -1, np.random.default_rng(0), {}),  # random
            (1, 2, None, -1, 0, dict(a=1, b=2)),  # adding kwargs
            (10, 2, None, -1, None, {}),  # larger niters
            (1, (1, 2), None, -1, None, {}),  # array of n
            (2, (1, 2), None, -1, None, {}),  # niters and array of n
        ],
    )
    def test_resampler(self, niter, n, frame, sample_axis, random, kwargs):
        """Test method ``resampler``."""
        resampler = self.inst.resampler(
            niter,
            n,
            frame=frame,
            sample_axis=sample_axis,
            random=random,
            **kwargs
        )

        # ------------

        resolve_frame = self.inst._preferred_frame_resolve

        iterniter = range(0, niter)
        if np.isscalar(n):
            itersamp = (n,)
        else:
            itersamp = n
        values = (iterniter, itersamp)
        values = (values, values[::-1])[sample_axis]
        Ns = [(j, i)[sample_axis] for i, j in itertools.product(*values)]

        for i, samp in enumerate(resampler):
            assert len(samp) == Ns[i]
            assert samp.frame.__class__() == resolve_frame(frame)

        # only test the last one b/c want overall len.
        assert (i + 1) == niter * len(itersamp)

    # /def

    # TODO! need to test the iteration order, and stuff in resampler

    # -------------------------------

    @pytest.mark.parametrize(
        "niter, n, frame, random, kwargs",
        [
            (1, 2, None, None, {}),  # basic
            (1, 2, "FK5", None, {}),  # specifying frame
            (1, 2, None, None, {}),  # sample axis
            (1, 2, None, None, {}),  # sample axis
            (1, 2, None, None, {}),  # random
            (1, 2, None, np.random.default_rng(0), {}),  # random
            (1, 2, None, None, dict(a=1, b=2)),  # adding kwargs
            (10, 2, None, None, {}),  # larger niters
            (1, (1, 2), None, None, {}),  # array of n
            (2, (1, 2), None, None, {}),  # niters and array of n
        ],
    )
    def test_resample(self, niter, n, frame, random, kwargs):
        """Test method ``resample``."""
        samples = self.inst.resample(
            niter, n, frame=frame, random=random, **kwargs
        )
        if isinstance(samples, np.ndarray):
            for s, n_ in zip(samples, n):
                if niter == 1:
                    assert s.shape == (n_,)  # correct shape
                elif n_ == 1:  # niter != 1
                    assert s.shape == (niter, n_)  # correct shape

        elif niter == 1:
            assert samples.shape == (n,)  # correct shape
        else:
            assert samples.shape == (niter, n)  # correct shape

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
