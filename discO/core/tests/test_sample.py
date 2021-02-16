# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.core.sample`."""

__all__ = [
    "Test_PotentialSampler",
]


##############################################################################
# IMPORTS

# BUILT-IN
import abc
import contextlib

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest

# PROJECT-SPECIFIC
from discO.core import sample
from discO.core.tests.test_core import Test_CommonBase as CommonBase_Test
from discO.utils.random import NumpyRNGContext

##############################################################################
# TESTS
##############################################################################


class Test_PotentialSampler(CommonBase_Test, obj=sample.PotentialSampler):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.potential = object()

        # register a unittest examples
        class SubClassUnitTest(cls.obj, key="unittest"):
            def __call__(
                self,
                n,
                *,
                frame=None,
                representation_type=None,
                random=None,
                **kwargs
            ):
                # Get preferred frames
                frame = self._infer_frame(frame)
                representation_type = self._infer_representation(
                    representation_type,
                )

                if random is None:
                    random = np.random
                elif isinstance(random, int):
                    random = np.random.default_rng(random)

                # return
                rep = coord.UnitSphericalRepresentation(
                    lon=random.uniform(size=n) * u.deg,
                    lat=2 * random.uniform(size=n) * u.deg,
                )

                if representation_type is None:
                    representation_type = rep.__class__
                sample = coord.SkyCoord(
                    frame.realize_frame(
                        rep,
                        representation_type=representation_type,
                    ),
                    copy=False,
                )
                sample.mass = np.ones(n)
                sample.potential = cls.potential

                return sample

        cls.SubClassUnitTest = SubClassUnitTest
        # /class

        # make instance. It depends.
        if cls.obj is sample.PotentialSampler:
            cls.inst = cls.obj(cls.potential, key="unittest")

    # /def

    @classmethod
    def teardown_class(cls):
        """Teardown fixtures for testing."""
        cls.SubClassUnitTest._registry.pop("unittest", None)

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

            class SubClass1(self.obj, key="pytest"):
                pass

        except Exception:
            pass
        finally:
            sample.SAMPLER_REGISTRY.pop("pytest", None)

        # -------------------------------
        # error when already in registry

        try:
            # registered
            class SubClass1(self.obj, key="pytest"):
                pass

            # doing it again raises error
            with pytest.raises(KeyError):

                class SubClass1(self.obj, key="pytest"):
                    pass

        except Exception:
            pass
        finally:  # cleanup
            sample.SAMPLER_REGISTRY.pop("pytest", None)

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
        assert isinstance(self.obj._registry, dict)

        # The unittest is already registered, so can
        # test for that.
        assert "unittest" in self.obj._registry.keys()
        assert self.SubClassUnitTest in self.obj._registry.values()
        assert self.obj._registry["unittest"] is self.SubClassUnitTest

    # /def

    # -------------------------------

    def test___class_getitem__(self):
        """Test method ``__class_getitem__``."""
        # run tests on super
        super().test___class_getitem__()

        # -------------------------------
        # test a specific item in the registry
        assert self.obj["unittest"] is self.SubClassUnitTest

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
                "PotentialSampler has no registered sampler for key: builtins"
            ) in str(e.value)

            # ---------------
            # with return_specific_class

            key, klass = tuple(self.obj._registry.items())[0]

            msamp = self.obj(self.potential, key=key)

            # test class type
            assert isinstance(msamp, klass)
            assert isinstance(msamp, self.obj)

            # test inputs
            assert msamp._potential == self.potential

        # --------------------------
        else:  # never hit in Test_PotentialSampler, only in subs

            # ---------------
            # Can't have the "key" argument

            with pytest.raises(ValueError) as e:
                self.obj(self.potential, key="not None")

            assert "Can't specify 'key'" in str(e.value)

            # ---------------
            # AOK

            msamp = self.obj(self.potential, frame="icrs")

            assert self.obj is not sample.PotentialSampler
            assert isinstance(msamp, self.obj)
            assert isinstance(msamp, sample.PotentialSampler)
            assert not hasattr(msamp, "_instance")
            assert msamp._potential == self.potential

    # /def

    # -------------------------------

    @abc.abstractmethod
    def test___init__(self):
        """Test method ``__init__``."""
        # run tests on super
        super().test___init__()

        # --------------------------
        pass  # for subclasses. The setup_class actually tests this for here.

    # /def

    # -------------------------------

    def test_potential(self):
        """Test method ``potential``."""
        assert self.inst.potential.__wrapped__ is self.potential

    # /def

    # -------------------------------

    def test_frame(self):
        """Test method ``frame``."""
        assert self.inst.frame is self.inst.potential.frame

    # /def

    # -------------------------------

    def test_representation_type(self):
        """Test method ``representation_type``."""
        assert (
            self.inst.representation_type
            is self.inst.potential.representation_type
        )

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

        # raises error if called
        if self.obj is sample.PotentialSampler:

            with pytest.raises(NotImplementedError) as e:
                self.obj.__call__(self.inst)

            assert "Implemented in subclass." in str(e.value)

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
        assert isinstance(res, coord.SkyCoord)

    # /def

    # -------------------------------

    @pytest.mark.parametrize(
        "n, niter, frame, representation, random, kwargs",
        [
            (2, 1, None, None, None, {}),  # basic
            (2, 1, "FK5", None, None, {}),  # specifying frame
            (2, 1, None, None, None, {}),  # sample axis
            (2, 1, None, None, np.random.default_rng(0), {}),  # random
            (2, 1, None, None, None, dict(a=1, b=2)),  # adding kwargs
            (2, 10, None, None, None, {}),  # larger niters
            ((1, 2), 1, None, None, None, {}),  # array of n
            ((1, 2), 2, None, None, None, {}),  # niters and array of n
        ],
    )
    def test_sample(self, n, niter, frame, representation, random, kwargs):
        """Test method ``resample``."""
        # print(n, niter, frame, representation, random, kwargs)
        samples = self.inst.sample(
            n=n,
            niter=niter,
            frame=frame,
            representation_type=representation,
            random=random,
            **kwargs
        )
        if isinstance(samples, np.ndarray):
            for s, n_ in zip(samples, n):
                if niter == 1:
                    assert s.shape == (n_,)  # correct shape
                elif n_ == 1:  # niter != 1
                    assert s.shape == (n_, niter)  # correct shape

        elif niter == 1:
            assert samples.shape == (n,)  # correct shape
        else:
            assert samples.shape == (n, niter)  # correct shape

    # /def

    def test_sample_error(self):
        """Test method ``resample`` raises error."""
        with pytest.raises(ValueError):
            self.inst.sample(10, 0)

    # /def

    # -------------------------------

    def test__infer_frame(self):
        """Test method ``_infer_frame``."""
        # None -> own frame
        assert self.inst._infer_frame(None) == self.inst.potential.frame

        # own frame is passed through
        assert (
            self.inst._infer_frame(self.inst.potential.frame)
            == self.inst.potential.frame
        )

        # "icrs"
        assert self.inst._infer_frame("icrs") == coord.ICRS()

    # /def

    # -------------------------------

    def test__infer_representation(self):
        """Test method ``_infer_representation``."""
        # ----------------
        # None -> own frame

        assert (
            self.inst._infer_representation(None)
            == self.inst.potential.representation_type
        )

        # ----------------
        # still None

        old_representation_type = self.inst.representation_type
        self.inst.potential._representation_type = None
        assert self.inst._infer_representation(None) is None
        self.inst.potential._representation_type = old_representation_type

        # ----------------

        assert (
            self.inst._infer_representation(coord.CartesianRepresentation)
            is coord.CartesianRepresentation
        )

        assert (
            self.inst._infer_representation(
                coord.CartesianRepresentation((1, 2, 3)),
            )
            == coord.CartesianRepresentation
        )

        assert (
            self.inst._infer_representation("cartesian")
            == coord.CartesianRepresentation
        )

    # /def

    def test__random_context(self):
        """Test method ``_random_context``.

        contents are tested elsewhere, only need to test here that it returns
        the expected stuff.

        """
        # ----------------
        # int or randomState

        ctx = self.inst._random_context(0)
        assert isinstance(ctx, NumpyRNGContext)

        ctx = self.inst._random_context(np.random.RandomState(0))
        assert isinstance(ctx, NumpyRNGContext)

        # ----------------
        # else

        ctx = self.inst._random_context(None)
        assert isinstance(ctx, contextlib.nullcontext)

        ctx = self.inst._random_context(np.random.default_rng(0))
        assert isinstance(ctx, contextlib.nullcontext)

    # /def

    #################################################################
    # Usage Tests


# /class


# -------------------------------------------------------------------


##############################################################################
# END
