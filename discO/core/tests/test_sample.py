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
from discO.core.tests.test_common import Test_CommonBase as CommonBase_Test
from discO.core.wrapper import PotentialWrapper
from discO.utils.random import NumpyRNGContext
from discO.setup_package import HAS_GALPY

##############################################################################
# TESTS
##############################################################################


class TestDF(object):
    """docstring for TestDF"""

    def __init__(self, potential):
        self._pot = potential

    # /def


# /class


##############################################################################


class Test_PotentialSampler(CommonBase_Test, obj=sample.PotentialSampler):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.potential = object()

        # register a unittest examples
        class SubClassUnitTest(cls.obj, key="unittest"):
            def __call__(self, n, *, random=None, **kwargs):
                # Get preferred frames
                frame = self.frame
                representation_type = self.representation_type

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
                sample.potential = self.potential

                return sample

        cls.SubClassUnitTest = SubClassUnitTest
        # /class

        # make instance. It depends.
        if cls.obj is sample.PotentialSampler:
            cls.inst = cls.obj(
                PotentialWrapper(cls.potential),
                key="unittest",
                total_mass=10 * u.solMass,
            )

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
        # Need the "potential" argument

        # potential must be a PotentialWrapper
        with pytest.raises(TypeError, match="must be a PotentialWrapper"):
            self.obj(object)

        # --------------------------
        if self.obj is sample.PotentialSampler:

            # ---------------
            # Need the "potential" argument
            with pytest.raises(TypeError, match="argument: 'potential'"):
                self.obj()

            # --------------------------
            # for object not in registry

            with pytest.raises(ValueError, match="key: builtins"):
                self.obj(PotentialWrapper(self.potential))

            # ---------------
            # with subclass

            try:  # see if galpy is working
                key = "galpy"
                klass = self.obj._registry[key]
            except KeyError:
                key, klass = tuple(self.obj._registry.items())[0]
                potential = self.potential
                msamp = self.obj(
                    PotentialWrapper(potential),
                    total_mass=10 * u.solMass,
                    key=key,
                )
            else:
                msamp = self.obj(
                    PotentialWrapper(self.potential),
                    total_mass=10 * u.solMass,
                    df=TestDF,
                    key=key,
                )

            # test class type
            assert isinstance(msamp, klass)
            assert isinstance(msamp, self.obj)

            # test inputs
            assert msamp._potential == self.potential

            # ---------------
            # potential is PotentialWrapper

            try:
                key = "galpy"
                klass = self.obj._registry[key]
            except KeyError:
                key, klass = tuple(self.obj._registry.items())[0]
                potential = self.potential
                msamp = self.obj(
                    PotentialWrapper(potential),
                    total_mass=10 * u.solMass,
                    key=key,
                )
            else:
                msamp = self.obj(
                    PotentialWrapper(self.potential),
                    key=key,
                    df=TestDF,
                    total_mass=10 * u.solMass,
                )

            # test class type
            assert isinstance(msamp, klass)
            assert isinstance(msamp, self.obj)

            # test inputs
            assert msamp._potential == self.potential

        # --------------------------
        else:  # never hit in Test_PotentialSampler, only in subs

            # ---------------
            # Can't have the "key" argument

            with pytest.raises(ValueError, match="Can't specify 'key'"):
                self.obj(PotentialWrapper(self.potential), key="not None")

            # ---------------
            # AOK

            msamp = self.obj(PotentialWrapper(self.potential, frame="icrs"))

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

        if self.obj is not sample.PotentialSampler:

            with pytest.raises(ValueError, match="mass is divergent"):
                self.obj(
                    PotentialWrapper(self.potential),
                    total_mass=np.inf * u.solMass,
                )

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

            with pytest.raises(NotImplementedError, match="in subclass."):
                self.obj.__call__(self.inst)

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
        "n, niter, random, kwargs",
        [
            (2, 1, None, {}),  # basic
            (2, 1, None, {}),  # specifying frame
            (2, 1, None, {}),  # sample axis
            (2, 1, np.random.RandomState(0), {}),  # random
            (2, 1, None, dict(a=1, b=2)),  # adding kwargs
            (2, 10, None, {}),  # larger niters
            # ((1, 2), 1, None, {}),  # array of n
            # ((1, 2), 2, None, {}),  # niters and array of n
        ],
    )
    def test_run(self, n, niter, random, kwargs):
        """Test method ``run``."""
        samples = self.inst.run(
            n=n, iterations=niter, random=random, batch=True, **kwargs
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
        """Test method ``run`` raises error."""
        with pytest.raises(ValueError):
            self.inst.run(10, 0)

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
        assert (
            self.inst._infer_representation(None)
            == self.inst.frame.default_representation
        )
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
        assert isinstance(ctx, contextlib.suppress)

        ctx = self.inst._random_context(np.random.default_rng(0))
        assert isinstance(ctx, contextlib.suppress)

    # /def

    #################################################################
    # Usage Tests


# /class


# -------------------------------------------------------------------


@pytest.mark.skipif(not HAS_GALPY, reason="needs real density function.")
class Test_MeshGridPotentialSampler(
    Test_PotentialSampler, obj=sample.MeshGridPotentialSampler
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        super().setup_class()

        nx = ny = nz = 76  # must be int and even
        nxr0 = nyr0 = nzr0 = 2.3 * 2

        X, Y, Z = (
            np.array(
                np.meshgrid(
                    np.linspace(-nxr0 / 2, nxr0 / 2, nx),
                    np.linspace(-nyr0 / 2, nyr0 / 2, ny),
                    np.linspace(-nzr0 / 2, nzr0 / 2, nz),
                    indexing="ij",
                )
            )
            * 1
        )
        XYZ = coord.CartesianRepresentation(X, Y, Z, unit=u.kpc)

        import galpy.potential as gpot

        cls.potential = gpot.HernquistPotential()
        cls.meshgrid = XYZ

        cls.inst = cls.obj(
            PotentialWrapper(cls.potential),
            cls.meshgrid,
            total_mass=10 * u.solMass,
        )

    #################################################################
    # Method Tests

    def test___new__(self):
        """Test method ``__new__``."""

        # ---------------
        # Can't have the "key" argument

        with pytest.raises(ValueError, match="Can't specify 'key'"):
            self.obj(
                PotentialWrapper(self.potential), self.meshgrid, key="not None"
            )

        # ---------------
        # AOK

        msamp = self.obj(
            PotentialWrapper(self.potential, frame="icrs"), self.meshgrid
        )

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

        with pytest.raises(ValueError, match="mass is divergent"):
            self.obj(
                PotentialWrapper(self.potential),
                meshgrid=self.meshgrid,
                total_mass=np.inf * u.solMass,
            )

    # /def


# /class


##############################################################################
# END
