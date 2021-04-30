# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.core.pipeline`."""

__all__ = [
    "Test_Pipeline",
    "Test_PipelineResult",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest

# PROJECT-SPECIFIC
from discO.core import pipeline
from discO.core.fitter import PotentialFitter
from discO.core.measurement import MeasurementErrorSampler
from discO.core.sample import PotentialSampler
from discO.core.wrapper import PotentialWrapper

##############################################################################
# PYTEST


def setup_module(module):
    """Setup fixtures for module."""

    class SubClassUnitTest(PotentialSampler, key="unittest"):
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
            sample.potential = object()

            return sample

    module.SubClassUnitTest = SubClassUnitTest
    # /class

    # ------------------

    class FitterSubClass(PotentialFitter, key="unittest"):
        def __call__(self, c, **kwargs):
            c.represent_as(coord.CartesianRepresentation)
            return PotentialWrapper(object(), frame=None)

        # /def

    module.FitterSubClass = FitterSubClass
    # /class


# /def

# -------------------------------------------------------------------


def teardown_module(module):
    """Teardown fixtures for module."""
    module.SubClassUnitTest._registry.pop("unittest", None)
    module.FitterSubClass._registry.pop("unittest", None)


# /def


# -------------------------------------------------------------------


class TestSampler(PotentialSampler):
    """Dunder Sampler."""

    def __call__(
        self, n, *, frame=None, representation_type=None, random=None, **kwargs
    ):
        # Get preferred frames
        frame = self._infer_frame(frame)
        representation_type = self._infer_representation(representation_type)

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
            frame.realize_frame(rep, representation_type=representation_type),
            copy=False,
        )
        sample.mass = np.ones(n)
        sample.potential = object()

        return sample

    # /def


# /class


##############################################################################
# TESTS
##############################################################################


class Test_Pipeline(object):
    """Test Pipeline."""

    def setup_class(cls):
        """Set up fixtures for testing."""
        cls.mass = 1e12 * u.solMass
        cls.r0 = 10 * u.kpc  # scale factor

        # sampling a potential that lives in a galactocentric frame
        # and enforcing a Cartesian representation
        cls.sampler = PotentialSampler(
            PotentialWrapper(object(), frame="galactocentric"),
            key="unittest",
            frame="galactocentric",
            representation_type="cartesian",
            total_mass=10*u.solMass, 
        )
        # but when we measure, it's 1% errors in icrs, Spherical
        cls.measurer = MeasurementErrorSampler(
            c_err=1 * u.percent,
            method=("rvs", "Gaussian"),
            frame="icrs",
            representation_type="spherical",
        )
        # fitting is done with an SCF potential
        cls.fitter = PotentialFitter(
            object(),
            key="unittest",
            frame=cls.sampler.frame,
            representation_type=cls.sampler.representation_type,
        )
        # the residual function
        cls.residualer = None
        # the statistic function
        cls.statistic = None

        cls.inst = pipeline.Pipeline(
            sampler=cls.sampler,
            measurer=cls.measurer,
            fitter=cls.fitter,
            residualer=cls.residualer,
            statistic=cls.statistic,
        )

    # /def

    #######################################################
    # Method tests

    def test___init__(self):
        """Test method ``__init__``."""
        # -------------------
        # basic, same as inst

        pipe = pipeline.Pipeline(
            sampler=self.sampler,
            measurer=self.measurer,
            fitter=self.fitter,
            residualer=self.residualer,
            statistic=self.statistic,
        )

        assert isinstance(pipe, pipeline.Pipeline)
        assert isinstance(pipe._sampler, PotentialSampler)
        assert isinstance(pipe._measurer, MeasurementErrorSampler)
        assert isinstance(pipe._fitter, PotentialFitter)
        assert isinstance(pipe._residualer, object)
        assert isinstance(pipe._statisticer, object)
        assert pipe._result is None

        # -------------------
        # CAN set `fitter` without `measurer`

        pipe = pipeline.Pipeline(
            sampler=self.sampler,
            # measurer=self.measurer,
            fitter=self.fitter,
            residualer=self.residualer,
            statistic=self.statistic,
        )

        assert isinstance(pipe, pipeline.Pipeline)
        assert isinstance(pipe._sampler, PotentialSampler)
        assert pipe._measurer is None
        assert isinstance(pipe._fitter, PotentialFitter)
        assert isinstance(pipe._residualer, object)
        assert isinstance(pipe._statisticer, object)
        assert pipe._result is None

        # -------------------
        # can't set `residualer` without `fitter`

        with pytest.raises(ValueError) as e:
            pipeline.Pipeline(
                sampler=self.sampler,
                measurer=self.measurer,
                # fitter=self.fitter,  # skipping fitter
                residualer=object(),
                statistic=self.statistic,
            )

        assert "Can't set `residualer` without `fitter`." in str(e.value)

        # -------------------
        # can't set `residualer` without `fitter`

        with pytest.raises(ValueError) as e:
            pipeline.Pipeline(
                sampler=self.sampler,
                measurer=self.measurer,
                fitter=self.fitter,
                # residualer=self.residualer,  # skipping residualer
                statistic=object(),
            )

        assert "Can't set `statistic` without `residualer`" in str(e.value)

    # /def

    def test_sampler(self):
        """Test property ``sampler``."""
        assert self.inst.sampler is self.inst._sampler

    # /def

    def test_potential(self):
        """Test property ``potential``."""
        assert self.inst.potential is self.inst.sampler.potential

    # /def

    def test_potential_frame(self):
        """Test property ``potential_frame``."""
        assert self.inst.potential_frame is self.inst.sampler.frame

    # /def

    def test_potential_representation_type(self):
        """Test property ``potential_representation_type``."""
        assert (
            self.inst.potential_representation_type
            is self.inst.sampler.representation_type
        )

    # /def

    def test_measurer(self):
        """Test property ``measurer``."""
        assert self.inst.measurer is self.inst._measurer

    # /def

    def test_observer_frame(self):
        """Test property ``observer_frame``."""
        assert self.inst.observer_frame is self.inst.measurer.frame

    # /def

    def test_observer_representation_type(self):
        """Test property ``observer_representation_type``."""
        assert (
            self.inst.observer_representation_type
            is self.inst.measurer.representation_type
        )

    # /def

    def test_fitter(self):
        """Test property ``fitter``."""
        assert self.inst.fitter is self.inst._fitter

    # /def

    def test_residualer(self):
        """Test property ``residualer``."""
        assert self.inst.residualer is self.inst._residualer

    # /def

    def test_statisticer(self):
        """Test property ``statisticer``."""
        assert self.inst.statisticer is self.inst._statisticer

    # /def

    # -----------------------------------------------------

    @pytest.mark.skip("TODO")
    def test___call__(self):
        """Test method ``__call__``.

        Even though ``__call__`` just runs ``run`` with ``niter=1``,
        it's still worth running through all the tests.

        Since 3rd party packages provide the backend for the sampling
        and fitting, we use dunder methods here and implement
        pipeline tests in the plugins.

        .. todo::

            There are so many variables, this will need some pytest
            parametrize with itertools methods

        """
        # make pipeline
        # sampler = TestSampler(object())
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test_run(self):
        """Test method ``run``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test___or__(self):
        """Test method ``__or__``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test___ior__(self):
        """Test method ``__ior__``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test___repr__(self):
        """Test method ``__repr__``."""
        assert False

        s = self.inst.__repr__()

        assert "Pipeline:" in s
        assert "sampler:" in s
        assert "measurer:" in s
        assert "fitter:" in s
        assert "residual:" in s
        assert "statistic:" in s

    # /def

    #######################################################
    # Usage tests


# /class


#####################################################################


class Test_PipelineResult(object):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.pipe = pipeline.Pipeline(object())
        cls.inst = pipeline.PipelineResult(
            cls.pipe,
            samples=1,
            measured="2",
            fit=[3, 3.0],
            residual=object(),
            statistic=NotImplemented,
        )

    # /def

    #######################################################
    # Method tests

    def test___init__(self):
        """Test method ``__init__``."""
        pipe = pipeline.Pipeline(object())

        # ------------------
        # inits, defaults

        pr = pipeline.PipelineResult(pipe)

        assert isinstance(pr, pipeline.PipelineResult)
        assert pr._parent_ref() is pipe
        assert pr._samples is None
        assert pr._measured is None
        assert pr._fit is None
        assert pr._residual is None
        assert pr._statistic is None

        # ------------------
        # inits, values

        assert isinstance(self.inst, pipeline.PipelineResult)

    # /def

    def test__parent(self):
        """Test method ``_parent``."""
        assert self.inst._parent_ref() is self.pipe

    # /def

    def test_samples(self):
        """Test method ``samples``."""
        assert self.inst.samples is self.inst._samples
        assert self.inst.samples == 1

    # /def

    def test_potential_frame(self):
        """Test method ``potential_frame``."""
        with pytest.raises(AttributeError):
            self.inst.potential_frame

    # /def

    def test_potential_representation_type(self):
        """Test method ``potential_representation_type``."""
        with pytest.raises(AttributeError):
            self.inst.potential_representation_type

    # /def

    def test_measured(self):
        """Test method ``measured``."""
        assert self.inst.measured is self.inst._measured
        assert self.inst.measured == "2"

    # /def

    def test_observation_frame(self):
        """Test method ``observation_frame``."""
        with pytest.raises(AttributeError):
            self.inst.observation_frame

    # /def

    def test_observation_representation_type(self):
        """Test method ``observation_representation_type``."""
        with pytest.raises(AttributeError):
            self.inst.potential_representation_type

    # /def

    def test_fit(self):
        """Test method ``fit``."""
        assert self.inst.fit is self.inst._fit
        assert all(np.equal(self.inst._fit, [3, 3.0]))

    # /def

    def test_residual(self):
        """Test method ``residual``."""
        assert self.inst.residual is self.inst._residual
        assert isinstance(self.inst.residual, object)

    # /def

    def test_statistic(self):
        """Test method ``statistic``."""
        assert self.inst.statistic is self.inst._statistic
        assert self.inst._statistic is NotImplemented

    # /def

    #######################################################
    # Usage tests


# /class


##############################################################################
# END
