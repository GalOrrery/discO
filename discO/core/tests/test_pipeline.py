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
        def __call__(self, n, *, random=None, **kwargs):
            representation_type = self.representation_type  # can be None
            if random is None:
                random = np.random
            elif isinstance(random, int):
                random = np.random.RandomState(random)

            # return
            rep = coord.SphericalRepresentation(
                lon=random.uniform(size=n) * u.deg,
                lat=2 * random.uniform(size=n) * u.deg,
                distance=10 * u.kpc,
            )
            sample = coord.SkyCoord(
                self.frame.realize_frame(
                    rep,
                    representation_type=representation_type or rep.__class__,
                ),
                copy=False,
            )
            sample.cache["mass"] = np.ones(n)
            sample.cache["potential"] = object()

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


class MockSampler(PotentialSampler):
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
        sample.cache["mass"] = np.ones(n)
        sample.cache["potential"] = object()

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
            total_mass=10 * u.solMass,
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

    def test___init__cant_set_Y_without_X(self):
        # -------------------
        # can't set `residualer` without `fitter`

        with pytest.raises(ValueError, match="`residualer` without `fitter`."):
            pipeline.Pipeline(
                sampler=self.sampler,
                measurer=self.measurer,
                # fitter=self.fitter,  # skipping fitter
                residualer=object(),
                statistic=self.statistic,
            )

        # -------------------
        # can't set `statistic` without `residualer`

        with pytest.raises(
            ValueError,
            match="`statistic` without `residualer`",
        ):
            pipeline.Pipeline(
                sampler=self.sampler,
                measurer=self.measurer,
                fitter=self.fitter,
                # residualer=self.residualer,  # skipping residualer
                statistic=object(),
            )

    def test___init__frame_mismatch(self):

        fitter = PotentialFitter(
            object(),
            key="unittest",
            frame="icrs",
            representation_type=self.sampler.representation_type,
        )

        with pytest.raises(ValueError, match="must have the same frame."):
            pipeline.Pipeline(
                sampler=self.sampler,
                measurer=self.measurer,
                fitter=fitter,
            )

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
        res = self.inst(10)

        assert isinstance(res, np.record)

        assert isinstance(res.sample, coord.SkyCoord)
        assert isinstance(res.sample.frame, coord.Galactocentric)
        assert isinstance(res.sample.data, coord.SphericalRepresentation)

        assert isinstance(res.measured, coord.SkyCoord)
        assert isinstance(res.measured.frame, coord.ICRS)
        assert isinstance(res.measured.data, coord.SphericalRepresentation)

    # /def

    def test___call__with_sample(self):
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
        sample = self.inst.sampler(10)
        res = self.inst(sample)

        assert isinstance(res, np.record)

        assert isinstance(res.sample, coord.SkyCoord)
        assert isinstance(res.sample.frame, coord.Galactocentric)
        assert isinstance(res.sample.data, coord.SphericalRepresentation)

        assert isinstance(res.measured, coord.SkyCoord)
        assert isinstance(res.measured.frame, coord.ICRS)
        assert isinstance(res.measured.data, coord.SphericalRepresentation)

    # /def

    def test___call__error(self):
        """Test method ``__call__`` with bad input"""
        with pytest.raises(TypeError):
            self.inst(TypeError)

    # /def

    def test_run(self):
        """Test method ``run``."""
        res = self.inst.run(10, 2, batch=True)

        assert isinstance(res, pipeline.PipelineResult)
        assert isinstance(res[0], np.record)
        assert isinstance(res[1], np.record)

        assert len(res.sample) == 2
        assert len(res.sample[0]) == 10
        assert len(res.measured) == 2
        assert len(res.measured[1]) == 10

    # /def

    def test___repr__(self):
        """Test method ``__repr__``."""
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
            sample=1,
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
        assert pr["sample"][0] is None
        assert pr["measured"][0] is None
        assert pr["fit"][0] is None
        assert pr["residual"][0] is None
        assert pr["statistic"][0] is None

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
        # assert self.inst["sample"] is self.inst._samples
        assert self.inst["sample"] == 1

    # /def

    def test_measured(self):
        """Test method ``measured``."""
        # assert self.inst.measured is self.inst._measured
        assert self.inst["measured"] == "2"

    # /def

    def test_fit(self):
        """Test method ``fit``."""
        # assert self.inst.fit is self.inst._fit
        assert all(np.equal(self.inst["fit"][0], [3, 3.0]))

    # /def

    def test_residual(self):
        """Test method ``residual``."""
        # assert self.inst.residual is self.inst._residual
        assert isinstance(self.inst["residual"], object)

    # /def

    def test_statistic(self):
        """Test method ``statistic``."""
        # assert self.inst.statistic is self.inst._statistic
        assert self.inst["statistic"][0] is NotImplemented

    # /def

    def test___repr__(self):
        """Test method ``__repr__``."""
        got = repr(self.inst)

        expected = np.recarray.__repr__(self.inst)
        expected = expected.replace("rec.array", self.inst.__class__.__name__)

        assert got == expected

    #######################################################
    # Usage tests

    def test_connection(self):
        assert self.inst._parent is self.pipe


# /class


##############################################################################
# END
