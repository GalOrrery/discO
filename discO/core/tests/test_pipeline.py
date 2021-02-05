# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.core.pipeline`."""

__all__ = [
    "Test_Pipeline",
    "Test_PipelineResult",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import numpy as np
import pytest

# PROJECT-SPECIFIC
from discO.core import pipeline

##############################################################################
# TESTS
##############################################################################


class Test_Pipeline(object):
    """docstring for Test_Pipeline"""

    #######################################################
    # Method tests

    @pytest.mark.skip("TODO")
    def test___init__(self):
        """Test method ``__init__``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test___call__(self):
        """Test method ``__call__``."""
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
        assert self.inst._samples == 1

    # /def

    def test_measured(self):
        """Test method ``measured``."""
        assert self.inst._measured == "2"

    # /def

    def test_fit(self):
        """Test method ``fit``."""
        assert all(np.equal(self.inst._fit, [3, 3.0]))

    # /def

    def test_residual(self):
        """Test method ``residual``."""
        assert isinstance(self.inst._residual, object)

    # /def

    def test_statistic(self):
        """Test method ``statistic``."""
        assert self.inst._statistic is NotImplemented

    # /def

    #######################################################
    # Usage tests


# /class


##############################################################################
# END
