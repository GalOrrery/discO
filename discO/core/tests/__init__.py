# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Tests for :mod:`~discO.core`."""

__all__ = [
    # modules
    "common_tests",
    "sample_tests",
    "measurement_tests",
    "fitter_tests",
    "residual_tests",
    "pipeline_tests",
    # instance
    "test",
]


##############################################################################
# IMPORTS

# BUILT-IN
from pathlib import Path

# THIRD PARTY
from astropy.tests.runner import TestRunner

# PROJECT-SPECIFIC
from . import test_common as common_tests
from . import test_fitter as fitter_tests
from . import test_measurement as measurement_tests
from . import test_pipeline as pipeline_tests
from . import test_residual as residual_tests
from . import test_sample as sample_tests

##############################################################################
# TESTS
##############################################################################

test = TestRunner.make_test_runner_in(Path(__file__).parent.parent)

##############################################################################
# END
