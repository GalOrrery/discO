# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Tests for :mod:`~starkman_thesis`."""

__all__ = [
    # modules
    "sim_CC_100_tests",
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
from . import test_sim_CC_100 as sim_CC_100_tests

##############################################################################
# TESTS
##############################################################################

test = TestRunner.make_test_runner_in(Path(__file__).parent.parent)

##############################################################################
# END
