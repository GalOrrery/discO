# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Test :mod:`~discO.plugin.galpy`."""

__all__ = [
    "fitter_tests",
    "sample_tests",
    "type_hints_tests",
    "wrapper_tests",
]


##############################################################################
# IMPORTS

# PROJECT-SPECIFIC
from . import test_fitter as fitter_tests
from . import test_sample as sample_tests
from . import test_type_hints as type_hints_tests
from . import test_wrapper as wrapper_tests

##############################################################################
# END
