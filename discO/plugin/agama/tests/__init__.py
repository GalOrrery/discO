# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Test :mod:`~discO.plugin.agama`."""

__all__ = [
    "sample_tests",
    "fitter_tests",
    "wrapper_tests",
    "type_hints_tests",
]


##############################################################################
# IMPORTS

# LOCAL
from . import test_fitter as fitter_tests
from . import test_sample as sample_tests
from . import test_type_hints as type_hints_tests
from . import test_wrapper as wrapper_tests

##############################################################################
# END
