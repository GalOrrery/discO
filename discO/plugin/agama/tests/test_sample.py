# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.plugin.agama.sample`."""

__all__ = [
    "Test_AGAMAPotentialSampler",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import pytest

# PROJECT-SPECIFIC
from discO.core.tests.test_sample import Test_PotentialSampler
from discO.plugin.agama import sample


##############################################################################
# TESTS
##############################################################################


class Test_AGAMAPotentialSampler(
    Test_PotentialSampler, obj=sample.AGAMAPotentialSampler
):

    # -------------------------------

    @pytest.mark.skip("TODO")
    def test_method(self):
        """Test :class:`PACKAGE.CLASS.METHOD`."""
        assert False

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
