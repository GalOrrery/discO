# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.common`."""

__all__ = [
    "Test_QuantityType",
]


##############################################################################
# IMPORTS

# PROJECT-SPECIFIC
from discO import config
from discO.tests.helper import ConfigNamespacTests

##############################################################################
# TESTS
##############################################################################


class Test_Config(ConfigNamespacTests, conf=config.conf):
    pass


# /class

# -------------------------------------------------------------------


##############################################################################
# END
