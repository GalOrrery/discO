# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.config`."""

__all__ = [
    "Test_conf",
]


##############################################################################
# IMPORTS

# PROJECT-SPECIFIC
from discO.config import conf
from discO.tests.helper import ConfigNamespacTests

##############################################################################
# TESTS
##############################################################################


class Test_conf(ConfigNamespacTests, conf=conf):
    pass


# /class

# -------------------------------------------------------------------


##############################################################################
# END
