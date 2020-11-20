# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Configuration."""


##############################################################################
# IMPORTS

# THIRD PARTY
from astropy import config as _config

__all__ = [
    "conf",
]

#############################################################################
# CONFIGURATIONS


class Conf(_config.ConfigNamespace):
    """Configuration parameters for `skycover.footprint`."""

    # Footprint

    default_frame = _config.ConfigItem(
        "icrs",
        description="Default Footprint Frame.",
        cfgtype="string",
    )


conf = Conf()
# /class

#############################################################################
# END
