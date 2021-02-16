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
    """Configuration parameters for :mod:`~discO`."""

    default_frame = _config.ConfigItem(
        "icrs",
        description="Default Footprint Frame.",
        cfgtype="string",
    )

    default_representation_type = _config.ConfigItem(
        "cartesian",
        description="Default Representation Type.",
        cfgtype="string",
    )


conf = Conf()
# /class

#############################################################################
# END
