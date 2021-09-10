# -*- coding: utf-8 -*-

"""`~astropy.config.ConfigNamespace` Tests."""

__all__ = [
    "ConfigNamespacTests",
]


##############################################################################
# IMPORTS

# THIRD PARTY
from astropy import config as _config

# LOCAL
from .objecttest import ObjectTest

##############################################################################
# CODE
##############################################################################


class ConfigNamespacTests(ObjectTest, obj=_config.ConfigNamespace):
    """`~astropy.config.ConfigNamespace` Testing Framework."""

    def __init_subclass__(cls, conf: object, **kwargs):
        """Initialize subclass.

        Parameters
        ----------
        conf : `~astropy.config.ConfigNamespace` instance
            The configuration that will be tested.

        """
        cls.conf = conf

    # /def

    # -------------------------------
    # Some simple sanity checks

    def test_type(self):
        assert isinstance(self.conf, _config.ConfigNamespace)

    # /def

    def test_with(self):
        """Test values for any shenanigans."""
        for name, c in self.conf.items():
            # check value
            assert getattr(self.conf, name) == c.defaultvalue

            # setting to same value
            with self.conf.set_temp(name, c.defaultvalue):
                assert getattr(self.conf, name) == c.defaultvalue

    # /def

    # -------------------------------
    # TODO more rigorous tests


# /class


# -------------------------------------------------------------------

##############################################################################
# END
