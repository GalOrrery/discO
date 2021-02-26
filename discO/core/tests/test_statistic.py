# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.core.statistic`."""

__all__ = [
    "test_rms",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np

# PROJECT-SPECIFIC
from discO.core import statistic
from discO.utils.vectorfield import CartesianVectorField

##############################################################################
# TESTS
##############################################################################


def test_rms():
    """:func:`~discO.core.statistic.rms`"""
    x = np.linspace(0, 1, num=49) * u.kpc
    vf_x = np.ones(len(x)) * u.km / u.s

    points = coord.CartesianRepresentation((x, x, x))
    vf = CartesianVectorField(points, vf_x=vf_x, vf_y=vf_x, vf_z=vf_x)

    got = statistic.rms(vf)

    assert got == np.sqrt(3) * u.km / u.s

# /def


##############################################################################
# END
