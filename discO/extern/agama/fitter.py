# -*- coding: utf-8 -*-

"""**DOCSTRING**."""

__all__ = [
    "AGAMAPotentialFitter",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord

# PROJECT-SPECIFIC
from discO.common import CoordinateType, SkyCoordType
from discO.core.fitter import PotentialFitter

##############################################################################
# CODE
##############################################################################


class AGAMAPotentialFitter(PotentialFitter, package="agama"):
    """Fit a set of particles"""

    # FIXME! these are specific to multipole
    def __init__(
        self,
        pot_type="Multipole",
        symmetry="a",
        gridsizeR=20,
        lmax=2,
        **kwargs,
    ):
        import agama

        self._fitter = agama.Potential
        self._kwargs = {
            "type": pot_type,
            "symmetry": symmetry,
            "gridsizeR": gridsizeR,
            "lmax": lmax,
            **kwargs,
        }

    # /defs

    def __call__(self, c: CoordinateType) -> SkyCoordType:
        """Fit Potential given particles."""

        position = c.represent_as(coord.CartesianRepresentation).xyz.T
        # TODO! velocities
        mass = c.mass  # TODO! what if don't have?

        particles = (position, mass)

        return self._fitter(particles=particles, **self._kwargs)

    # /def


# /class


##############################################################################
# END
