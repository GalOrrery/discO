# -*- coding: utf-8 -*-

"""Fit a potential to data with :mod:`~agama`."""

__all__ = [
    "AGAMAPotentialFitter",
    "AGAMAMultipolePotentialFitter",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T
import warnings
from types import MappingProxyType

# THIRD PARTY
import agama
import astropy.coordinates as coord

# PROJECT-SPECIFIC
from discO.common import CoordinateType, SkyCoordType
from discO.core.fitter import PotentialFitter

##############################################################################
# PARAMETERS

AGAMA_FITTER_REGISTRY = dict()  # package : samplers

##############################################################################
# CODE
##############################################################################


class AGAMAPotentialFitter(PotentialFitter, key="agama"):
    """Fit a set of particles"""

    #######################################################
    # On the class

    _registry = AGAMA_FITTER_REGISTRY

    #################################################################
    # On the instance

    def __new__(
        cls,
        *,
        pot_type: T.Optional[str] = None,
        return_specific_class: bool = False,
        **kwargs,
    ):
        self = super().__new__(cls, agama.Potential)

        # The class AGAMAPotentialFitter is a wrapper for anything in its
        # registry If directly instantiating a AGAMAPotentialFitter (not
        # subclass) we must also instantiate the appropriate subclass. Error
        # if can't find.
        if cls is AGAMAPotentialFitter:

            if pot_type not in cls._registry:
                raise ValueError(
                    "PotentialFitter has no registered fitter for `pot_type`: "
                    f"{pot_type}",
                )

            # from registry. Registered in __init_subclass__
            instance = cls._registry[pot_type](**kwargs)

            if return_specific_class:
                return instance

            self._instance = instance

        elif pot_type is not None:
            raise ValueError(
                "Can't specify 'pot_type' on PotentialFitter subclasses.",
            )

        elif return_specific_class is not False:
            warnings.warn("Ignoring argument `return_specific_class`")

        return self

    # /def

    def __init__(
        self,
        pot_type: T.Optional[str] = None,
        symmetry: str = "a",
        **kwargs,
    ):
        if pot_type is None:
            raise ValueError("must specify a `pot_type`")

        if self.__class__ is AGAMAPotentialFitter:
            self._kwargs = MappingProxyType(self._instance._kwargs)
        else:
            self._kwargs = {
                "type": pot_type,
                "symmetry": symmetry,
                **kwargs,
            }

    # /defs

    #######################################################
    # Fitting

    def __call__(self, c: CoordinateType) -> SkyCoordType:
        """Fit Potential given particles."""

        position = c.represent_as(coord.CartesianRepresentation).xyz.T
        # TODO! velocities
        mass = c.mass  # TODO! what if don't have? have as parameter?

        particles = (position, mass)

        return self._fitter(particles=particles, **self._kwargs)

    # /def


# /class


#####################################################################


class AGAMAMultipolePotentialFitter(AGAMAPotentialFitter, key="multipole"):
    """Fit a set of particles with a Multipole expansion."""

    def __init__(
        self,
        symmetry: str = "a",
        gridsizeR: int = 20,
        lmax: int = 2,
        **kwargs,
    ):
        kwargs.pop("pot_type", None)  # clear from kwargs
        super().__init__(
            pot_type="Multipole",
            symmetry=symmetry,
            gridsizeR=gridsizeR,
            lmax=lmax,
            **kwargs,
        )

    # /def


# /class


##############################################################################
# END
