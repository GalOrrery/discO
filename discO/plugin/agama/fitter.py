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
import discO.type_hints as TH
from .wrapper import AGAMAPotentialWrapper
from discO.core.fitter import PotentialFitter

##############################################################################
# PARAMETERS

AGAMA_FITTER_REGISTRY: T.Dict[str, object] = dict()  # package : samplers

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
        cls, potential_cls: T.Optional[str] = None, **kwargs,
    ):
        self = super().__new__(cls, agama.Potential)

        # The class AGAMAPotentialFitter is a wrapper for anything in its
        # registry If directly instantiating a AGAMAPotentialFitter (not
        # subclass) we must also instantiate the appropriate subclass. Error
        # if can't find.
        if cls is AGAMAPotentialFitter:

            if potential_cls not in cls._registry:
                raise ValueError(
                    "PotentialFitter has no registered fitter for "
                    f"`potential_cls`: {potential_cls}",
                )

            # from registry. Registered in __init_subclass__
            return cls._registry[potential_cls]

        elif potential_cls is not None:
            raise ValueError(
                "Can't specify 'potential_cls' on PotentialFitter subclasses.",
            )

        return self

    # /def

    def __init__(
        self,
        potential_cls: T.Optional[str] = None,
        *,
        frame: T.Optional[TH.FrameLikeType] = None,
        representation_type: T.Optional[TH.RepresentationType] = None,
        symmetry: str = "a",
        **kwargs,
    ) -> None:
        super().__init__(
            agama.Potential,
            frame=frame,
            representation_type=representation_type,
        )

        if potential_cls is None:
            raise ValueError("must specify a `potential_cls`")

        if self.__class__ is AGAMAPotentialFitter:
            self._kwargs = MappingProxyType(self._instance._kwargs)
        else:
            self._kwargs = {
                "type": potential_cls,
                "symmetry": symmetry,
                **kwargs,
            }

    # /def

    #######################################################
    # Fitting

    def __call__(
        self, sample: TH.CoordinateType, mass: T.Optional[TH.QuantityType] = None, **kwargs
    ) -> AGAMAPotentialWrapper:
        """Fit Potential given particles.

        Parameters
        ----------
        sample : coord-like

        Returns
        -------
        :class:`~astropy.coordinates.SkyCoord`

        """

        position = sample.represent_as(coord.CartesianRepresentation).xyz.T
        # TODO! velocities
        if mass is None:
            mass = sample.mass  # TODO! what if don't have? have as parameter?

        particles = (position, mass)

        # kwargs
        kw = dict(self.potential_kwargs.items())  # deepcopy MappingProxyType
        kw.update(kwargs)

        potential = self._fitter(particles=particles, **kw)

        return AGAMAPotentialWrapper(
            potential,
            frame=self.frame,
            representation_type=self.representation_type,
        )

    # /def


# /class


#####################################################################


class AGAMAMultipolePotentialFitter(AGAMAPotentialFitter, key="multipole"):
    """Fit a set of particles with a Multipole expansion.

    Parameters
    ----------
    symmetry : str
    gridsizeR : int (optional)
    lmax : int (optional)
    **kwargs
        Passed to :class:`~agama.Potential`

    """

    def __init__(
        self,
        *,
        frame: T.Optional[TH.FrameLikeType] = None,
        representation_type: T.Optional[TH.RepresentationType] = None,
        symmetry: str = "a",
        gridsizeR: int = 20,
        lmax: int = 2,
        **kwargs,
    ) -> None:
        kwargs.pop("pot_type", None)  # clear from kwargs
        super().__init__(
            frame=frame,
            representation_type=representation_type,
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
