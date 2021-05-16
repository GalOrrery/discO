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
from types import MappingProxyType

# THIRD PARTY
import agama
import astropy.coordinates as coord

# PROJECT-SPECIFIC
import discO.type_hints as TH
from .wrapper import AGAMAPotentialWrapper
from discO.core.fitter import PotentialFitter
from discO.utils.coordinates import (
    resolve_framelike,
    resolve_representationlike,
)

##############################################################################
# PARAMETERS

AGAMA_FITTER_REGISTRY: T.Dict[str, object] = dict()  # package : samplers

##############################################################################
# CODE
##############################################################################


class AGAMAPotentialFitter(PotentialFitter, key="agama"):
    """Fit a set of particles with an AGAMA potential.

    Parameters
    ----------
    potential_cls : str or None

    frame: frame-like or None (optional, keyword-only)
       The frame of the observational errors, ie the frame in which
        the error function should be applied along each dimension.
    representation_type: |Representation| or None (optional, keyword-only)
        The coordinate representation in which to resample along each
        dimension.


    """

    #######################################################
    # On the class

    _registry = AGAMA_FITTER_REGISTRY

    #################################################################
    # On the instance

    def __new__(
        cls,
        *,
        potential_cls: T.Optional[str] = None,
        **kwargs,
    ):
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
            kls = cls._registry[potential_cls]
            return kls.__new__(kls, potential_cls=None, **kwargs)

        elif potential_cls is not None:
            raise ValueError(
                f"Can't specify 'potential_cls' on {cls.__name__}.",
            )

        return super().__new__(cls, agama.Potential, **kwargs)

    # /def

    def __init__(
        self,
        *,
        potential_cls: str,
        frame: TH.OptFrameLikeType = None,
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
        self,
        sample: TH.CoordinateType,
        mass: T.Optional[TH.QuantityType] = None,
        *,
        frame: TH.OptFrameLikeType = None,
        representation_type: TH.OptRepresentationLikeType = None,
        **kwargs,
    ) -> AGAMAPotentialWrapper:
        """Fit Potential given particles.

        Parameters
        ----------
        sample : coord-like

        Returns
        -------
        :class:`~astropy.coordinates.SkyCoord`

        Other Parameters
        ----------------
        frame: frame-like or None (optional, keyword-only)
            The frame of the fit potential.

            .. warning::

                Care should be taken that this matches the frame of the
                sampling potential.

        representation_type: |Representation| or None (optional, keyword-only)
            The coordinate representation.

        """
        # --------------
        # frame and representation
        # None -> default

        frame = (
            resolve_framelike(self.frame)  # incase self.frame is None or ...
            if frame is None
            else resolve_framelike(frame)
        )
        representation_type = (
            resolve_representationlike(
                self.representation_type,
                error_if_not_type=False,
            )
            if representation_type is None
            else resolve_representationlike(representation_type)
        )

        # --------------
        if mass is None:
            mass = sample.cache.get("mass")

        sample = sample.transform_to(frame)  # FIXME!
        position = sample.represent_as(coord.CartesianRepresentation).xyz.T
        # TODO! velocities

        particles = (position, mass)

        # kwargs
        kw = dict(self.potential_kwargs.items())  # deepcopy MappingProxyType
        kw.update(kwargs)

        potential = self.potential_cls(particles=particles, **kw)

        return AGAMAPotentialWrapper(
            potential,
            frame=frame,
            representation_type=representation_type,
        )

    # /def


# /class


#####################################################################


class AGAMAMultipolePotentialFitter(AGAMAPotentialFitter, key="multipole"):
    """Fit a set of particles with a Multipole expansion.

    Parameters
    ----------
    frame: frame-like or None (optional, keyword-only)
       The frame of the observational errors, ie the frame in which
        the error function should be applied along each dimension.
    representation_type: |Representation| or None (optional, keyword-only)
        The coordinate representation in which to resample along each
        dimension.

    symmetry : str (optional)
        The symmetry of the potential. See AGAMA reference.
    gridsizeR : int (optional)
        See AGAMA reference.
    lmax : int (optional)
        See AGAMA reference.
    **kwargs
        Passed to :class:`~agama.Potential`

    """

    def __init__(
        self,
        *,
        frame: TH.OptFrameLikeType = None,
        representation_type: TH.OptRepresentationLikeType = None,
        symmetry: str = "a",
        gridsizeR: int = 20,
        lmax: int = 10,
        **kwargs,
    ) -> None:
        # pop what we shouldn't pass
        kwargs.pop("potential_cls", None)
        kwargs.pop("key", None)
        # initialize
        super().__init__(
            potential_cls="Multipole",
            frame=frame,
            representation_type=representation_type,
            symmetry=symmetry,
            gridsizeR=gridsizeR,
            lmax=lmax,
            **kwargs,
        )

    # /def


# /class

##############################################################################
# END
