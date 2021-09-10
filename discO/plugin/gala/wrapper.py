# -*- coding: utf-8 -*-
# see LICENSE.rst

""":class:`~gala.potential.PotentialBase` wrapper."""

__all__ = [
    "GalaPotentialWrapper",
]


##############################################################################
# IMPORTS

# STDLIB
import typing as T

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np

# LOCAL
import discO.type_hints as TH
from .type_hints import PotentialType
from discO.core.wrapper import PotentialWrapper, PotentialWrapperMeta
from discO.utils import resolve_representationlike, vectorfield

##############################################################################
# PARAMETERS

_KMS2 = u.km / u.s ** 2

##############################################################################
# CODE
##############################################################################


class GalaPotentialMeta(PotentialWrapperMeta):
    """Metaclass for wrapping :mod:`~gala` potentials."""

    def total_mass(self, potential: PotentialType) -> TH.QuantityType:
        """Evaluate the total mass.

        Parameters
        ----------
        potential : object
            The potential.

        Returns
        -------
        mass : u.Quantity

        """
        q = [np.inf, np.inf, np.inf] * u.kpc  # TODO! upstream fix
        return potential.mass_enclosed(q, t=0.0)

    # /def

    # -----------------------------------------------------

    def density(
        self,
        potential: PotentialType,
        points: TH.PositionType,
        *,
        frame: TH.OptFrameLikeType = None,
        representation_type: TH.OptRepresentationLikeType = None,
        **kwargs,
    ) -> T.Tuple[TH.FrameType, TH.QuantityType]:
        """Evaluate the density in the potential-density pair.

        Parameters
        ----------
        potential : `~gala.potential.PotentialBase` subclass instance
            The potential.
        points : coord-array or |Representation| or None (optional)
            The points at which to evaluate the density.
        frame : |CoordinateFrame| or None (optional, keyword-only)
            The frame of the potential. Potentials do not have an intrinsic
            reference frame, but if one is assigned, then anything needs to be
            converted to that frame.
        representation_type : |Representation| or None (optional, keyword-only)
            The representation type in which to return data.
        **kwargs
            Arguments into the potential.

        Return
        ------
        points : |CoordinateFrame|
            The points, in their original frame
        values : |Quantity|
            The density at `points`.

        """
        p, _ = self._convert_to_frame(points, frame, representation_type)
        r = p.represent_as(coord.CartesianRepresentation)
        values = potential.density(r.xyz, **kwargs)

        return p, values

    # /def

    # -----------------------------------------------------

    def potential(
        self,
        potential: PotentialType,
        points: TH.PositionType,
        *,
        frame: TH.OptFrameLikeType = None,
        representation_type: TH.OptRepresentationLikeType = None,
        **kwargs,
    ) -> T.Tuple[TH.FrameType, TH.QuantityType]:
        """Evaluate the potential.

        Parameters
        ----------
        potential : `~gala.potential.PotentialBase` subclass instance
            The potential.
        points : coord-array or |Representation| or None (optional)
            The points at which to evaluate the potential.
        frame : |CoordinateFrame| or None (optional, keyword-only)
            The frame of the potential. Potentials do not have an intrinsic
            reference frame, but if one is assigned, then anything needs to be
            converted to that frame.
        representation_type : |Representation| or None (optional, keyword-only)
            The representation type in which to return data.
        **kwargs
            Arguments into the potential.

        Return
        ------
        points : |CoordinateFrame|
            The points, in their original frame
        values : |Quantity|
            The potential at `points`.

        """
        p, _ = self._convert_to_frame(points, frame, representation_type)
        r = p.represent_as(coord.CartesianRepresentation)
        values = potential(r.xyz, **kwargs)

        return p, values

    # /def

    # -----------------------------------------------------

    def specific_force(
        self,
        potential: PotentialType,
        points: TH.PositionType,
        *,
        frame: TH.OptFrameLikeType = None,
        representation_type: TH.OptRepresentationLikeType = None,
        **kwargs,
    ) -> vectorfield.BaseVectorField:
        """Evaluate the specific force.

        Parameters
        ----------
        potential : `~gala.potential.PotentialBase` subclass instance
            The potential.
        points : coord-array or |Representation| or None (optional)
            The points at which to evaluate the potential.
        frame : |CoordinateFrame| or None (optional, keyword-only)
            The frame of the potential. Potentials do not have an intrinsic
            reference frame, but if one is assigned, then anything needs to be
            converted to that frame.
        representation_type : |Representation| or None (optional, keyword-only)
            The representation type in which to return data.
        **kwargs
            Arguments into the potential.

        Returns
        -------
        `~discO.utils.vectorfield.BaseVectorField` subclass instance
            Type set by `representation_type`

        """
        p, _ = self._convert_to_frame(points, frame, representation_type)
        r = p.represent_as(coord.CartesianRepresentation)

        a = potential.acceleration(r.xyz, **kwargs)

        vf = vectorfield.CartesianVectorField(
            points=r,
            vf_x=a[0],
            vf_y=a[1],
            vf_z=a[2],
            frame=frame,
        )

        if representation_type is not None:
            vf = vf.represent_as(
                resolve_representationlike(representation_type),
            )

        return vf

    # /def

    acceleration = specific_force


# /class


class GalaPotentialWrapper(
    PotentialWrapper,
    key="gala",
    metaclass=GalaPotentialMeta,
):
    """Wrap :mod:`~gala` :class:`~gala.Potential` objects."""


# /class

##############################################################################
# END
