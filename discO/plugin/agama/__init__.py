# -*- coding: utf-8 -*-
# see LICENSE.rst

"""AGAMA interface.

If using :mod:`~agama`, the units must be set to

.. code-block:: python

    agama.setUnits(mass=1, length=1, velocity=1)


"""

__all__ = [
    "AGAMAPotentialWrapper",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T

# THIRD PARTY
import agama
import astropy.coordinates as coord
import astropy.units as u

# PROJECT-SPECIFIC
import discO.type_hints as TH
from . import fitter, sample
from .fitter import *  # noqa: F401, F403
from .sample import *  # noqa: F401, F403
from .type_hints import PotentialType
from discO.core.core import PotentialWrapper, PotentialWrapperMeta
from discO.utils import vectorfield

# __all__
__all__ += sample.__all__
__all__ += fitter.__all__


##############################################################################
# Parameters

agama.setUnits(mass=1, length=1, velocity=1)  # FIXME! bad

##############################################################################
# CODE
##############################################################################


class AGAMAPotentialMeta(PotentialWrapperMeta):
    """docstring for AGAMAPotentialMeta"""

    def specific_potential(
        self,
        potential: PotentialType,
        points: TH.PositionType,
        *,
        frame: T.Optional[TH.FrameType] = None,
        representation_type: T.Optional[TH.RepresentationType] = None,
        **kwargs
    ) -> T.Tuple[TH.SkyCoordType, TH.QuantityType]:
        """Evaluate the specific potential.

        Parameters
        ----------
        potential : `~agama.Potential`
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
        points: :class:`~astropy.coordinates.CoordinateFrame`
            The points
        values : :class:`~astropy.unit.Quantity`
            Array of the specific-potential value at the points.

        """
        shape = points.shape[:]  # copy the shape

        p, from_frame = self._convert_to_frame(points, frame)
        r = p.represent_as(coord.CartesianRepresentation)
        r = r.reshape(-1)  # unfortunately can't flatten in-place

        agama.setUnits(mass=1, length=1, velocity=1)  # TODO! bad
        values = potential.potential(r.xyz.T) * (u.km ** 2 / u.s ** 2)

        # reshape
        r.shape = shape
        values.shape = shape

        points = self._return_points(  # get points to right rep
            points,
            r,
            representation_type,
            from_frame,
        )

        # TODO! ScalarField to package together
        return points, values

    # /def

    def specific_force(
        self,
        potential: PotentialType,
        points: TH.PositionType,
        *,
        frame: T.Optional[TH.FrameType] = None,
        representation_type: T.Optional[TH.RepresentationType] = None,
        **kwargs
    ) -> vectorfield.BaseVectorField:
        """Evaluate the specific force.

        Parameters
        ----------
        potential : `~agama.Potential`
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
        shape = points.shape[:]  # copy the shape
        p, from_frame = self._convert_to_frame(points, frame)
        r = p.represent_as(coord.CartesianRepresentation)
        r = r.reshape(-1)  # unfortunately can't flatten in-place

        agama.setUnits(mass=1, length=1, velocity=1)  # TODO! bad
        Fx, Fy, Fz = (
            potential.force(r.xyz.T).T
            / u.kpc.to(u.km)  # adjustment in AGAMA units
            * (u.km / u.s ** 2)
        )

        # reshape
        r.shape = shape
        Fx.shape = shape
        Fy.shape = shape
        Fz.shape = shape

        # return vectorfield
        # TODO? convert back to from_frame?
        vf = vectorfield.CartesianVectorField(
            points=r,
            vf_x=Fx,
            vf_y=Fy,
            vf_z=Fz,
            frame=frame,
        )

        if representation_type is not None:
            vf = vf.represent_as(representation_type)

        return vf

    # /def

    acceleration = specific_force


# /class


class AGAMAPotentialWrapper(
    PotentialWrapper,
    key="agama",
    metaclass=AGAMAPotentialMeta,
):
    """Wrap :mod:`~agama` :class:`~agama.Potential` objects."""


# /class


##############################################################################
# END
