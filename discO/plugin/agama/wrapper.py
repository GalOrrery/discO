# -*- coding: utf-8 -*-
# see LICENSE.rst

""":class:`~agama.Potential` wrappers."""

__all__ = [
    "AGAMAPotentialWrapper",
]


##############################################################################
# IMPORTS

# STDLIB
import tempfile
import typing as T

# THIRD PARTY
import agama
import astropy.coordinates as coord
import astropy.units as u
import numpy as np

# LOCAL
import discO.type_hints as TH
from .type_hints import PotentialType
from discO.core.wrapper import PotentialWrapper, PotentialWrapperMeta
from discO.utils import resolve_representationlike, vectorfield

##############################################################################
# Parameters

agama.setUnits(mass=1, length=1, velocity=1)  # FIXME! bad

##############################################################################
# CODE
##############################################################################


class AGAMAPotentialMeta(PotentialWrapperMeta):
    """Metaclass for wrapping :mod:`~agama` potentials."""

    def total_mass(self, potential: TH.PositionType) -> TH.QuantityType:
        """Evaluate the total mass.

        Parameters
        ----------
        potential : object
            The potential.

        Raises
        ------
        NotImplementedError

        """
        return potential.totalMass() * u.solMass  # FIXME! b/c agama units

    # /def

    # -----------------------------------------------------

    def density(
        self,
        potential: PotentialType,
        points: TH.PositionType,
        *,
        frame: TH.OptFrameLikeType = None,
        representation_type: TH.OptRepresentationLikeType = None,
        **kwargs
    ) -> T.Tuple[TH.SkyCoordType, TH.QuantityType]:
        """Evaluate the density.

        Parameters
        ----------
        potential : `~agama.Potential`
            The potential.
        points : coord-array or |Representation| or None (optional)
            The points at which to evaluate the density.
        frame : |CoordinateFrame| or None (optional, keyword-only)
            The frame of the density. Potentials do not have an intrinsic
            reference frame, but if one is assigned, then anything needs to be
            converted to that frame.
        representation_type : |Representation| or None (optional, keyword-only)
            The representation type in which to return data.
        **kwargs
            Arguments into the density.

        Returns
        -------
        points: :class:`~astropy.coordinates.CoordinateFrame`
            The points.
        values : :class:`~astropy.unit.Quantity`
            Array of the specific-potential value at the points.

        """
        shape = points.shape[:]  # copy the shape

        p, _ = self._convert_to_frame(points, frame, representation_type)
        r = p.represent_as(coord.CartesianRepresentation)
        r = r.reshape(-1)  # unfortunately can't flatten in-place

        agama.setUnits(mass=1, length=1, velocity=1)  # TODO! bad
        values = potential.density(r.xyz.T) * (u.solMass / u.pc ** 3)

        # reshape
        r.shape = shape
        values.shape = shape

        # TODO! ScalarField to package together
        p, _ = self._convert_to_frame(p, frame, representation_type)
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
        **kwargs
    ) -> T.Tuple[TH.SkyCoordType, TH.QuantityType]:
        """Evaluate the potential.

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

        p, _ = self._convert_to_frame(points, frame, representation_type)
        r = p.represent_as(coord.CartesianRepresentation)
        r = r.reshape(-1)  # unfortunately can't flatten in-place

        agama.setUnits(mass=1, length=1, velocity=1)  # TODO! bad
        values = potential.potential(r.xyz.T) * (u.km ** 2 / u.s ** 2)

        # reshape
        r.shape = shape
        values.shape = shape

        # TODO! ScalarField to package together
        p, _ = self._convert_to_frame(p, frame, representation_type)
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
        **kwargs
    ) -> vectorfield.BaseVectorField:
        """Evaluate the specific force.

        Parameters
        ----------
        potential : :class:`~agama.Potential`
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
        p, _ = self._convert_to_frame(points, frame, representation_type)
        # AGAMA uses a flattened Cartesian representation
        r = p.represent_as(coord.CartesianRepresentation).reshape(-1)

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
            vf = vf.represent_as(
                resolve_representationlike(representation_type),
            )

        return vf

    # /def

    acceleration = specific_force

    # -----------------------------------------------------

    def coefficients(
        self,
        potential: PotentialType,
    ) -> T.Optional[T.Dict[str, T.Any]]:
        """Coefficients of the potential.

        Parameters
        ----------
        potential : :class:`~agama.Potential`
            The potential.

        Returns
        -------
        None or dict
            None if there aren't coefficients, a dict of the coefficients
            if there are.

        """
        coeffs = None

        # There's no way way to tell if a potential is an expansion or not
        # except by importing it to a text file and reading out the type.
        # We make a temporary file, and work within that.
        with tempfile.NamedTemporaryFile() as file:
            potential.export(file.name)

            ptype = file.readline()

            if b"Multipole" in ptype:

                n_radial = int(file.readline().split(b"\t")[0])
                l_max = int(file.readline().split(b"\t")[0])
                unused = int(file.readline().split(b"\t")[0])
                _ = file.readline()[:-1]  # TODO
                radius, *lminfo = file.readline().split(b"\t")
                # rest = file.readlines()  # TODO! dPhidr

                Phi = np.loadtxt(
                    file.name,
                    skiprows=6,
                    max_rows=51,
                )  # TODO! what should max_rows be!

                coeffs = dict(
                    type="Multipole",
                    info=dict(
                        n_radial=n_radial,
                        l_max=l_max,
                        unused=unused,
                        lminfo=lminfo,
                    ),
                    coeffs=Phi,
                )

            elif b"CylSpline" in ptype:
                raise NotImplementedError("TODO")

        # /with

        return coeffs

    # /def


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
