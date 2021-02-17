# -*- coding: utf-8 -*-
# see LICENSE.rst

""":class:`~galpy.potential.Potential` wrapper."""

__all__ = [
    "GalpyPotentialWrapper",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import galpy.potential as gpot
import numpy as np

# PROJECT-SPECIFIC
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


class GalpyPotentialMeta(PotentialWrapperMeta):
    """Metaclass for wrapping :mod:`~galpy` potentials."""

    def total_mass(self, potential: PotentialType) -> TH.QuantityType:
        """Evaluate the total mass.

        Parameters
        ----------
        potential : object
            The potential.

        Raises
        ------
        NotImplementedError

        """
        return potential.mass(np.inf)

    # /def

    # -----------------------------------------------------

    def specific_potential(
        self,
        potential: PotentialType,
        points: TH.PositionType,
        *,
        frame: TH.OptFrameLikeType = None,
        representation_type: TH.OptRepresentationLikeType = None,
        **kwargs,
    ) -> T.Tuple[TH.FrameType, TH.QuantityType]:
        """Evaluate the specific potential.

        Parameters
        ----------
        potential : `~galpy.potential.Potential`
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
            The specific potential at `points`.

        """
        print(
            f"points: {points[:4]}, {points.__class__}",
            f"frame: {frame}",
            f"representation_type: {representation_type}",
            sep="\n",
        )
        p, _ = self._convert_to_frame(points, frame, representation_type)
        r = p.represent_as(coord.CylindricalRepresentation)

        # TODO! be careful about phi definition!
        values = potential(r.rho, r.z, phi=r.phi, **kwargs)

        # TODO! ScalarField to package together
        p, _ = self._convert_to_frame(p, frame, representation_type)
        return p, values

    # /def

    # -----------------------------------------------------

    def specific_force(
        self,
        potential,
        points: TH.PositionType,
        *,
        frame: TH.OptFrameLikeType = None,
        representation_type: TH.OptRepresentationLikeType = None,
        **kwargs,
    ) -> vectorfield.BaseVectorField:
        """Evaluate the specific force.

        Parameters
        ----------
        potential : `~galpy.potential.Potential`
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
        r = p.represent_as(coord.CylindricalRepresentation)

        # the specific force = acceleration
        Frho = potential.Rforce(r.rho, r.z, phi=r.phi, **kwargs).to(_KMS2)
        Fphi = (
            potential.phiforce(r.rho, r.z, phi=r.phi, **kwargs) / r.rho
        ).to(_KMS2)
        Fz = potential.zforce(r.rho, r.z, phi=r.phi, **kwargs).to(_KMS2)

        vf = vectorfield.CylindricalVectorField(
            points=r,
            vf_rho=Frho,
            vf_phi=Fphi,
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

    def coefficients(self, potential) -> T.Optional[T.Dict[str, T.Any]]:
        """Coefficients of the potential.

        Parameters
        ----------
        potential : :class:`~galpy.potential.Potential`
            The potential.

        Returns
        -------
        None or dict
            None if there aren't coefficients, a dict of the coefficients
            if there are.

        """
        coeffs = None  # start with None, then figure out.

        if isinstance(potential, gpot.SCFPotential):
            coeffs = dict(
                type="SCF",
                Acos=potential._Acos,
                Asin=potential._Asin,
            )
        elif isinstance(potential, gpot.DiskSCFPotential):
            coeffs = dict(
                type="diskSCF",
                Acos=potential._scf._Acos,
                Asin=potential._scf._Asin,
            )

        return coeffs

    # /def


# /class


class GalpyPotentialWrapper(
    PotentialWrapper,
    key="galpy",
    metaclass=GalpyPotentialMeta,
):
    """Wrap :mod:`~galpy` :class:`~galpy.Potential` objects.

    .. todo::

        Be careful about galpy coordinate conventions!
        phi is different?

    """


# /class

##############################################################################
# END
