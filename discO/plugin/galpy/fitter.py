# -*- coding: utf-8 -*-

"""Fit a potential to data with :mod:`~galpy`."""

__all__ = [
    # "GalpyPotentialFitter",
    "GalpySCFPotentialFitter",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T
from types import MappingProxyType

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
from galpy.potential import SCFPotential

# PROJECT-SPECIFIC
import discO.type_hints as TH
from .wrapper import GalpyPotentialWrapper
from discO.core.fitter import PotentialFitter
from discO.extern.galpy_potentials import scf_compute_coeffs_nbody

##############################################################################
# PARAMETERS

GALPY_FITTER_REGISTRY: T.Dict[str, object] = dict()  # package : samplers

##############################################################################
# CODE
##############################################################################


class GalpyPotentialFitter(PotentialFitter, key="galpy"):
    """Fit a set of particles"""

    #######################################################
    # On the class

    _registry = GALPY_FITTER_REGISTRY

    #################################################################
    # On the instance

    def __new__(
        cls, potential_cls: T.Any, *, key: T.Optional[str] = None, **kwargs,
    ):
        self = super().__new__(cls, potential_cls, key=None, **kwargs)

        # The class GalpyPotentialFitter is a wrapper for anything in its
        # registry If directly instantiating a GalpyPotentialFitter (not
        # subclass) we must also instantiate the appropriate subclass. Error
        # if can't find.
        if cls is GalpyPotentialFitter:

            if key not in cls._registry:
                raise ValueError(
                    "PotentialFitter has no registered fitter for key: "
                    f"{key}",
                )

            # from registry. Registered in __init_subclass__
            return cls._registry[key]

        elif key is not None:
            raise ValueError(
                "Can't specify 'key' on GalpyPotentialFitter subclasses.",
            )

        return self

    # /def

    @property
    def potential_kwargs(self) -> MappingProxyType:
        """Potential kwargs."""
        return MappingProxyType(self._kwargs)

    # /def

    #######################################################
    # Fitting

    def __call__(
        self,
        c: TH.CoordinateType,
        mass: T.Optional[TH.QuantityType] = None,
        **kwargs,
    ) -> TH.SkyCoordType:
        """Fit Potential given particles.

        Parameters
        ----------
        c : coord-like

        Returns
        -------
        :class:`~astropy.coordinates.SkyCoord`

        """
        raise NotImplementedError()

    # /def


# /class


#####################################################################


class GalpySCFPotentialFitter(GalpyPotentialFitter, key="scf"):
    """Fit a set of particles with a Multipole expansion.

    Parameters
    ----------
    symmetry : str
    gridsizeR : int (optional)
    lmax : int (optional)
    **kwargs
        Passed to :class:`~galpy.potential.Potential`

    """

    def __new__(cls, **kwargs):
        return super().__new__(cls, SCFPotential, key=None, **kwargs)

    # /def

    def __init__(
        self,
        frame: T.Optional[TH.FrameLikeType] = None,
        representation_type: T.Optional[TH.RepresentationType] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            potential_cls=SCFPotential,
            frame=frame,
            representation_type=representation_type,
            **kwargs,
        )

    # /def

    #######################################################
    # Fitting

    def __call__(
        self,
        sample: TH.CoordinateType,
        mass: T.Optional[TH.QuantityType] = None,
        *,
        Nmax: int = 10,
        Lmax: int = 10,
        scale_factor: TH.QuantityType = 1 * u.one,
        **kwargs,
    ) -> TH.SkyCoordType:
        """Fit Potential given particles.

        .. todo::

            amp != mass. Do this correctly.

        Parameters
        ----------
        sample : |CoordinateFrame| or |SkyCoord|
        Nmax, Lmax : int
            > 0.
        scale_factor : scalar |Quantity|
            units of distance or dimensionless

        Returns
        -------
        :class:`~astropy.coordinates.SkyCoord`

        """
        # --------------
        # Validation

        if Nmax <= 0 or Lmax <= 0:
            raise ValueError("Nmax & Lmax must be >0.")

        if scale_factor.unit.physical_type not in ("length", "dimensionless"):
            raise u.UnitsError(
                "scale factor must have units of length or be dimensionless",
            )
        elif not scale_factor.isscalar:
            raise ValueError("scale factor must be a scalar.")

        # --------------

        position = sample.represent_as(coord.CartesianRepresentation).xyz
        if mass is None:
            mass = sample.mass

        # kwargs
        kw = dict(self.potential_kwargs.items())  # deepcopy MappingProxyType
        kw.update(kwargs)

        # a dimensionless scale factor is assigned the same units as the
        # positions, so that (r / a) does not introduce an inadvertent scaling
        # from the units.
        if scale_factor.unit == u.one:
            scale_factor = scale_factor.value * position.unit

        Acos, Asin = scf_compute_coeffs_nbody(
            position, mass, N=Nmax, L=Lmax, a=scale_factor, **kw
        )

        return GalpyPotentialWrapper(
            self._fitter(
                amp=mass.sum(), Acos=Acos, Asin=Asin, a=scale_factor
            ),
            frame=self.frame,
            representation_type=self.representation_type,
        )

    # /def


# /class

##############################################################################
# END
