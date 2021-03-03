# -*- coding: utf-8 -*-

"""Fit a potential to data with :mod:`~galpy`."""

__all__ = [
    "GalpyPotentialFitter",
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
from galpy.potential import SCFPotential, scf_compute_coeffs_nbody

# PROJECT-SPECIFIC
import discO.type_hints as TH
from .wrapper import GalpyPotentialWrapper
from discO.core.fitter import PotentialFitter

# from discO.extern.galpy_potentials import scf_compute_coeffs_nbody
from discO.utils.coordinates import (
    resolve_framelike,
    resolve_representationlike,
)

##############################################################################
# PARAMETERS

GALPY_FITTER_REGISTRY: T.Dict[str, object] = dict()  # package : samplers

##############################################################################
# CODE
##############################################################################


class GalpyPotentialFitter(PotentialFitter, key="galpy"):
    """Fit a set of particles with Galpy.

    Parameters
    ----------
    potential_cls : object or str or None


    Other Parameters
    ----------------
    key : str or None



    """

    #######################################################
    # On the class

    _registry = GALPY_FITTER_REGISTRY

    #################################################################
    # On the instance

    def __new__(
        cls,
        *,
        potential_cls: T.Any = None,
        key: T.Optional[str] = None,
        **kwargs,
    ):
        # The class GalpyPotentialFitter is a wrapper for anything in its
        # registry If directly instantiating a GalpyPotentialFitter (not
        # subclass) we must also instantiate the appropriate subclass. Error
        # if can't find.
        if cls is GalpyPotentialFitter:

            # potential_cls overrides key
            key = potential_cls if isinstance(potential_cls, str) else key

            if key not in cls._registry:
                raise ValueError(
                    "PotentialFitter has no registered fitter for key: "
                    f"{key}",
                )

            # from registry. Registered in __init_subclass__
            kls = cls._registry[key]
            return kls.__new__(
                kls, potential_cls=potential_cls, key=None, **kwargs
            )

        elif key is not None:
            raise ValueError(f"Can't specify 'key' on {cls.__name__}.")

        return super().__new__(
            cls, potential_cls=potential_cls, key=None, **kwargs
        )

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

        frame: frame-like or None (optional, keyword-only)
            The frame of the fit potential.

            .. warning::

                Care should be taken that this matches the frame of the
                sampling potential.

        representation_type: |Representation| or None (optional, keyword-only)
            The coordinate representation.

        Returns
        -------
        :class:`~astropy.coordinates.SkyCoord`

        """
        raise NotImplementedError("Implement in subclass.")

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
        kwargs.pop("potential_cls", None)
        kwargs.pop("key", None)
        return super().__new__(
            cls, potential_cls=SCFPotential, key=None, **kwargs
        )

    # /def

    def __init__(
        self,
        frame: TH.OptFrameLikeType = None,
        representation_type: T.Optional[TH.RepresentationType] = None,
        **kwargs,
    ) -> None:
        kwargs.pop("potential_cls", None)
        kwargs.pop("key", None)
        super().__init__(
            potential_cls=SCFPotential,
            key=None,
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
        Nmax: int = None,
        Lmax: int = None,
        scale_factor: TH.QuantityType = None,
        **kwargs,
    ) -> TH.SkyCoordType:
        """Fit Potential given particles.

        .. todo::

            - amp != mass. Do this correctly.
            - work on ``n=[]`` array, not just niters

        Parameters
        ----------
        sample : |CoordinateFrame| or |SkyCoord|
        mass : |QuantityType|
        Nmax, Lmax : int or None (optional, keyword-only)
            The number of radial (N) and angular (L) coefficients.
            Must be integers > 0.
            If None (default) tries to draw from kwargs set at class
            initialization. If None set, raises ValueError.
        scale_factor : scalar |Quantity|
            units of distance or dimensionless

        Returns
        -------
        :class:`~astropy.coordinates.SkyCoord`

        Raises
        ------
        ValueError
            If `Nmax`, `Lmax` are None and no default set at initialization.

        """
        # --------------
    
        if mass is None:
            mass = sample.mass

        # kwargs
        kw = dict(self.potential_kwargs.items())  # deepcopy MappingProxyType
        kw.update(kwargs)

        # get from defaults if not passed
        _Nmax = kw.pop("Nmax", None)  # always try to pop
        Nmax = Nmax if Nmax is not None else _Nmax
        _Lmax = kw.pop("Lmax", None)  # always try to pop
        Lmax = Lmax if Lmax is not None else _Lmax

        if Nmax is None or Lmax is None:
            raise ValueError(
                "Nmax, Lmax have no default and must be specified.",
            )

        _scale_factor = kw.pop("scale_factor", 1 * u.one)
        scale_factor = (
            scale_factor if scale_factor is not None else _scale_factor
        )

        # --------------

        representation_type = resolve_representationlike(
            self.representation_type, error_if_not_type=False,
        )

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

        sample = sample.transform_to(self.frame)
        position = sample.represent_as(coord.CartesianRepresentation).xyz

        # a dimensionless scale factor is assigned the same units as the
        # positions, so that (r / a) does not introduce an inadvertent scaling
        # from the units.
        if scale_factor.unit == u.one:
            scale_factor = scale_factor.value * position.unit

        # TODO don't do ``to_value`` when galpy supports units
        Acos, Asin = scf_compute_coeffs_nbody(
            position.to_value(position.unit),
            mass=mass.to_value(1e12 * u.solMass),
            N=Nmax,
            L=Lmax,
            a=scale_factor.to_value(position.unit),
            **kw,
        )

        return GalpyPotentialWrapper(
            self.potential_cls(
                amp=mass.sum(), Acos=Acos, Asin=Asin, a=scale_factor,
            ),
            frame=self.frame,
            representation_type=representation_type,
        )

    # /def


# /class

##############################################################################
# END
