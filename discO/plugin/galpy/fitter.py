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


# class GalpyPotentialFitter(PotentialFitter, key="galpy"):
#     """Fit a set of particles"""

#     #######################################################
#     # On the class

#     _registry = GALPY_FITTER_REGISTRY

#     #################################################################
#     # On the instance

#     def __new__(
#         cls,
#         potential_cls: T.Any,
#         *,
#         key: T.Optional[str] = None,
#         return_specific_class: bool = False,
#         **kwargs,
#     ):
#         self = super().__new__(cls, potential_cls)

#         # The class GalpyPotentialFitter is a wrapper for anything in its
#         # registry If directly instantiating a GalpyPotentialFitter (not
#         # subclass) we must also instantiate the appropriate subclass. Error
#         # if can't find.
#         if cls is GalpyPotentialFitter:

#             if key not in cls._registry:
#                 raise ValueError(
#                     "PotentialFitter has no registered fitter for key: "
#                     f"{key}",
#                 )

#             # from registry. Registered in __init_subclass__
#             # some subclasses accept the potential_cls as an argument,
#             # others do not.
#             subcls = cls._registry[key]
#             sig = inspect.signature(subcls)
#             ba = sig.bind_partial(potential_cls=potential_cls, **kwargs)
#             ba.apply_defaults()

#             instance = cls._registry[key](*ba.args, **ba.kwargs)

#             if return_specific_class:
#                 return instance

#             self._instance = instance

#         elif key is not None:
#             raise ValueError(
#                 "Can't specify 'key' on GalpyPotentialFitter subclasses.",
#             )

#         elif return_specific_class is not False:
#             warnings.warn("Ignoring argument `return_specific_class`")

#         return self

#     # /def

#     #######################################################
#     # Fitting

#     # def __call__(self, c: TH.CoordinateType, **kwargs) -> TH.SkyCoordType:
#     #     """Fit Potential given particles.

#     #     Parameters
#     #     ----------
#     #     c : coord-like

#     #     Returns
#     #     -------
#     #     :class:`~astropy.coordinates.SkyCoord`

#     #     """

#     #     position = c.represent_as(coord.CartesianRepresentation).xyz.T
#     #     mass = c.mass  # TODO! what if don't have? have as parameter?

#     #     particles = (position, mass)

#     #     # kwargs
#     #     kw = dict(self.potential_kwargs.items())  # deepcopy MappingProxyType
#     #     kw.update(kwargs)

#     #     return self._fitter(particles=particles, **kw)

#     # # /def


# # /class


#####################################################################


class GalpySCFPotentialFitter(PotentialFitter, key="scf"):
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
        return super().__new__(
            cls, SCFPotential, key=None, return_specific_class=False, **kwargs
        )

    # /def

    def __init__(
        self, frame: T.Optional[TH.FrameLikeType] = None, **kwargs
    ) -> None:
        super().__init__(
            potential_cls=SCFPotential,
            frame=frame,
            **kwargs,
        )

    # /def

    #######################################################
    # Fitting

    def __call__(
        self,
        c: TH.CoordinateType,
        mass: T.Optional[TH.QuantityType] = None,
        Nmax: int = 10,
        Lmax: int = 10,
        scale_factor: TH.QuantityType = 1 * u.one,
        **kwargs,
    ) -> TH.SkyCoordType:
        """Fit Potential given particles.

        Parameters
        ----------
        c : |CoordinateFrame| or |SkyCoord|
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

        position = c.represent_as(coord.CartesianRepresentation).xyz
        if mass is None:
            mass = c.mass

        # a dimensionless scale factor is assigned the same units as the
        # positions, so that (r / a) does not introduce an inadvertent scaling
        # from the units.
        if scale_factor.unit == u.one:
            scale_factor = scale_factor.value * position.unit

        Acos, Asin = scf_compute_coeffs_nbody(
            position,
            mass,
            N=Nmax,
            L=Lmax,
            a=scale_factor,
            radial_order=None,  # TODO!
            costheta_order=None,  # TODO!
            phi_order=None,  # TODO!
        )

        return GalpyPotentialWrapper(
            self._fitter(amp=mass.sum(), Acos=Acos, Asin=Asin, a=scale_factor),
            frame=self.frame,
        )

    # /def


# /class

##############################################################################
# END
