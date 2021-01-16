# -*- coding: utf-8 -*-

"""Fit a Potential.

Registering a Fitter
********************
a

"""


__all__ = [
    "PotentialFitter",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T
from types import MappingProxyType, ModuleType

# PROJECT-SPECIFIC
from .core import PotentialBase
from discO.common import CoordinateType

##############################################################################
# PARAMETERS

FITTER_REGISTRY = dict()  # package : sampler
# _fitter_package_registry = dict()  # sampler : package


##############################################################################
# CODE
##############################################################################


class PotentialFitter(PotentialBase):
    """Fit a Potential.

    Parameters
    ----------
    pot_type
        The type of potential with which to fit the data.

    Other Parameters
    ----------------
    package : `~types.ModuleType` or str or None (optional, keyword only)
        The package to which the `potential` belongs.
        If not provided (None, default) tries to infer from `potential`.
    return_specific_class : bool (optional, keyword only)
        Whether to return a `PotentialSampler` or package-specific subclass.
        This only applies if instantiating a `PotentialSampler`.
        Default False.

    """

    _registry = MappingProxyType(FITTER_REGISTRY)

    def __init_subclass__(cls, package: T.Union[str, ModuleType]):
        super().__init_subclass__(package=package)

        FITTER_REGISTRY[cls._package] = cls

    # /defs

    def __new__(
        cls,
        pot_type: T.Any,
        *,
        package: T.Union[ModuleType, str, None] = None,
        return_specific_class: bool = False,
    ):
        self = super().__new__(cls)

        if cls is PotentialFitter:
            package = self._infer_package(pot_type, package)
            instance = FITTER_REGISTRY[package](pot_type)

            if return_specific_class:
                return instance
            else:
                self._instance = instance

        return self

    # /def

    # def __init__(self, pot_type, **kwargs):
    #     self._fitter = pot_type

    #################################################################
    # Fitting

    def __call__(
        self,
        c: CoordinateType,
        c_err: T.Optional[CoordinateType] = None,
        **kwargs,
    ):
        return self._instance(c, c_err=c_err, **kwargs)

    # /def

    def fit(
        self,
        c: CoordinateType,
        c_err: T.Optional[CoordinateType] = None,
        **kwargs,
    ):
        # pass to __call__
        return self(c, c_err=c_err, **kwargs)

    # /def

    # # TODO? wrong place for this
    # def draw_realization(self, c, c_err=None, **kwargs):
    #     """Draw a realization given the errors.

    #     .. todo::

    #         rename this function

    #         better than equal Gaussian errors

    #     See Also
    #     --------
    #     :meth:`~discO.core.sampler.draw_realization`

    #     """

    #     # for i in range(nrlz):

    #     #     # FIXME! this is shit
    #     #     rep = c.represent_as(coord.CartesianRepresentation)
    #     #     rep_err = c_err.re

    #     #     new_c = c.realize_frame(new_rep)

    #     #     yield self(c, c_err=c_err, **kwarg)

    # # /def


# /class

# -------------------------------------------------------------------


##############################################################################
# END
