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
import inspect
import typing as T
import warnings
from types import ModuleType

# THIRD PARTY
import numpy as np

# PROJECT-SPECIFIC
from .core import PotentialBase
from discO.common import CoordinateType

##############################################################################
# PARAMETERS

FITTER_REGISTRY = dict()  # package : samplers

##############################################################################
# CODE
##############################################################################


class PotentialFitter(PotentialBase):
    """Fit a Potential.

    .. todo::

        This is registering by the package, which I think may be the wrong
        approach most packages have multiple fitters, which should be easily
        accessible.

    Parameters
    ----------
    potential_cls
        The type of potential with which to fit the data.

    Other Parameters
    ----------------
    key : `~types.ModuleType` or str or None (optional, keyword only)
        The key to which the `potential` belongs.
        If not provided (None, default) tries to infer from `potential`.
    return_specific_class : bool (optional, keyword only)
        Whether to return a `PotentialSampler` or key-specific subclass.
        This only applies if instantiating a `PotentialSampler`.
        Default False.

    """

    #################################################################
    # On the class

    _registry = FITTER_REGISTRY

    def __init_subclass__(cls, key: T.Union[str, ModuleType] = None):
        """Initialize subclass, adding to registry by `key`.

        This method applies to all subclasses, not matter the
        inheritance depth, unless the MRO overrides.

        """
        super().__init_subclass__(key=key)

        if key is not None:  # same trigger as PotentialBase
            # get the registry on this (the parent) object
            # cls._key defined in super()
            cls.__bases__[0]._registry[cls._key] = cls

        # TODO? insist that subclasses define a __call__ method
        # this "abstractifies" the base-class even though it can be used
        # as a wrapper class.

    # /defs

    #################################################################
    # On the instance

    def __new__(
        cls,
        potential_cls: T.Any,
        *,
        key: T.Union[ModuleType, str, None] = None,
        return_specific_class: bool = False,
        **kwargs,
    ):
        self = super().__new__(cls)
        self._fitter = potential_cls

        # The class PotentialFitter is a wrapper for anything in its registry
        # If directly instantiating a PotentialFitter (not subclass) we must
        # also instantiate the appropriate subclass. Error if can't find.
        if cls is PotentialFitter:
            # infer the key, to add to registry
            key = self._infer_package(potential_cls, key).__name__

            if key not in cls._registry:
                raise ValueError(
                    "PotentialFitter has no registered fitter for key: "
                    f"{key}",
                )

            # from registry. Registered in __init_subclass__
            # some subclasses accept the potential_cls as an argument,
            # others do not.
            subcls = cls._registry[key]
            sig = inspect.signature(subcls)
            ba = sig.bind_partial(potential_cls=potential_cls, **kwargs)
            ba.apply_defaults()

            instance = cls._registry[key](*ba.args, **ba.kwargs)

            if return_specific_class:
                return instance

            self._instance = instance

        elif key is not None:
            raise ValueError(
                "Can't specify 'key' on PotentialFitter subclasses.",
            )

        elif return_specific_class is not False:
            warnings.warn("Ignoring argument `return_specific_class`")

        return self

    # /def

    #######################################################
    # Fitting

    def __call__(
        self,
        sample: CoordinateType,
        # sample_err: T.Optional[CoordinateType] = None,
        **kwargs,
    ):
        """Fit.

        Parameters
        ----------
        sample : `SkyCoord`
        **kwargs
            passed to underlying instance

        Returns
        -------
        Potential : object

        """
        # call on instance
        return self._instance(
            sample,
            # c_err=c_err,
            **kwargs,
        )

    # /def

    def fit(
        self,
        sample: CoordinateType,
        **kwargs,
    ):
        """Fit.

        .. todo::

            Subclass SkyCoord and have metadata mass and potential that
            carry-over. Or embed a SkyCoord in a table with the other
            attributes. or something so that doesn't need continual
            reassignment

        Parameters
        ----------
        sample : `SkyCoord`
            can have shape (nsamp, ) or (nsamp, niter)
        # sample_err: T.Optional[CoordinateType] = None,
        **kwargs
            passed to underlying instance

        Returns
        -------
        Potential : object

        """
        if len(sample.shape) == 1:  # (nsamp, ) -> (nsamp, niter=1)
            mass, potential = sample.mass, sample.potential
            sample = sample.reshape((-1, 1))
            sample.mass, sample.potential = mass.reshape((-1, 1)), potential

        # shape (niter, )
        niter = sample.shape[1]
        fits = np.empty(niter, dtype=sample.potential.__class__)

        # (niter, nsamp) -> iter on niter
        for i, (samp, mass) in enumerate(zip(sample.T, sample.mass.T)):
            samp.mass, samp.potential = mass, sample.potential
            fits[i] = self(samp, **kwargs)

        if niter == 1:
            return fits[0]
        else:
            return fits

    # /def


# /class

# -------------------------------------------------------------------


##############################################################################
# END