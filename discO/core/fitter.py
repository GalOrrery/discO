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
from types import MappingProxyType, ModuleType

# THIRD PARTY
import numpy as np

# PROJECT-SPECIFIC
import discO.type_hints as TH
from .core import CommonBase, PotentialWrapper

##############################################################################
# PARAMETERS

FITTER_REGISTRY: T.Dict[str, CommonBase] = dict()  # package : samplers

##############################################################################
# CODE
##############################################################################


class PotentialFitter(CommonBase):
    """Fit a Potential.

    Parameters
    ----------
    potential_cls
        The type of potential with which to fit the data.

    Other Parameters
    ----------------
    key : `~types.ModuleType` or str or None (optional, keyword-only)
        The key to which the `potential` belongs.
        If not provided (None, default) tries to infer from `potential`.
    return_specific_class : bool (optional, keyword-only)
        Whether to return a `PotentialSampler` or key-specific subclass.
        This only applies if instantiating a `PotentialSampler`.
        Default False.

    """

    #################################################################
    # On the class

    _registry = FITTER_REGISTRY

    def __init_subclass__(
        cls, key: T.Union[str, ModuleType, None] = None,
    ) -> None:
        """Initialize subclass, adding to registry by `key`.

        This method applies to all subclasses, not matter the
        inheritance depth, unless the MRO overrides.

        """
        super().__init_subclass__(key=key)

        if key is not None:  # same trigger as CommonBase
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
        **kwargs,  # includes frame
    ):
        self = super().__new__(cls)

        # The class PotentialFitter is a wrapper for anything in its registry
        # If directly instantiating a PotentialFitter (not subclass) we must
        # also instantiate the appropriate subclass. Error if can't find.
        if cls is PotentialFitter:
            # infer the key, to add to registry
            key: str = self._infer_package(potential_cls, key).__name__

            if key not in cls._registry:
                raise ValueError(
                    "PotentialFitter has no registered fitter for key: "
                    f"{key}",
                )

            # from registry. Registered in __init_subclass__
            return cls._registry[key]

        elif key is not None:
            raise ValueError(
                "Can't specify 'key' on PotentialFitter subclasses.",
            )

        return self

    # /def

    def __init__(
        self,
        potential_cls: T.Any,
        *,
        frame: T.Optional[TH.FrameLikeType] = None,
        representation_type: T.Optional[TH.RepresentationType] = None,
        **kwargs,
    ):
        self._fitter: T.Any = potential_cls
        self._frame: T.Optional[TH.FrameLikeType] = frame
        self._representation_type = representation_type

        self._kwargs: T.Dict[str, T.Any] = kwargs

    # /def

    @property
    def potential(self) -> T.Any:
        """The potential used for fitting."""
        return self._fitter

    @property
    def frame(self) -> T.Optional[TH.FrameLikeType]:
        """The frame of the potential"""
        return self._frame

    # /def

    @property
    def potential_kwargs(self):
        return MappingProxyType(self._kwargs)

    # /def

    #######################################################
    # Fitting

    def __call__(
        self,
        sample: TH.CoordinateType,
        mass: T.Optional[TH.QuantityType] = None,
        **kwargs,
    ) -> object:
        """Fit a potential given the data.

        Parameters
        ----------
        c : :class:`~astropy.coordinates.SkyCoord` instance
        **kwargs
            passed to underlying instance

        Returns
        -------
        Potential : object

        """
        # return (
        #     PotentialWrapper(potential, frame=self.frame)
        #     if not isinstance(potential, PotentialWrapper)
        #     else potential
        # )
        raise NotImplementedError()

    # /def

    def fit(
        self,
        sample: TH.CoordinateType,
        mass: T.Optional[TH.QuantityType] = None,
        **kwargs,
    ) -> object:
        """Fit.

        .. todo::

            Subclass SkyCoord and have metadata mass and potential that
            carry-over. Or embed a SkyCoord in a table with the other
            attributes. or something so that doesn't need continual
            reassignment

        Parameters
        ----------
        sample : :class:`~astropy.coordinates.SkyCoord` instance
            can have shape (nsamp, ) or (nsamp, niter)
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
            fits[i] = self(samp, mass=mass, **kwargs)

        if niter == 1:
            return fits[0]
        # else:
        return fits

    # /def


# /class

# -------------------------------------------------------------------


##############################################################################
# END
