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
import abc
import typing as T
from types import MappingProxyType, ModuleType

# THIRD PARTY
import numpy as np

# FIRST PARTY
from discO.utils.pbar import get_progress_bar

# PROJECT-SPECIFIC
import discO.type_hints as TH
from .common import CommonBase
from discO.utils.coordinates import (
    resolve_framelike,
    resolve_representationlike,
)

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

    frame: frame-like or None (optional, keyword-only)
        The frame of the fit potential.

        .. warning::

            Care should be taken that this matches the frame of the sampling
            potential.

    representation_type: |Representation| or None (optional, keyword-only)
        The coordinate representation.

    Other Parameters
    ----------------
    key : `~types.ModuleType` or str or None (optional, keyword-only)
        The key to which the `potential` belongs.
        If not provided (None, default) tries to infer from `potential`.

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
        # The class PotentialFitter is a wrapper for anything in its registry
        # If directly instantiating a PotentialFitter (not subclass) we must
        # also instantiate the appropriate subclass. Error if can't find.
        if cls is PotentialFitter:
            # infer the key, to add to registry
            key: str = cls._infer_package(potential_cls, key).__name__

            if key not in cls._registry:
                raise ValueError(
                    "PotentialFitter has no registered fitter for key: "
                    f"{key}",
                )

            # from registry. Registered in __init_subclass__
            kls = cls._registry[key]
            kwargs.pop("key", None)  # it's already used.
            return kls.__new__(
                kls, potential_cls=potential_cls, key=None, **kwargs
            )

        elif key is not None:
            raise ValueError(
                "Can't specify 'key' on PotentialFitter subclasses.",
            )

        return super().__new__(cls)

    # /def

    def __init__(
        self,
        potential_cls: T.Any,
        *,
        frame: TH.OptFrameLikeType = None,
        representation_type: TH.OptRepresentationLikeType = None,
        **params,
    ):
        self._potential_cls: T.Any = potential_cls
        self._frame: TH.FrameType = resolve_framelike(frame)
        self._representation_type: TH.OptRepresentationLikeType = (
            resolve_representationlike(representation_type)
            if representation_type not in (None, Ellipsis)
            else representation_type
        )

        # ----------------
        # kwargs
        # start by jettisoning baggage
        params.pop("key", None)
        self._kwargs: T.Dict[str, T.Any] = params

    # /def

    @property
    def potential_cls(self) -> T.Any:
        """The potential used for fitting."""
        return self._potential_cls

    @property
    def frame(self) -> TH.OptFrameLikeType:
        """The frame of the potential"""
        return self._frame

    # /def

    @property
    def representation_type(self) -> TH.OptRepresentationLikeType:
        """The representation type of the potential."""
        return self._representation_type

    # /def

    @property
    def potential_kwargs(self):
        return MappingProxyType(self._kwargs)

    # /def

    #######################################################
    # Fitting

    @abc.abstractmethod
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
            passed to fitting potential.

        Returns
        -------
        Potential : object

        """
        raise NotImplementedError("Implement in subclass.")

    # /def

    def _run_iter(
        self,
        sample: TH.CoordinateType,
        mass: T.Optional[TH.QuantityType] = None,
        *,
        progress: bool = True,
        **kwargs,
    ) -> object:
        """Fit.

        .. todo::

            - Subclass SkyCoord and have metadata mass and potential that
              carry-over. Or embed a SkyCoord in a table with the other
              attributes. or something so that doesn't need continual
              reassignment

        Parameters
        ----------
        sample : :class:`~astropy.coordinates.SkyCoord` instance
            can have shape (nsamp, ) or (nsamp, niter)

        **kwargs
            passed to fitting potential.

        Returns
        -------
        Potential : object

        """
        if mass is None:
            mass = sample.mass

        N, *iterations = sample.shape

        # get samples into the correct frame
        sample = sample.transform_to(self.frame)
        sample.mass = mass

        # get # iterations (& reshape no-iteration samples)
        if not iterations:  # only (N, ) -> (N, niter=1)
            iterations = 1
            sample = sample.reshape((-1, iterations))
            sample.mass = mass.reshape((-1, iterations))
        else:
            iterations = iterations[0]  # TODO! check shape

        # (iterations, N) -> iter on iterations
        with get_progress_bar(progress, iterations) as pbar:

            for samp, mass in zip(sample.T, sample.mass.T):
                pbar.update(1)

                # FIXME! need to assign these into sample
                # sample.potential
                samp.mass = mass

                yield self(
                    samp, mass=mass, **kwargs,
                )

    # /def

    def _run_batch(
        self,
        sample: TH.CoordinateType,
        mass: T.Optional[TH.QuantityType] = None,
        *,
        progress: bool = True,
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
            passed to fitting potential.

        Returns
        -------
        Potential : object

        """
        return np.array(
            tuple(
                self._run_iter(sample, mass=mass, progress=progress, **kwargs)
            )
        )

    # /def

    def run(
        self,
        sample: TH.CoordinateType,
        mass: T.Optional[TH.QuantityType] = None,
        *,
        batch: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> object:
        """Fit.

        Parameters
        ----------
        sample : :class:`~astropy.coordinates.SkyCoord` instance
            can have shape (nsamp, ) or (nsamp, niter)

        **kwargs
            passed to fitting potential.

        Returns
        -------
        Potential : object

        """
        run_func = self._run_batch if batch else self._run_iter

        return run_func(sample, mass=mass, progress=progress, **kwargs)

    # /def

    #######################################################
    # Utils

    def __repr__(self):
        s = super().__repr__()
        s += f"\n\tframe: {self.frame}"
        s += f"\n\tdefaults: {self._kwargs}"

        return s

    # /def


# /class

# -------------------------------------------------------------------


##############################################################################
# END
