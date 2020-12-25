# -*- coding: utf-8 -*-

"""Sample a Potential.

.. todo::

    Stretch Goals:

    - plugin for registering classes


Registering a Sampler
*********************
a


"""


__all__ = [
    "PotentialSampler",
]


##############################################################################
# IMPORTS

# BUILT-IN
import itertools
import typing as T
from types import MappingProxyType, ModuleType

# THIRD PARTY
import numpy as np
from astropy.coordinates import SkyCoord

# PROJECT-SPECIFIC
from .core import PotentialBase
from discO.common import FrameLikeType, SkyCoordType
from discO.utils import resolve_frame_like

##############################################################################
# PARAMETERS

SAMPLER_REGISTRY = dict()  # package : sampler

##############################################################################
# CODE
##############################################################################


class PotentialSampler(PotentialBase):
    """Sample a Potential.

    Parameters
    ----------
    potential
        The potential object.
    frame : frame-like or None (optional, keyword only)
        The preferred frame in which to sample.

    Returns
    -------
    `PotentialSampler` or subclass
        If `return_specific_class` is True, returns subclass.

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

    _registry = MappingProxyType(SAMPLER_REGISTRY)

    def __init_subclass__(cls, package: T.Union[str, ModuleType]):
        super().__init_subclass__(package=package)

        SAMPLER_REGISTRY[cls._package] = cls

    # /def

    def __new__(
        cls,
        potential: T.Any,
        *,
        frame: T.Optional[FrameLikeType] = None,
        package: T.Union[ModuleType, str, None] = None,
        return_specific_class: bool = False,
    ):
        self = super().__new__(cls)

        if cls is PotentialSampler:
            package = self._infer_package(potential, package)

            # from registry. Registered in __init_subclass__
            instance = SAMPLER_REGISTRY[package](potential)

            if return_specific_class:  # Whether to return class or subclass
                return instance
            else:  # return class, store instance
                self._instance = instance

        return self

    # /def

    def __init__(
        self, potential, *, frame: T.Optional[FrameLikeType] = None, **kwargs
    ):
        self._sampler = potential
        self._frame = resolve_frame_like(frame)

    # /def

    # ------------------------------------------
    # Sampling

    def __call__(
        self, n: int = 1, *, frame: T.Optional[FrameLikeType] = None, **kwargs
    ) -> SkyCoordType:
        """Sample.

        Parameters
        ----------
        n : int
            number of samples
        frame : frame-like or None
            output frame of samples
        **kwargs
            passed to underlying instance

        Returns
        -------
        `SkyCoord`

        """
        # # Get preferred frame
        # frame = self._preferred_frame_resolve(frame)

        # call on instance
        return self._instance(n=n, frame=frame, **kwargs)

    # /def

    def sample(
        self, n: int = 1, *, frame: T.Optional[FrameLikeType] = None, **kwargs
    ) -> SkyCoordType:
        """Draw a sample from the potential.

        Parameters
        ----------
        n : int
            number of sample points.
        frame : frame-like or None
            output frame of samples
        **kwargs
            passed to underlying instance

        Returns
        -------
        SkyCoord

        """
        # pass to __call__
        return self(n=n, frame=frame, **kwargs)

    # /def

    # TODO better name
    def resampler(
        self,
        niter: int,
        n: int = 1,
        *,
        frame: T.Optional[FrameLikeType] = None,
        sample_axis: int = -1,
        **kwargs,
    ):
        """Draw many samples from the potential.

        Parameters
        ----------
        niter : int
            Number of iterations
        n : int
            number of sample points.
        frame : frame-like or None
            output frame of samples
        sample_axis : int
            allowed values : 0, 1, -1
        **kwargs
            passed to underlying instance

        Yields
        ------
        SkyCoord
            sample

        """
        # # Get preferred frame
        # frame = self._preferred_frame_resolve(frame)

        iterniter = range(0, niter)
        if np.isscalar(n):
            itersamp = (n,)

        values = (iterniter, itersamp)
        values = (values, values[::-1])[sample_axis]

        for i, j in itertools.product(*values):
            N = (j, i)[sample_axis]  # todo more efficiently
            yield self(n=N, frame=frame, **kwargs)

    # /def

    def resample(
        self,
        niter: int,
        n: int = 1,
        *,
        frame: T.Optional[FrameLikeType] = None,
        sample_axis: int = -1,
        **kwargs,
    ):
        """Resample.

        .. todo::

            As a different return option,
            organize instead by N, return a list of SkyCoord, each of length
            ni in [n1, n2, n3, ...], with the 3rd axis being iteration.

        Parameters
        ----------
        niter : int
            Number of iterations
        n : int
            number of sample points.
        frame : frame-like or None
            output frame of samples
        sample_axis : int
            allowed values : 0, 1, -1
        **kwargs
            passed to underlying instance


        """
        sampler = self.resampler(
            niter=niter, n=n, frame=frame, sample_axis=sample_axis, **kwargs
        )

        # indices and values for niter
        idxniter = range(0, niter)
        # indices and values for nsamp
        if np.isscalar(n):
            idxsamp = (0,)
            lenn = 1
        else:
            lenn = len(n)
            idxsamp = range(lenn)

        indices = (idxniter, idxsamp)
        indices = (indices, indices[::-1])[sample_axis]

        nums = (niter, lenn)
        nums = (nums, nums[::-1])[sample_axis]
        array = np.empty(nums, dtype=SkyCoord)

        # TODO vectorized
        for (i, j), sample in zip(itertools.product(*indices), sampler):
            array[i, j] = sample

        return array

    # /def

    #################################################################
    # utils

    def _preferred_frame_resolve(self, frame: T.Optional[FrameLikeType]):
        """Call `resolve_frame_like`, but default to preferred frame.

        For frame is None ``resolve_frame_like`` returns the default
        frame from the config file. Instead, we want the default
        frame of the footprint.

        Returns
        -------
        `~astropy.coordinates.BaseCoordinateFrame` subclass instance
            Has no data.

        """
        if frame is None:
            frame = self._frame

        return resolve_frame_like(frame)

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
