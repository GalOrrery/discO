# -*- coding: utf-8 -*-

"""Sample a Potential.

Introduction
************

What is ``PotentialSampler``?


Registering a Sampler
*********************

Registering a sampler is easy. All you need to do is subclass
``PotentialSampler`` and provide information about the sampling object's
package.

For example

Let's do this for galpy

.. code-block::

    class GalpyPotentialSampler(PotentialSampler):



"""


__all__ = [
    "PotentialSampler",
]


##############################################################################
# IMPORTS

# BUILT-IN
import itertools
import typing as T
import warnings
from contextlib import nullcontext
from types import MappingProxyType, ModuleType

# THIRD PARTY
import numpy as np
from astropy.coordinates import SkyCoord, concatenate
from astropy.utils.misc import NumpyRNGContext

# PROJECT-SPECIFIC
from .core import PotentialBase
from discO.common import FrameLikeType, SkyCoordType
from discO.utils import resolve_framelike

##############################################################################
# PARAMETERS

SAMPLER_REGISTRY = dict()  # package : sampler

Random_Like = T.Union[int, np.random.Generator, np.random.RandomState, None]


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

    Raises
    ------
    ValueError
        If directly instantiating a PotentialSampler (not subclass) and cannot
        find the appropriate subclass, identified using ``package``.

    """

    #################################################################
    # On the class

    _registry = MappingProxyType(SAMPLER_REGISTRY)

    def __init_subclass__(cls, package: T.Union[str, ModuleType] = None):
        """Initialize subclass, adding to registry by `package`.

        This method applies to all subclasses, not matter the
        inheritance depth, unless the MRO overrides.

        """
        super().__init_subclass__(package=package)

        if package is not None:  # same trigger as PotentialBase
            # cls._package defined in super()
            SAMPLER_REGISTRY[cls._package] = cls

        # TODO? insist that subclasses define a __call__ method
        # this "abstractifies" the base-class even though it can be used
        # as a wrapper class.

    # /def

    #################################################################
    # On the instance

    def __new__(
        cls,
        potential: T.Any,
        *,
        frame: T.Optional[FrameLikeType] = None,
        package: T.Union[ModuleType, str, None] = None,
        return_specific_class: bool = False,
    ):
        self = super().__new__(cls)

        # The class PotentialSampler is a wrapper for anything in its registry
        # If directly instantiating a PotentialSampler (not subclass) we must
        # also instantiate the appropriate subclass. Error if can't find.
        if cls is PotentialSampler:
            # infer the package, to add to registry
            package = self._infer_package(potential, package)

            if package not in cls._registry:
                raise ValueError(
                    "PotentialSampler has no registered sampler for package: "
                    f"{package}",
                )

            # from registry. Registered in __init_subclass__
            instance = cls[package](potential)

            # Whether to return class or subclass
            # else continue, storing instance
            if return_specific_class:
                return instance

            self._instance = instance

        elif package is not None:
            raise ValueError(
                "Can't specify 'package' on PotentialSampler subclasses.",
            )

        elif return_specific_class is not False:
            warnings.warn("Ignoring argument `return_specific_class`")

        return self

    # /def

    # ---------------------------------------------------------------

    def __init__(
        self, potential, *, frame: T.Optional[FrameLikeType] = None, **kwargs
    ):
        super().__init__()
        self._sampler = potential
        self._frame = resolve_framelike(frame)

    # /def

    #################################################################
    # Sampling

    def __call__(
        self,
        n: int = 1,
        *,
        frame: T.Optional[FrameLikeType] = None,
        random: Random_Like = None,
        **kwargs,
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
        # call on instance
        # with NumpyRNGContext(random):
        if isinstance(random, int):
            ctx = NumpyRNGContext(random)
        else:  # None or Generator
            ctx = nullcontext()

        with ctx:
            return self._instance(n=n, frame=frame, random=random, **kwargs)

    # /def

    # ---------------------------------------------------------------

    def sample(
        self,
        n: int = 1,
        *,
        frame: T.Optional[FrameLikeType] = None,
        random: Random_Like = None,
        **kwargs,
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
        return self(n=n, frame=frame, random=random, **kwargs)

    # /def

    # ---------------------------------------------------------------

    def resampler(
        self,
        niter: int,
        n: T.Union[int, T.Sequence] = 1,
        *,
        frame: T.Optional[FrameLikeType] = None,
        sample_axis: int = -1,
        random: Random_Like = None,
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
        iterniter = range(0, niter)
        if np.isscalar(n):
            itersamp = (n,)
        else:
            itersamp = n

        values = (iterniter, itersamp)
        values = (values, values[::-1])[sample_axis]

        for i, j in itertools.product(*values):
            N = (j, i)[sample_axis]  # todo more efficiently
            yield self(n=N, frame=frame, random=random, **kwargs)

    # /def

    def resample(
        self,
        niter: int,
        n: int = 1,
        *,
        frame: T.Optional[FrameLikeType] = None,
        random: Random_Like = None,
        **kwargs,
    ):
        """Resample.

        Parameters
        ----------
        niter : int
            Number of iterations. Must be > 0.
        n : int or sequence
            Number of sample points.
            Can be a sequence of number of sample points
        frame : frame-like or None
            output frame of samples
        **kwargs
            passed to underlying instance

        Returns
        -------
        |SkyCoord| or array of |SkyCoord|
            singular if `n` is scalar, array if sequence.
            The shape of the SkyCoord is ``(niter, len(n))``
            where a scalar `n` has length 1.

        Raises
        ------
        ValueError
            if number if iterations not greater than 0.

        """
        # -----------
        # setup

        if not niter >= 1:
            raise ValueError("# of iterations not > 0.")

        if np.isscalar(n):
            itersamp = (n,)
        else:
            itersamp = n

        # -----------
        # resampling

        samples = np.empty(len(itersamp), dtype=SkyCoord)

        for i, N in enumerate(itersamp):
            samps = [None] * niter  # premake array

            for j in range(0, niter):
                samp = self(n=N, frame=frame, random=random, **kwargs)
                samps[j] = samp

            if j == 0:  # 0-dimensional doesn't need concat
                sample = samps[0]
            else:
                sample = concatenate(samps).reshape((N, niter))
            samples[i] = sample

        if np.isscalar(n):
            return samples[0]
        else:
            return samples

    # /def

    #################################################################
    # utils

    def _preferred_frame_resolve(self, frame: T.Optional[FrameLikeType]):
        """Call `resolve_framelike`, but default to preferred frame.

        For frame is None ``resolve_framelike`` returns the default
        frame from the config file. Instead, we want the default
        frame of the potential.

        Parameters
        ----------
        frame : frame-like or None

        Returns
        -------
        `~astropy.coordinates.BaseCoordinateFrame` subclass instance
            Has no data.

        """
        if frame is None:
            frame = self._frame

        return resolve_framelike(frame)

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
