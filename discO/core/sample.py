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
import abc
import contextlib
import typing as T
from types import ModuleType

# THIRD PARTY
import astropy.coordinates as coord
import numpy as np

# PROJECT-SPECIFIC
import discO.type_hints as TH
from .core import CommonBase
from discO.utils import UnFrame, resolve_framelike, resolve_representationlike
from discO.utils.random import NumpyRNGContext, RandomLike

##############################################################################
# PARAMETERS

SAMPLER_REGISTRY = dict()  # key : sampler

##############################################################################
# CODE
##############################################################################


class PotentialSampler(CommonBase):
    """Sample a Potential.

    Parameters
    ----------
    potential
        The potential object.

    frame: frame-like or None (optional, keyword-only)
       The frame in which to sample.
    representation_type: |Representation| or None (optional, keyword-only)
        The coordinate representation.

    Returns
    -------
    `PotentialSampler` or subclass
        If `key` is not None, returns subclass.

    Other Parameters
    ----------------
    key : `~types.ModuleType` or str or None (optional, keyword-only)
        The key to which the `potential` belongs.
        If not provided (None, default) tries to infer from `potential`.

    Raises
    ------
    ValueError
        If directly instantiating a PotentialSampler (not subclass) and cannot
        find the appropriate subclass, identified using ``key``.

    """

    #################################################################
    # On the class

    _registry = SAMPLER_REGISTRY

    def __init_subclass__(cls, key: T.Union[str, ModuleType] = None):
        """Initialize subclass, adding to registry by `key`.

        This method applies to all subclasses, no matter the
        inheritance depth, unless the MRO overrides.

        """
        super().__init_subclass__(key=key)

        if key is not None:  # same trigger as CommonBase
            # cls._key defined in super()
            cls.__bases__[0]._registry[cls._key] = cls

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
        frame: T.Optional[TH.FrameLikeType] = None,
        representation_type: T.Optional[TH.RepresentationLikeType] = None,
        key: T.Union[ModuleType, str, None] = None,
        **kwargs,
    ):
        # The class PotentialSampler is a wrapper for anything in its registry
        # If directly instantiating a PotentialSampler (not subclass) we must
        # also instantiate the appropriate subclass. Error if can't find.
        if cls is PotentialSampler:
            # infer the key, to add to registry
            key = cls._infer_package(potential, key).__name__

            if key not in cls._registry:
                raise ValueError(
                    "PotentialSampler has no registered sampler for key: "
                    f"{key}",
                )

            # from registry. Registered in __init_subclass__
            kls = cls[key]
            return kls.__new__(
                kls,
                potential,
                key=None,
                frame=frame,
                representation_type=representation_type,
                **kwargs,
            )

        elif key is not None:
            raise ValueError(
                "Can't specify 'key' on PotentialSampler subclasses.",
            )

        return super().__new__(cls)

    # /def

    # ---------------------------------------------------------------

    def __init__(
        self,
        potential: T.Any,
        *,
        frame: T.Optional[TH.FrameLikeType] = None,
        representation_type: T.Optional[TH.RepresentationLikeType] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self._sampler = potential

        # frame and representation type
        # None stays as None
        self._frame: T.Optional[TH.FrameType] = None
        self._representation_type: T.Optional[TH.RepresentationType] = None
        self._frame = self._infer_frame(frame)
        self._representation_type = self._infer_representation(
            representation_type,
        )

    # /def

    # ---------------------------------------------------------------

    @property
    def potential(self):
        return self._sampler

    # /def

    @property
    def frame(self) -> T.Optional[TH.FrameType]:
        """The frame of the data. Can be None."""
        return self._frame

    # /def

    @property
    def representation_type(self) -> T.Optional[TH.RepresentationType]:
        """The representation type of the data. Can be None."""
        return self._representation_type

    # /def

    #################################################################
    # Sampling

    @abc.abstractmethod
    def __call__(
        self,
        n: int = 1,
        *,
        frame: T.Optional[TH.FrameLikeType] = None,
        representation_type: T.Optional[TH.RepresentationLikeType] = None,
        random: RandomLike = None,
        **kwargs,
    ) -> TH.SkyCoordType:
        """Sample.

        Parameters
        ----------
        n : int (optional)
            number of samples

        frame: frame-like or None (optional, keyword-only)
           The frame of the samples.
        representation_type: |Representation| or None (optional, keyword-only)
            The coordinate representation.

        random : int or |RandomState| or None (optional, keyword-only)
            Random state.
        **kwargs
            passed to underlying instance

        Returns
        -------
        :class:`~astropy.coordinates.SkyCoord`

        """
        raise NotImplementedError("Implemented in subclass.")

    # /def

    # ---------------------------------------------------------------

    def sample(
        self,
        n: T.Union[int, T.Sequence[int]] = 1,
        niter: int = 1,
        *,
        frame: T.Optional[TH.FrameLikeType] = None,
        representation_type: T.Optional[TH.RepresentationLikeType] = None,
        random: RandomLike = None,
        **kwargs,
    ) -> T.Union[TH.SkyCoordType, T.Sequence[TH.SkyCoordType]]:
        """Sample the potential.

        .. todo::

            - Subclass SkyCoord and have metadata mass and potential that
            carry-over. Or embed a SkyCoord in a table with the other
            attributes. or something so that doesn't need continual
            reassignment

            - manage random here?

        Parameters
        ----------
        n : int or sequence thereof (optional)
            Number of sample points.
            Can be a sequence of number of sample points
        niter : int (optional)
            Number of iterations. Must be > 0.

        frame: frame-like or None (optional, keyword-only)
           The frame of the samples.
        representation_type: |Representation| or None (optional, keyword-only)
            The coordinate representation.

        random : int or |RandomState| or None (optional, keyword-only)
            Random state or seed.
        **kwargs
            Passed to underlying instance

        Returns
        -------
        |SkyCoord| or array of |SkyCoord|
            singular if `n` is scalar, array if sequence.
            The shape of the SkyCoord is ``(niter, len(n))``
            where a scalar `n` has length 1.

        Raises
        ------
        ValueError
            If number if iterations not greater than 0.

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

        # premake array
        samples = np.empty(len(itersamp), dtype=coord.SkyCoord)

        # iterate thru number of samples
        for i, N in enumerate(itersamp):
            samps = [None] * niter  # premake array
            mass = [None] * niter  # premake array

            for j in range(0, niter):  # thru iterations
                # call sampler
                samp = self(
                    n=N,
                    frame=frame,
                    representation_type=representation_type,
                    random=random,
                    **kwargs,
                )
                # store samples & mass
                samps[j] = samp
                mass[j] = samp.mass

            # Now need to concatenate iterations
            if j == 0:  # 0-dimensional doesn't need concat
                sample = samps[0]
            else:
                sample = coord.concatenate(samps).reshape((N, niter))
                sample.mass = np.vstack(mass).T
                sample.potential = samp.potential  # all the same

            # concat iters stored in nsamp array
            samples[i] = sample

        # -----------

        if np.isscalar(n):  # nsamp scalar -> scalar
            return samples[0]
        else:
            return samples

    # /def

    #################################################################
    # utils

    def _infer_frame(
        self,
        frame: T.Optional[TH.FrameLikeType],
    ) -> T.Optional[TH.FrameType]:
        """Call `resolve_framelike`, but default to preferred frame.

        For frame is None ``resolve_framelike`` returns the default
        frame from the config file. Instead, we want the default
        frame of the potential. If that's None, return that.

        Parameters
        ----------
        frame : frame-like or None

        Returns
        -------
        `~astropy.coordinates.BaseCoordinateFrame` subclass instance
            Has no data.

        """
        if frame is None:  # get default
            frame = self._frame

        if frame is None:  # still None
            return UnFrame()  # TODO? move to resolve_framelike

        return resolve_framelike(frame)

    # /def

    def _infer_representation(
        self,
        representation_type: T.Optional[TH.RepresentationLikeType],
    ) -> T.Optional[TH.RepresentationType]:
        """Call `resolve_representation_typelike`, but default to preferred.

        Parameters
        ----------
        representation_type : representation-like or None

        Returns
        -------
        `~astropy.coordinates.BaseReprentation` subclass
            Has no data.

        """
        if representation_type is None:  # get default
            representation_type = self._representation_type

        if representation_type is None:  # still None
            return representation_type

        return resolve_representationlike(representation_type)

    # /def

    @staticmethod
    def _random_context(
        random: RandomLike,
    ) -> T.Union[NumpyRNGContext, contextlib.nullcontext]:
        """Get a random-state context manager.

        This is used to supplement samplers that do not have a random seed.

        """
        if isinstance(random, (int, np.random.RandomState)):
            context = NumpyRNGContext(random)
        else:  # None or Generator
            context = contextlib.nullcontext()

        return context

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
