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
from discO.utils.pbar import get_progress_bar

# PROJECT-SPECIFIC
import discO.type_hints as TH
from .common import CommonBase
from .wrapper import PotentialWrapper
from discO.utils import resolve_representationlike
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
    potential : :class:`~discO.PotentialWrapper`
        The potential object. Must have a frame.

    total_mass : |Quantity| or None (optional)
        The total mass of the potential.
        Necessary if the mass is divergent, must be None otherwise.
    representation_type: |Representation| or None (optional, keyword-only)
        The coordinate representation in which to return samples.
        If None (default) uses representation type from `potential`.

    **defaults
        default arguments for sampling parameters. In ``run``, parameters with
        default `None` will draw from these defaults.

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
        - If directly instantiating a PotentialSampler (not subclass) and
          cannot find the appropriate subclass, identified using ``key``.
        - If the total mass of the potential is divergent and `total_mass`
          is None.
    TypeError
        If `potential` is not :class:`~discO.PotentialWrapper`

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

    # /def

    #################################################################
    # On the instance

    def __new__(
        cls,
        potential: T.Any,
        *,
        total_mass: T.Optional[TH.QuantityType] = None,
        representation_type: TH.OptRepresentationLikeType = None,
        key: T.Union[ModuleType, str, None] = None,
        **other,
    ):
        if not isinstance(potential, PotentialWrapper):
            raise TypeError("potential must be a PotentialWrapper.")

        # The class PotentialSampler is a wrapper for anything in its registry
        # If directly instantiating a PotentialSampler (not subclass) we must
        # also instantiate the appropriate subclass. Error if can't find.
        if cls is PotentialSampler:
            # infer the key, to add to registry
            # TODO! infer_package should return str
            key = cls._infer_package(potential.wrapped, key).__name__

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
                total_mass=total_mass,
                representation_type=representation_type,
                **other,
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
        potential: PotentialWrapper,
        *,
        total_mass: T.Optional[TH.QuantityType] = None,
        representation_type: TH.OptRepresentationLikeType = None,
        **defaults,
    ) -> None:
        super().__init__()

        # check that the mass is not divergent.
        # and that the argument total_mass is correct
        mtot = potential.total_mass()
        if not np.isfinite(mtot):
            raise ValueError(
                "The potential`s mass is divergent, "
                "the argument `total_mass` cannot be None."
            )
        elif total_mass is not None:  # mass is not divergent.
            raise ValueError(
                "The potential`s mass is not divergent, "
                "the argument `total_mass` must be None."
            )
        else:  # not divergent and total_mass is None
            total_mass = mtot

        # potential is checked in __new__ as a PotentialWrapper
        # we wrap again here to override the representation_type
        self._wrapper_potential = PotentialWrapper(
            potential, representation_type=representation_type,
        )

        self._total_mass: T.Optional[TH.QuantityType] = total_mass

        # keep the kwargs
        self._defaults: dict = defaults

    # /def

    # ---------------------------------------------------------------

    @property
    def potential(self):
        """The potential, wrapped."""
        return self._wrapper_potential

    # /def

    @property
    def _potential(self):
        """The wrapped potential."""
        return self.potential.wrapped

    # /def

    @property
    def frame(self) -> TH.FrameType:
        """The frame of the data. Can be None."""
        return self.potential.frame

    # /def

    @property
    def representation_type(self) -> T.Optional[TH.RepresentationType]:
        """The representation type of the data. Can be None."""
        return self.potential.representation_type

    # /def

    #################################################################
    # Sampling

    @abc.abstractmethod
    def __call__(
        self,
        n: int = 1,
        *,
        representation_type: TH.OptRepresentationLikeType = None,
        random: RandomLike = None,
        **kwargs,
    ) -> TH.SkyCoordType:
        """Sample.

        Parameters
        ----------
        n : int (optional)
            number of samples

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

    def _run_iter(
        self,
        n: T.Union[int, T.Sequence[int]] = 1,
        iterations: int = 1,
        *,
        representation_type: TH.OptRepresentationLikeType = None,
        random: RandomLike = None,
        # extra
        progress: bool = True,
        **kwargs,
    ) -> TH.SkyCoordType:
        """Iteratively sample the potential.

        .. todo::

            - Subclass SkyCoord and have metadata mass and potential that
            carry-over. Or embed a SkyCoord in a table with the other
            attributes. or something so that doesn't need continual
            reassignment

        Parameters
        ----------
        n : int (optional)
            Number of sample points.
        iterations : int (optional)
            Number of iterations. Must be > 0.

        representation_type: |Representation| or None (optional, keyword-only)
            The coordinate representation.
        random : int or |RandomState| or None (optional, keyword-only)
            Random state or seed.
        progress : bool (optional, keyword-only)
            If True, a progress bar will be shown as the sampler progresses.
            If a string, will select a specific tqdm progress bar - most
            notable is 'notebook', which shows a progress bar suitable for
            Jupyter notebooks. If False, no progress bar will be shown.
        **kwargs
            Passed to underlying instance

        Yields
        ------
        |SkyCoord|
            If `sequential` is False.
            The shape of the SkyCoord is ``(n, niter)``
            where a scalar `n` has length 1.

        Raises
        ------
        ValueError
            If number if iterations not greater than 0.

        """
        with get_progress_bar(progress, iterations) as pbar:
            for i in range(0, iterations):  # thru iterations
                pbar.update(1)
                yield self(
                    n=n,
                    representation_type=representation_type,
                    random=random,
                    **kwargs,
                )

    # /def

    # ---------------------------------------------------------------

    def _run_batch(
        self,
        n: T.Union[int, T.Sequence[int]] = 1,
        iterations: int = 1,
        *,
        representation_type: TH.OptRepresentationLikeType = None,
        random: RandomLike = None,
        # extra
        progress: bool = True,
        **kwargs,
    ) -> TH.SkyCoordType:
        """Sample the potential.

        Parameters
        ----------
        n : int (optional)
            Number of sample points.
        iterations : int (optional)
            Number of iterations. Must be > 0.
        representation_type: |Representation| or None (optional, keyword-only)
            The coordinate representation.
        random : int or |RandomState| or None (optional, keyword-only)
            Random state or seed.
        sequential : bool (optional, keyword-only)
            Whether to batch sample or yield sequentially.
        **kwargs
            Passed to underlying instance

        Return
        ------
        |SkyCoord|
            If `sequential` is True.
            The shape of the SkyCoord is ``(n,)``

        Raises
        ------
        ValueError
            If number if iterations not greater than 0.

        """
        samps = [None] * iterations  # premake array
        mass = [None] * iterations  # premake array

        run_gen = self._run_iter(
            n=n,
            iterations=iterations,
            representation_type=representation_type,
            random=random,
            progress=progress,
            **kwargs,
        )

        for j, samp in enumerate(run_gen):  # thru iterations
            # store samples & mass
            samps[j] = samp
            mass[j] = samp.mass

        # Now need to concatenate iterations
        if j == 0:  # 0-dimensional doesn't need concat
            sample = samps[0]
        else:
            sample = coord.concatenate(samps).reshape((n, iterations))
            sample.mass = np.vstack(mass).T
            sample.potential = samp.potential  # all the same

        return sample

    # /def

    def run(
        self,
        n: T.Union[int, T.Sequence[int]] = 1,
        iterations: int = 1,
        *,
        representation_type: TH.OptRepresentationLikeType = None,
        random: RandomLike = None,
        # extra
        batch: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> TH.SkyCoordType:
        """Iteratively sample the potential.

        .. todo::

            - Subclass SkyCoord and have metadata mass and potential that
            carry-over. Or embed a SkyCoord in a table with the other
            attributes. or something so that doesn't need continual
            reassignment

        Parameters
        ----------
        n : int (optional)
            Number of sample points.
        iterations : int (optional)
            Number of iterations. Must be > 0.

        representation_type: |Representation| or None (optional, keyword-only)
            The coordinate representation.
        random : int or |RandomState| or None (optional, keyword-only)
            Random state or seed.
        progress : bool (optional, keyword-only)
            If True, a progress bar will be shown as the sampler progresses.
            If a string, will select a specific tqdm progress bar - most
            notable is 'notebook', which shows a progress bar suitable for
            Jupyter notebooks. If False, no progress bar will be shown.
        **kwargs
            Passed to underlying instance

        Yields
        ------
        |SkyCoord|
            If `sequential` is False.
            The shape of the SkyCoord is ``(n, niter)``
            where a scalar `n` has length 1.

        Raises
        ------
        ValueError
            If number if iterations not greater than 0.

        """
        run_func = self._run_batch if batch else self._run_iter

        # need to resolve RandomState
        if not isinstance(random, np.random.RandomState):
            random = np.random.RandomState(random)

        if not iterations >= 1:
            raise ValueError("# of iterations not > 0.")
        elif not isinstance(n, int):
            raise TypeError

        return run_func(
            n=n,
            iterations=iterations,
            representation_type=representation_type,
            random=random,
            progress=progress,
            **kwargs,
        )

    # /def

    #################################################################
    # utils

    def _infer_representation(
        self, representation_type: TH.OptRepresentationLikeType,
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
            representation_type = self.representation_type

        if representation_type is None:  # still None
            return None

        return resolve_representationlike(representation_type)

    # /def

    @staticmethod
    def _random_context(
        random: RandomLike,
    ) -> T.Union[NumpyRNGContext, contextlib.suppress]:
        """Get a random-state context manager.

        This is used to supplement samplers that do not have a random seed.

        """
        if isinstance(random, (int, np.random.RandomState)):
            context = NumpyRNGContext(random)
        else:  # None or Generator
            context = contextlib.suppress()

        return context

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
