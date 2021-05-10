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
    "MeshGridPotentialSampler",
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
import astropy.units as u
import numpy as np

# PROJECT-SPECIFIC
import discO.type_hints as TH
from .common import CommonBase
from .wrapper import PotentialWrapper
from discO.utils import resolve_representationlike
from discO.utils.decorators import frompyfunc
from discO.utils.pbar import get_progress_bar
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
        mtot = potential.total_mass() if total_mass is None else total_mass
        if not np.isfinite(mtot):  # divergent
            raise ValueError(
                "The potential`s mass is divergent, "
                "the argument `total_mass` cannot be None.",
            )

        # potential is checked in __new__ as a PotentialWrapper
        # we wrap again here to override the representation_type
        self._wrapper_potential = PotentialWrapper(
            potential,
            representation_type=representation_type,
        )

        self._total_mass: T.Optional[TH.QuantityType] = mtot

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
        n: int = 1,
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
        n: int = 1,
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
            # breakpoint()
            sample = coord.concatenate(samps).reshape((n, iterations))
            sample.mass = np.vstack(mass).T
            sample.potential = samp.potential  # all the same

        return sample

    # /def

    def run(
        self,
        n: int = 1,
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
        self,
        representation_type: TH.OptRepresentationLikeType,
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


##############################################################################


class MeshGridPotentialSampler(PotentialSampler):
    """Mesh-Grid Position Distribution.

    Parameters
    ----------
    pot : PotentialWrapper
    meshgrid : coord-like
        Should be "ij", not "xy" indexed.

    """

    def __new__(
        cls,
        potential: T.Any,
        meshgrid: coord.BaseRepresentation,
        *,
        total_mass: T.Optional[TH.QuantityType] = None,
        representation_type: TH.OptRepresentationLikeType = None,
        key: T.Union[ModuleType, str, None] = None,
        **other,
    ):
        return super().__new__(
            cls,
            potential,
            total_mass=total_mass,
            representation_type=representation_type,
            key=key,
            **other,
        )

    # /def

    def __init__(
        self,
        pot,
        meshgrid,
        *,
        total_mass: T.Optional[TH.QuantityType] = None,
        representation_type: TH.OptRepresentationLikeType = None,
        **defaults,
    ):
        super().__init__(
            pot,
            total_mass=total_mass,
            representation_type=representation_type,
            **defaults,
        )

        self._meshgrid = meshgrid
        self._gridshape = np.shape(meshgrid)
        self._dims = self._get_dimensions(meshgrid)

        weight = self.potential.density(meshgrid)[1].flatten()
        # distribution of flattened indices in [0, 1]
        self._index_partition = np.cumsum(weight).value
        self._normalization = np.sum(weight).value

    # /def

    @staticmethod
    def _get_dimensions(meshgrid):
        """Get dimensions.

        TODO! handle uneven dimensions

        Parameters
        ----------
        meshgrid : Representation

        """
        vals = meshgrid._values
        keys = tuple(vals.dtype.fields.keys())

        q1 = vals[keys[0]][1:, :, :] - vals[keys[0]][:-1, :, :]
        if np.allclose(q1, q1[0, 0, 0]):
            q1dim = q1[0, 0, 0] * meshgrid._units[keys[0]]
        else:
            raise NotImplementedError

        q2 = vals[keys[1]][:, 1:, :] - vals[keys[1]][:, :-1, :]
        if np.allclose(q2, q2[0, 0, 0]):
            q2dim = q2[0, 0, 0] * meshgrid._units[keys[1]]
        else:
            raise NotImplementedError

        q3 = vals[keys[2]][:, :, 1:] - vals[keys[2]][:, :, :-1]
        if np.allclose(q3, q3[0, 0, 0]):
            q3dim = q3[0, 0, 0] * meshgrid._units[keys[2]]
        else:
            raise NotImplementedError

        return q1dim, q2dim, q3dim

    # /def

    @property
    def _imapper(self):
        @frompyfunc(nin=1, nout=1)
        def imapper(uniform_draw):
            iflat = np.where(
                uniform_draw * self._normalization <= self._index_partition,
            )[0][0]
            i = np.unravel_index(iflat, self._gridshape)
            return i

        return imapper

    # /def

    def __call__(
        self, n: int, rng: T.Optional[np.random.Generator] = None, **kw
    ):
        """Sample.

        .. todo::

            handle uneven voxels

        Parameters
        ----------
        n : int
            number of sample points
        rng : `~numpy.random.Generator` or None
        **kw : Any

        Returns
        -------
        `~astropy.coordinates.BaseRepresentation`

        """
        _rng = np.random.default_rng() if rng is None else rng

        us = _rng.uniform(0, 1, size=n)
        indices = self._imapper(us)

        if all((np.shape(x) == () for x in self._dims)):
            # get voxel dims from meshgrid
            rep = self._make_samples_in_even_sized_voxels(
                indices,
                self._meshgrid,
                u.Quantity(self._dims),
            )
        else:
            raise NotImplementedError

        samples = coord.SkyCoord(self.frame.realize_frame(rep))

        # TODO! better storage of these properties, so stay when transform.
        samples.potential = self.potential
        # from init if divergent mass, preloaded total_mass() otherwise.
        samples.mass = np.ones(n) * self._total_mass / n  # AGAMA compatibility

        return samples

    # /def

    def _make_samples_in_even_sized_voxels(
        self,
        indices: np.ndarray,
        centers: coord.CartesianRepresentation,
        dims: u.Quantity,
        rng=None,
    ):
        rng = np.random.default_rng() if rng is None else rng
        offset = rng.uniform(-1, 1, size=(len(indices), 3)) / 2 * dims

        # TODO! vectorize
        mids = centers.xyz.T
        c = u.Quantity([mids[tuple(i)] for i in indices])
        return coord.CartesianRepresentation((c + offset).T)

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
