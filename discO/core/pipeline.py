# -*- coding: utf-8 -*-

"""Analysis Pipeline."""


__all__ = [
    "Pipeline",
    "PipelineResult",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T
import weakref

# THIRD PARTY
import astropy.coordinates as coord
import numpy as np
import typing_extensions as TE

# FIRST PARTY
from discO.utils.pbar import get_progress_bar

# PROJECT-SPECIFIC
import discO.type_hints as TH
from .fitter import PotentialFitter
from .measurement import CERR_Type, MeasurementErrorSampler
from .residual import ResidualMethod
from .sample import PotentialSampler, RandomLike
from .wrapper import PotentialWrapper

##############################################################################
# CODE
##############################################################################


class Pipeline:
    """Analysis Pipeline.

    Parameters
    ----------
    sampler : `PotentialSampler`
        The object for sampling the potential.
        Can have a frame and representation type.

    measurer : `MeasurementErrorSampler` or None (optional)
        The object for re-sampling, given observational errors.

    fitter : `PotentialFitter` or None  (optional)

    residualer : None (optional)

    statistic : None (optional)


    Raises
    ------
    ValueError
        If can't set `residualer` without `fitter`.
        If can't set `statistic` without `residualer`.

    """

    def __init__(
        self,
        sampler: PotentialSampler,
        measurer: T.Optional[MeasurementErrorSampler] = None,
        fitter: T.Optional[PotentialFitter] = None,
        residualer: T.Optional[ResidualMethod] = None,
        statistic: T.Optional[T.Callable] = None,
    ):
        # CAN set `fitter` without `measurer`
        if fitter is not None and measurer is None:
            pass
        # can't set `residualer` without `fitter`
        if residualer is not None and fitter is None:
            raise ValueError("Can't set `residualer` without `fitter`.")
        # can't set `statistic` without `residualer`
        if statistic is not None and residualer is None:
            raise ValueError("Can't set `statistic` without `residualer`")

        if sampler is not None and fitter is not None:
            if fitter.frame != sampler.frame:
                raise ValueError(
                    "sampler and fitter must have the same frame.",
                )
            # nice, but not necessary
            # if fitter.representation_type != sampler.representation_type:
            #     raise ValueError(
            #         "sampler and fitter must have the same representation.",
            #     )

        self._sampler = sampler
        self._measurer = measurer
        self._fitter = fitter
        self._residualer = residualer
        self._statisticer = statistic

        self._result = None

    # /def

    # ---------------------------------------------------------------

    # @property
    # def frame(self) -> TH.OptFrameType:
    #     """The frame or None or Ellipse."""
    #     return self._frame

    # # /def

    # @property
    # def representation_type(self) -> TH.OptRepresentationLikeType:
    #     """The representation type or None or Ellipse."""
    #     return self._representation_type

    # # /def

    @property
    def sampler(self) -> PotentialSampler:
        """The sampler."""
        return self._sampler

    # /def

    @property
    def potential(self) -> T.Any:
        """The potential from which we sample."""
        return self.sampler.potential

    # /def

    @property
    def potential_frame(self) -> TH.OptFrameType:
        """The frame in which the potential is sampled and fit."""
        return self.sampler.frame

    # /def

    @property
    def potential_representation_type(self) -> TH.OptRepresentationType:
        """Representation type of potential."""
        return self.sampler.representation_type

    # /def

    @property
    def measurer(self) -> T.Optional[MeasurementErrorSampler]:
        """The measurer."""
        return self._measurer

    # /def

    @property
    def observer_frame(self) -> TH.OptFrameType:
        """Observer frame."""
        return self._measurer.frame

    # /def

    @property
    def observer_representation_type(self) -> TH.OptRepresentationType:
        """Observer representation type."""
        return self._measurer.representation_type

    # /def

    @property
    def fitter(self) -> T.Optional[PotentialFitter]:
        """The fitter."""
        return self._fitter

    # /def

    @property
    def residualer(self) -> T.Optional[ResidualMethod]:
        """The residual function."""
        return self._residualer

    # /def

    @property
    def statisticer(self) -> T.Optional[T.Callable]:
        """The statistic function."""
        return self._statisticer

    # /def

    #################################################################
    # Call

    def __call__(
        self,
        n_or_sample: T.Union[int, TH.SkyCoordType],
        *,
        # sampler
        total_mass: TH.QuantityType = None,
        # observer
        c_err: T.Optional[CERR_Type] = None,
        # residual
        observable: T.Optional[str] = None,
        # extra
        random: T.Optional[RandomLike] = None,
        **kwargs,
    ) -> object:
        """Run the pipeline for 1 iteration.

        Parameters
        ----------
        n_or_sample : int or (N,) SkyCoord (optional)
            number of sample points

        observable : str or None (optional, keyword-only)

        **kwargs
            Passed to ``run``.

        Returns
        -------
        (1,) :class:`PipelineResult`

        Notes
        -----
        This actually calls the more general function ``run``, with
        ``niter`` pinned to 1.

        """
        # We will make a pipeline result and then work thru it.
        result = PipelineResult(self)

        # TODO! resolve_randomstate(random)
        # we need to resolve the random state now, so that an `int` isn't
        # set as the same random state each time
        random = (
            np.random.RandomState(random)
            if not isinstance(random, np.random.RandomState)
            else random
        )

        # ----------
        # 1) sample

        if isinstance(n_or_sample, int):
            sample: TH.SkyCoordType = self.sampler(
                n_or_sample, total_mass=total_mass, random=random, **kwargs,
            )
        elif isinstance(n_or_sample, coord.SkyCoord):
            sample = n_or_sample
        else:
            raise TypeError

        result["sample"][0] = sample

        # ----------
        # 2) measure
        # optionally skip this step if c_err is False

        if self.measurer is not None and c_err is not False:

            sample: TH.SkyCoordType = self.measurer(
                sample, random=random, c_err=c_err, **kwargs,
            )
            result["measured"][0] = sample

        # ----------
        # 3) fit
        # we force the fit to be in the same frame & representation type
        # as the samples.

        fit_pot: T.Any = self.fitter(sample, **kwargs)
        result["fit"][0] = fit_pot

        # ----------
        # 4) residual
        # only if 3)

        if self.residualer is not None:

            resid: T.Any = self.residualer(
                fit_pot,
                original_potential=self.potential,
                observable=observable,
                **kwargs,
            )
            result["residual"][0] = resid

        # ----------
        # 5) statistic
        # only if 4)

        if self.statisticer is not None:

            stat: T.Any = self.statisticer(resid, **kwargs)
            result["statistic"][0] = stat

        # ----------

        self._result: PipelineResult = result  # link to most recent result
        return result[0]

    # /defs

    # -----------------------------------------------------------------

    def _run_iter(
        self,
        n_or_sample: T.Union[int, TH.SkyCoordType],
        iterations: int = 1,
        *,
        # observer
        c_err: T.Optional[CERR_Type] = None,
        # residual
        observable: T.Optional[str] = None,
        # extra
        random: T.Optional[RandomLike] = None,
        progress: bool = True,
        **kwargs,
    ) -> object:
        """Run pipeline, yielding :class:`PipelineResult` over ``iterations``.

        .. todo::

            - See ``emcee`` for the backend.

        Parameters
        ----------
        n_or_sample : int (optional)
            number of sample points

        iterations : int (optional)
            Number of iterations. Must be > 0.
            Only used if `n_or_sample` is int.

        random : int or |RandomState| or None (optional, keyword-only)
            Random state or seed.

        original_pot : object or None (optional, keyword-only)
        observable : str or None (optional, keyword-only)

        Yields
        ------
        :class:`PipelineResult`
            For each of ``iterations``

        """
        # reshape n_or_sample
        if isinstance(n_or_sample, int):
            n_or_sample = [n_or_sample] * iterations
        elif isinstance(n_or_sample, coord.SkyCoord):
            if len(n_or_sample.shape) == 1:  # scalar
                n_or_sample = [n_or_sample]
            else:  # TODO! not use jank iterator

                def jank_iter(samples, masses):
                    for samp, mass in zip(samples, masses):
                        samp.mass = mass
                        yield samp

                n_or_sample = jank_iter(n_or_sample.T, n_or_sample.mass.T)

        # iterate over number of iterations
        # for _ in tqdm(range(niter), desc="Running Pipeline...", total=niter):
        with get_progress_bar(progress, iterations) as pbar:

            for arg in n_or_sample:
                pbar.update(1)

                yield self(
                    arg,
                    random=random,
                    # observer
                    c_err=c_err,
                    # residual
                    observable=observable,
                    **kwargs,
                )

        # /with

    # /def

    # ---------------------------------------------------------------

    def _run_batch(
        self,
        n_or_sample: T.Union[int, T.Sequence[int]],
        iterations: int = 1,
        *,
        random: T.Optional[RandomLike] = None,
        # sampler
        total_mass: TH.QuantityType = None,
        # observer
        c_err: T.Union[CERR_Type, None, TE.Literal[False]] = None,
        # fitter
        # residual
        observable: T.Optional[str] = None,
        progress: bool = False,
        **kwargs,
    ) -> object:
        """Call.

        Parameters
        ----------
        n : int (optional)
            number of sample points
        iterations : int (optional)
            Number of iterations. Must be > 0.

        random : int or |RandomState| or None (optional, keyword-only)
            Random state or seed.
            In order that a sequence of samples is different in each element
            we here resolve random seeds into a |RandomState|.

        original_pot : object or None (optional, keyword-only)
        observable : str or None (optional, keyword-only)

        Returns
        -------
        :class:`PipelineResult`

        """
        # reshape n_or_sample
        if isinstance(n_or_sample, coord.SkyCoord):
            if len(n_or_sample.shape) == 1:  # scalar
                iterations = 1
            else:
                iterations = n_or_sample.shape[1]

        # We will make a pipeline result and then work thru it.
        results = np.recarray(
            (iterations,),
            dtype=[
                ("sample", coord.SkyCoord),
                ("measured", coord.SkyCoord),
                ("fit", PotentialWrapper),
                ("residual", object),
                ("statistic", object),
            ],
        ).view(PipelineResult)
        results._parent_ref = weakref.ref(self)

        run_gen = self._run_iter(
            n_or_sample,
            iterations,
            random=random,
            total_mass=total_mass,
            c_err=c_err,
            observable=observable,
            progress=progress,
            **kwargs,
        )

        for i, result in enumerate(run_gen):
            results[i] = result

        return results

    # /defs

    # ---------------------------------------------------------------

    def run(
        self,
        n_or_sample: T.Union[int, T.Sequence[int]],
        iterations: int = 1,
        *,
        random: T.Optional[RandomLike] = None,
        # sampler
        total_mass: TH.QuantityType = None,
        # observer
        c_err: T.Union[CERR_Type, None, TE.Literal[False]] = None,
        # residual
        observable: T.Optional[str] = None,
        # extra
        batch: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> object:
        """Call.

        Parameters
        ----------
        n : int (optional)
            number of sample points
        iterations : int (optional)
            Number of iterations. Must be > 0.

        random : int or |RandomState| or None (optional, keyword-only)
            Random state or seed.
            In order that a sequence of samples is different in each element
            we here resolve random seeds into a |RandomState|.

        original_pot : object or None (optional, keyword-only)
        observable : str or None (optional, keyword-only)

        Returns
        -------
        :class:`PipelineResult`

        """
        run_func = self._run_batch if batch else self._run_iter

        # we need to resolve the random state now, so that an `int` isn't
        # set as the same random state each time
        random = (
            np.random.RandomState(random)
            if not isinstance(random, np.random.RandomState)
            else random
        )

        return run_func(
            n_or_sample,
            iterations,
            random=random,
            total_mass=total_mass,
            c_err=c_err,
            observable=observable,
            progress=progress,
            **kwargs,
        )

    # /def

    #################################################################
    # utils

    def __repr__(self) -> str:
        """String Representation.

        Returns
        -------
        str

        """
        s = (
            "Pipeline:\n"
            f"    sampler: {self._sampler}\n"
            f"    measurer: {self._measurer}\n"
            f"    fitter: {self._fitter}\n"
            f"    residual: {self._residualer}\n"
            f"    statistic: {self._statisticer}\n"
        )

        return s

    # /def


# /class


#####################################################################


class PipelineResult(np.recarray):
    """:class:`~discO.core.Pipeline` Evaluation Result."""

    def __new__(
        cls,
        pipe: Pipeline,
        sample: T.Optional[TH.SkyCoordType] = None,
        measured: T.Optional[TH.SkyCoordType] = None,
        fit: T.Optional[T.Any] = None,
        residual: T.Optional[T.Any] = None,
        statistic: T.Optional[T.Any] = None,
    ):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.array(
            [(sample, measured, fit, residual, statistic)],
            dtype=[
                ("sample", coord.SkyCoord),
                ("measured", coord.SkyCoord),
                ("fit", PotentialWrapper),
                ("residual", object),
                ("statistic", object),
            ],
        ).view(cls)[:]
        # add the new attribute to the created instance
        obj._parent_ref = weakref.ref(pipe)
        # Finally, we must return the newly created object:
        return obj

    # /def

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self._parent_ref = getattr(obj, "_parent_ref", None)

    # /def

    def __repr__(self):
        s = super().__repr__()
        s = s.replace("rec.array", self.__class__.__name__)

        return s

    # /def

    # -----------------
    # Properties

    @property
    def _parent(self):
        return self._parent_ref()

    # /def

    # def _attr(self, name):
    #     attr = super().__getattr__(name)
    #     if len(self) == 1:
    #         return attr[0]
    #     return attr

    # @property
    # def sample(self):
    #     return self._attr("sample")

    # @property
    # def measured(self):
    #     return self._attr("measured")

    # @property
    # def fit(self):
    #     return self._attr("fit")

    # @property
    # def residual(self):
    #     return self._attr("residual")

    # @property
    # def statistic(self):
    #     return self._attr("statistic")

    #################################################################
    # Plotting


##############################################################################
# END
