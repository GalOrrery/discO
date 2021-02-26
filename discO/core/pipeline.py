# -*- coding: utf-8 -*-

"""Analysis Pipeline."""


__all__ = [
    "Pipeline",
    "PipelineResult",
]


##############################################################################
# IMPORTS

# BUILT-IN
import copy
import typing as T
import weakref

# THIRD PARTY
import astropy.coordinates as coord
import numpy as np

# PROJECT-SPECIFIC
import discO.type_hints as TH
from .fitter import PotentialFitter
from .measurement import CERR_Type, MeasurementErrorSampler
from .residual import ResidualMethod
from .sample import PotentialSampler, RandomLike
from discO.setup_package import tqdm
from discO.utils.coordinates import (
    resolve_framelike,
    resolve_representationlike,
)

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
        *,
        frame: TH.OptFrameLikeType = None,
        representation_type: TH.OptRepresentationLikeType = None,
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

        # -------------------
        # Frame and Representation type

        self._frame: TH.OptFrameType = (
            frame
            if (frame is None or frame is Ellipsis)
            else resolve_framelike(frame)
        )
        self._representation_type: TH.OptRepresentationType = (
            representation_type
            if representation_type in (None, Ellipsis)
            else resolve_representationlike(representation_type)
        )

    # /def

    # ---------------------------------------------------------------

    @property
    def frame(self) -> TH.OptFrameType:
        """The frame or None or Ellipse."""
        return self._frame

    # /def

    @property
    def representation_type(self) -> TH.OptRepresentationLikeType:
        """The representation type or None or Ellipse."""
        return self._representation_type

    # /def

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
        n: int,
        *,
        random: T.Optional[RandomLike] = None,
        # sampler
        sample_and_fit_frame: TH.OptFrameType = None,
        sample_and_fit_representation_type: TH.OptRepresentationType = None,
        # observer
        c_err: T.Optional[CERR_Type] = None,
        observer_frame: TH.OptFrameType = None,
        observer_representation_type: TH.OptRepresentationType = None,
        # fitter
        # residual
        observable: T.Optional[str] = None,
        **kwargs,
    ) -> object:
        """Run the pipeline.

        Parameters
        ----------
        n : int (optional)
            number of sample points
        niter : int (optional)
            Number of iterations. Must be > 0.

        observable : str or None (optional, keyword-only)

        **kwargs
            Passed to ``run``.

        Returns
        -------
        :class:`PipelineResult`

        """
        # due to line length
        sample_and_fit_rep_type = sample_and_fit_representation_type
        return self.run(
            n,
            niter=1,
            random=random,
            sample_and_fit_frame=sample_and_fit_frame,
            sample_and_fit_representation_type=sample_and_fit_rep_type,
            # observer
            c_err=c_err,
            observer_frame=observer_frame,
            observer_representation_type=observer_representation_type,
            # fitter
            # residual
            observable=observable,
            **kwargs,
        )

    # /defs

    def run_with_samples(
        self,
        samples,
        *,
        random: T.Optional[RandomLike] = None,
        # sampler
        sample_and_fit_frame: TH.OptFrameType = None,
        sample_and_fit_representation_type: TH.OptRepresentationType = None,
        # observer
        c_err: T.Union[CERR_Type, None, T.Literal[False]] = None,
        observer_frame: TH.OptFrameType = None,
        observer_representation_type: TH.OptRepresentationType = None,
        # fitter
        # residual
        observable: T.Optional[str] = None,
        **kwargs,
    ) -> object:
        """Call.

        Parameters
        ----------
        n : int (optional)
            number of sample points
        niter : int (optional)
            Number of iterations. Must be > 0.

        random : int or |RandomState| or None (optional, keyword-only)
            Random state or seed.

        observer_frame: frame-like or None or Ellipse (optional, keyword-only)
           The frame of the observational errors, ie the frame in which
            the error function should be applied along each dimension.
            None defaults to the value set at initialization.
        observer_representation_type: representation-resolvable (optional, keyword-only)
            The coordinate representation in which to resample along each
            dimension.
            None defaults to the value set at initialization.

        original_pot : object or None (optional, keyword-only)
        observable : str or None (optional, keyword-only)

        Returns
        -------
        :class:`PipelineResult`

        """
        # We will make a pipeline result and then work thru it.
        result = PipelineResult(self)
        # Now evaluate, passing around the output -> input

        # TODO! resolve_randomstate(random)
        # we need to resolve the random state now, so that an `int` isn't
        # set as the same random state each time
        random = (
            np.random.RandomState(random)
            if not isinstance(random, np.random.RandomState)
            else random
        )

        # frame
        sample_and_fit_frame = (
            self.frame
            if sample_and_fit_frame is None
            else sample_and_fit_frame
        )  # NOTE! can still be None

        # representation type
        sample_and_fit_representation_type = (
            self.representation_type
            if sample_and_fit_representation_type is None
            else sample_and_fit_representation_type
        )

        # ----------
        # 1) sample

        # first store,
        result._samples: TH.SkyCoordType = samples

        # we are going to make an assumption that if `samples` is a list of
        # SkyCoord, a la ``run(n=[list])``, that all the SC have the same
        # frame
        if sample_and_fit_frame is None:
            if not isinstance(samples, coord.SkyCoord):
                _frame = samples[0].frame.replicate_without_data()

            else:
                _frame = samples.frame.replicate_without_data()

            sample_and_fit_frame = _frame

        # ----------
        # 2) measure
        # optionally skip this step if c_err is False

        if self.measurer is not None and c_err is not False:

            samples: TH.SkyCoordType = self.measurer.run(
                samples,
                random=random,
                c_err=c_err,
                frame=observer_frame,
                representation_type=observer_representation_type,
                **kwargs,
            )
            result._measured: TH.SkyCoordType = samples

        # ----------
        # 3) fit
        # we force the fit to be in the same frame & representation type
        # as the samples.

        fit_pot: T.Any = self.fitter.run(
            samples,
            frame=sample_and_fit_frame,
            representation_type=sample_and_fit_representation_type,
            **kwargs,
        )
        result._fit: T.Any = fit_pot

        # ----------
        # 4) residual
        # only if 3)

        if self.residualer is not None:

            resid: T.Any = self.residualer.run(
                fit_pot,
                original_potential=self.potential,
                observable=observable,
                **kwargs,
            )
            result._residual: T.Any = resid

        # ----------
        # 5) statistic
        # only if 4)

        if self.statisticer is not None:

            stat: T.Any = self.statisticer(resid, **kwargs)
            result._statistic: T.Any = stat

        # ----------

        self._result: PipelineResult = result  # link to most recent result
        return result

    # /defs

    def run(
        self,
        n: T.Union[int, T.Sequence[int]],
        niter: int = 1,
        *,
        random: T.Optional[RandomLike] = None,
        # sampler
        total_mass: TH.QuantityType = None,
        sample_and_fit_frame: TH.OptFrameType = None,
        sample_and_fit_representation_type: TH.OptRepresentationType = None,
        # observer
        c_err: T.Union[CERR_Type, None, T.Literal[False]] = None,
        observer_frame: TH.OptFrameType = None,
        observer_representation_type: TH.OptRepresentationType = None,
        # fitter
        # residual
        observable: T.Optional[str] = None,
        **kwargs,
    ) -> object:
        """Call.

        Parameters
        ----------
        n : int (optional)
            number of sample points
        niter : int (optional)
            Number of iterations. Must be > 0.

        random : int or |RandomState| or None (optional, keyword-only)
            Random state or seed.

        observer_frame: frame-like or None or Ellipse (optional, keyword-only)
           The frame of the observational errors, ie the frame in which
            the error function should be applied along each dimension.
            None defaults to the value set at initialization.
        observer_representation_type: representation-resolvable (optional, keyword-only)
            The coordinate representation in which to resample along each
            dimension.
            None defaults to the value set at initialization.

        original_pot : object or None (optional, keyword-only)
        observable : str or None (optional, keyword-only)

        Returns
        -------
        :class:`PipelineResult`

        """
        # we need to resolve the random state now, so that an `int` isn't
        # set as the same random state each time
        random = (
            np.random.RandomState(random)
            if not isinstance(random, np.random.RandomState)
            else random
        )

        # frame
        sample_and_fit_frame = (
            self.frame
            if sample_and_fit_frame is None
            else sample_and_fit_frame
        )  # NOTE! can still be None

        # representation type
        sf_rep_type = (
            self.representation_type
            if sample_and_fit_representation_type is None
            else sample_and_fit_representation_type
        )

        # ----------
        # 1) sample

        samples: TH.SkyCoordType = self.sampler.run(
            n,
            niter=niter,
            frame=sample_and_fit_frame,
            representation_type=sf_rep_type,
            total_mass=total_mass,
            random=random,
            **kwargs,
        )

        result = self.run_with_samples(
            samples,
            random=random,
            sample_and_fit_frame=sample_and_fit_frame,
            sample_and_fit_representation_type=sf_rep_type,
            c_err=c_err,
            observer_frame=observer_frame,
            observer_representation_type=observer_representation_type,
            observable=observable,
        )
        return result

    # /defs

    def run_iter(
        self,
        n: int,
        niter: int = 1,
        *,
        random: T.Optional[RandomLike] = None,
        # sampler
        sample_and_fit_frame: TH.OptFrameType = None,
        sample_and_fit_representation_type: TH.OptRepresentationType = None,
        # observer
        c_err: T.Optional[CERR_Type] = None,
        observer_frame: TH.OptFrameType = None,
        observer_representation_type: TH.OptRepresentationType = None,
        # fitter
        # residual
        observable: T.Optional[str] = None,
        **kwargs,
    ) -> object:
        """Run pipeline, yielding :class:`PipelineResult` over ``niter``.

        Parameters
        ----------
        n : int (optional)
            number of sample points

            .. warning::

                does not (yet) support a sequence of int
        niter : int (optional)
            Number of iterations. Must be > 0.

        random : int or |RandomState| or None (optional, keyword-only)
            Random state or seed.

        observer_frame: frame-like or None or Ellipse (optional, keyword-only)
           The frame of the observational errors, ie the frame in which
            the error function should be applied along each dimension.
            None defaults to the value set at initialization.
        observer_representation_type: representation-resolvable (optional, keyword-only)
            The coordinate representation in which to resample along each
            dimension.
            None defaults to the value set at initialization.

        original_pot : object or None (optional, keyword-only)
        observable : str or None (optional, keyword-only)

        Yields
        ------
        :class:`PipelineResult`
            For each of ``niter``

        """
        # we need to resolve the random state now, so that an `int` isn't
        # set as the same random state each time
        random = (
            np.random.RandomState(random)
            if not isinstance(random, np.random.RandomState)
            else random
        )

        # only for line length
        sample_and_fit_rep_type = sample_and_fit_representation_type

        # iterate over number of iterations
        for _ in tqdm(range(niter), desc="Running Pipeline...", total=niter):
            yield self.run(
                n=n,
                random=random,
                # sampler
                sample_and_fit_frame=sample_and_fit_frame,
                sample_and_fit_representation_type=sample_and_fit_rep_type,
                # observer
                c_err=c_err,
                observer_frame=observer_frame,
                observer_representation_type=observer_representation_type,
                # fitter
                # residual
                observable=observable,
                **kwargs,
            )

    # /def

    #################################################################
    # Pipeline

    def __or__(
        self,
        other: T.Union[
            MeasurementErrorSampler,
            PotentialFitter,
            ResidualMethod,
            T.Callable,
        ],
    ):
        """Combine ``other`` into a copy of Pipeline.

        Parameters
        ----------
        other : object
            Combine into Pipeline.

        Returns
        -------
        :class:`Pipeline`
            With `other` mixed in.

        """
        # copy  # TODO, manually?
        pipe = copy.deepcopy(self)

        # add in-place
        pipe.__ior__(other)

        return pipe

    # /def

    def __ior__(self, other):  # FIXME
        """Combine ``other`` into Pipeline.

        Parameters
        ----------
        other : object
            Combine into Pipeline.

        Returns
        -------
        :class:`Pipeline`
            Current pipeline With `other` mixed in.

        """
        # For PotentialSampler
        if self._sampler is None:  # a sanity check
            raise Exception("sampler is None. Why'd you do that?")
        elif isinstance(other, PotentialSampler):
            raise TypeError("can't do that.")

        # For MeasurementErrorSampler
        elif isinstance(other, MeasurementErrorSampler):

            if self._fitter is not None:
                raise TypeError(
                    "can't pass measurer when there's already a fitter",
                )
            elif self._measurer is not None:
                raise ValueError("already have one of those")

            self._measurer = other

        elif isinstance(other, PotentialFitter):

            if self._fitter is not None:
                raise ValueError("already have one of those")

            self._fitter = other

        # elif isinstance(other, residual):  # TODO

        #     if self._fitter is None:
        #         raise TypeError(
        #             "need a fitter"
        #         )
        #     elif self._residual is not None:
        #         raise ValueError("already have one of those")

        #     pipe = Pipeline(
        #         sampler=self._sampler,
        #         measurer=self._measurer,
        #         fitter=self._residual,
        #         residual=other,
        #     )

        else:

            raise TypeError("'other' not one of supported types.")

        return self

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


class PipelineResult:
    """:class:`~discO.core.Pipeline` Evaluation Result."""

    def __init__(
        self,
        pipe: Pipeline,
        samples: T.Optional[TH.SkyCoordType] = None,
        measured: T.Optional[TH.SkyCoordType] = None,
        fit: T.Optional[T.Any] = None,
        residual: T.Optional[T.Any] = None,
        statistic: T.Optional[T.Any] = None,
    ):
        # reference to parent
        self._parent_ref = weakref.ref(pipe)

        # results
        self._samples: T.Optional[TH.SkyCoordType] = samples
        self._measured: T.Optional[TH.SkyCoordType] = measured
        self._fit: T.Optional[T.Any] = fit
        self._residual: T.Optional[T.Any] = residual
        self._statistic: T.Optional[T.Any] = statistic

    # /def

    # -----------------
    # Properties

    @property
    def _parent(self):
        return self._parent_ref()

    # /def

    @property
    def samples(self) -> T.Optional[TH.SkyCoordType]:
        """The samples."""
        return self._samples

    # /def

    @property
    def potential_frame(self):
        return self.samples.frame

    # /def

    @property
    def potential_representation_type(self):
        return self.samples.representation_type

    # /def

    @property
    def measured(self) -> T.Optional[TH.SkyCoordType]:
        """The re-samples."""
        return self._measured

    # /def

    @property
    def observation_frame(self):
        return self.measured.frame

    # /def

    @property
    def observation_representation_type(self):
        return self.measured.representation_type

    # /def

    @property
    def fit(self) -> T.Optional[T.Any]:
        """The fit potential."""
        return self._fit

    # /def

    @property
    def residual(self) -> T.Optional[T.Any]:
        """The residual between the original and fit potential."""
        return self._residual

    # /def

    @property
    def statistic(self) -> T.Optional[T.Any]:
        """The statistic on the residual."""
        return self._statistic

    # /def

    #################################################################
    # Plotting


# /class

##############################################################################
# END
