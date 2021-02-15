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

# PROJECT-SPECIFIC
import discO.type_hints as TH
from .fitter import PotentialFitter
from .measurement import CERR_Type, MeasurementErrorSampler
from .sample import PotentialSampler, RandomLike

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
        residualer: T.Optional[T.Callable] = None,
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
            if fitter.representation_type != sampler.representation_type:
                raise ValueError(
                    "sampler and fitter must have the same representation.",
                )

        self._sampler = sampler
        self._measurer = measurer
        self._fitter = fitter
        self._residualer = residualer
        self._statisticer = statistic

        self._result = None

    # /def

    # ---------------------------------------------------------------

    @property
    def potential(self):
        """The potential from which we sample."""
        return self.sampler.potential

    # /def

    @property
    def potential_frame(self) -> T.Optional[TH.FrameType]:
        """The frame in which the potential is sampled and fit."""
        return self.sampler.frame

    # /def

    @property
    def potential_representation_type(
        self,
    ) -> T.Optional[TH.RepresentationType]:
        return self.sampler.representation_type

    # /def

    @property
    def observer_frame(self) -> T.Optional[TH.FrameType]:
        return self._measurer.frame

    # /def

    @property
    def observer_representation_type(
        self,
    ) -> T.Optional[TH.RepresentationType]:
        return self._measurer.representation_type

    # /def

    #################################################################
    # Call

    def __call__(
        self,
        n: int,
        *,
        random: T.Optional[RandomLike] = None,
        # sampler
        # frame: T.Optional[TH.FrameLikeType] = None,
        # representation_type: T.Optional[TH.RepresentationType] = None,
        # observer
        c_err: T.Optional[CERR_Type] = None,
        observer_frame: T.Optional[TH.FrameLikeType] = None,
        observer_representation_type: T.Optional[TH.RepresentationType] = None,
        # fitter
        # residual
        observable: T.Optional[str] = None,
        **kwargs,
    ):
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
        return self.run(
            n,
            niter=1,
            random=random,
            # frame=frame,
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

    def run(
        self,
        n: T.Union[int, T.Sequence[int]],
        niter: int = 1,
        *,
        random: T.Optional[RandomLike] = None,
        # sampler
        # frame: T.Optional[TH.FrameLikeType] = None,
        # representation_type: T.Optional[TH.RepresentationType] = None,
        # observer
        c_err: T.Optional[CERR_Type] = None,
        observer_frame: T.Optional[TH.FrameLikeType] = None,
        observer_representation_type: T.Optional[TH.RepresentationType] = None,
        # fitter
        # residual
        observable: T.Optional[str] = None,
        **kwargs,
    ):
        """Call.

        Parameters
        ----------
        n : int (optional)
            number of sample points
        niter : int (optional)
            Number of iterations. Must be > 0.

        random : int or |RandomState| or None (optional, keyword-only)
            Random state or seed.

        observer_frame: frame-like or None (optional, keyword-only)
           The frame of the observational errors, ie the frame in which
            the error function should be applied along each dimension.
            None defaults to the value set at initialization.
        observer_representation_type: |Representation| or None (optional, keyword-only)
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

        # ----------
        # 1) sample

        oi = self._sampler.sample(
            n,
            niter=niter,
            # frame=frame, representation_type=representation_type,
            random=random,
            **kwargs,
        )
        result._samples = oi

        # ----------
        # 2) measure

        if self._measurer is not None:

            oi = self._measurer.resample(
                oi,
                random=random,
                # c_err=c_err,
                frame=observer_frame,
                representation_type=observer_representation_type,
                **kwargs,
            )
            result._measured = oi

        # ----------
        # 3) fit
        # we forced the fit to be in the same frame & representation type
        # as the samples.

        oi = self._fitter.fit(oi, **kwargs)
        result._fit = oi

        # ----------
        # 4) residual

        if self._residualer is not None:

            oi = self._residualer.evaluate(
                oi,
                original_pot=self.potential,
                observable=observable,
                **kwargs,
            )
            result._residual = oi

        # ----------
        # 5) statistic

        if self._statisticer is not None:

            oi = self._statisticer(oi, **kwargs)
            result._statistic = oi

        # ----------

        self._result = result  # link to most recent result
        return result

    # /defs

    #################################################################
    # Pipeline

    def __or__(self, other):  # FIXME

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

            pipe = Pipeline(
                sampler=self._sampler,
                measurer=other,
                fitter=None,
                residual=None,
            )

        elif isinstance(other, PotentialFitter):

            if self._fitter is not None:
                raise ValueError("already have one of those")

            pipe = Pipeline(
                sampler=self._sampler,
                measurer=self._measurer,
                fitter=other,
                residual=None,
            )

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

        return pipe

    # /def

    def __ior__(self, other):  # FIXME

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


# -------------------------------------------------------------------


class PipelineResult:
    """:class:`~discO.core.Pipeline` Evaluation Result."""

    def __init__(
        self,
        pipe: Pipeline,
        samples=None,
        measured=None,
        fit=None,
        residual=None,
        statistic=None,
    ):
        # reference to parent
        self._parent_ref = weakref.ref(pipe)

        # results
        self._samples = samples
        self._measured = measured
        self._fit = fit
        self._residual = residual
        self._statistic = statistic

    # /def

    # -----------------
    # Properties

    @property
    def _parent(self):
        return self._parent_ref()

    # /def

    @property
    def samples(self):
        return self._samples

    @property
    def measured(self):
        return self._measured

    @property
    def fit(self):
        return self._fit

    @property
    def residual(self):
        return self._residual

    @property
    def statistic(self):
        return self._statistic


# /class

##############################################################################
# END
