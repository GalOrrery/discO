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
from .fitter import PotentialFitter
from .measurement import CERR_Type, MeasurementErrorSampler
from .sample import PotentialSampler, Random_Like
from discO.type_hints import FrameLikeType

##############################################################################
# CODE
##############################################################################


class Pipeline:
    """Analysis Pipeline.

    Parameters
    ----------
    sampler : `PotentialSampler`
        Sample the potential.

    measurer : `MeasurementErrorSampler` or None (optional)

    fitter : `PotentialFitter` or None  (optional)

    residualer : None (optional)

    statisticer : None (optional)


    Raises
    ------
    ValueError
        If give `residual` and not also `fitter`.

    """

    def __init__(
        self,
        sampler: PotentialSampler,
        measurer: T.Optional[MeasurementErrorSampler] = None,
        fitter: T.Optional[PotentialFitter] = None,
        residualer: T.Optional[T.Callable] = None,
        statisticer: T.Optional[T.Callable] = None,
    ):
        # can set `fit` without `measure`
        if fitter is not None and measurer is None:
            pass
        # cannot set `residualer` without `fitter`
        if residualer is not None and fitter is None:
            raise ValueError
        # cannot set `statistic` without `residualer`
        if statisticer is not None and residualer is None:
            raise ValueError

        self._sampler = sampler
        self._measurer = measurer
        self._fitter = fitter
        self._residualer = residualer
        self._statisticer = statisticer

        self._result = None

    # /def

    #################################################################
    # Call

    def __call__(
        self,
        n: int,
        *,
        frame: T.Optional[FrameLikeType] = None,
        random: T.Optional[Random_Like] = None,
        c_err: T.Optional[CERR_Type] = None,
        original_pot: T.Optional[object] = None,
        observable: T.Optional[str] = None,
        **kwargs,
    ):
        """Call.

        Parameters
        ----------
        n
        frame : frame-like or None (optional, keyword-only)
        random : |RandomGenerator| or int or None (optional, keyword-only)
        c_err : coord-like or callable or number (optional, keyword-only)
        original_pot : object or None (optional, keyword-only)
        observable : str or None (optional, keyword-only)
        **kwargs

        Returns
        -------
        object

        """
        return self.run(
            n,
            niter=1,
            frame=frame,
            random=random,
            c_err=c_err,
            original_pot=original_pot,
            observable=observable,
            **kwargs,
        )

    # /defs

    def run(
        self,
        n: T.Union[int, T.Sequence[int]],
        niter: int = 1,
        *,
        frame: T.Optional[FrameLikeType] = None,
        random: T.Optional[Random_Like] = None,
        c_err: T.Optional[CERR_Type] = None,
        original_pot: T.Optional[object] = None,
        observable: T.Optional[str] = None,
        **kwargs,
    ):
        """Call.

        Parameters
        ----------
        *args
        **kwargs

        Returns
        -------
        object

        """
        result = PipelineResult(self)
        # Now evaluate, passing around the output -> input

        # ----------
        # sample

        oi = self._sampler.sample(
            n, niter=niter, frame=frame, random=random, **kwargs
        )
        result._samples = oi

        # ----------
        # measure

        if self._measurer is not None:

            oi = self._measurer.resample(
                oi, c_err=c_err, random=random, **kwargs
            )
            result._measured = oi

        # ----------
        # fit

        oi = self._fitter.fit(oi, **kwargs)
        result._fit = oi

        # ----------
        # residual

        if self._residualer is not None:

            oi = self._residualer.evaluate(
                oi,
                original_pot=original_pot,
                observable=observable,
                **kwargs,
            )
            result._residual = oi

        # ----------
        # statistic

        if self._statisticer is not None:

            oi = self._statisticer(oi, **kwargs)
            result._statistic = oi

        # ----------

        self._result = result  # link
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
