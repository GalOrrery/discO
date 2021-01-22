# -*- coding: utf-8 -*-

"""Analysis Pipeline."""


__all__ = [
    "Pipeline",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T

# PROJECT-SPECIFIC
from .fitter import PotentialFitter
from .measurement import MeasurementErrorSampler
from .sample import PotentialSampler

##############################################################################
# PARAMETERS


##############################################################################
# CODE
##############################################################################


class Pipeline:
    """Analysis Pipeline.

    Parameters
    ----------
    sampler : `PotentialSampler`
        Sample the potential.

    measurer : `MeasurementErrorSampler` (optional)

    fitter : `PotentialFitter` (optional)

    residual :


    Raises
    ------
    ValueError
        If give `residual` and not also `fitter`.

    """

    def __init__(
        self,
        sample: PotentialSampler,
        measure: T.Optional[MeasurementErrorSampler] = None,
        fit: T.Optional[PotentialFitter] = None,
        residual: T.Optional[T.Callable] = None,
        statistic: T.Optional[T.Callable] = None,
    ):
        # can set `fit` without `measure`
        if fit is not None and measure is None:
            pass
        # cannot set `residual` without `fit`
        if residual is not None and fit is None:
            raise ValueError
        # cannot set `statistic` without `residual`
        if statistic is not None and residual is None:
            raise ValueError

        self._sampler = sample
        self._measure = measure
        self._fitter = fit
        self._residual = residual
        self._statistic = statistic

        self._sample_result = None
        self._measure_result = None
        self._fit_result = None
        self._residual_result = None
        self._statistic_result = None

    # /def

    def __call__(self, c_err=None, *args, **kwargs):
        """Call.

        Parameters
        ----------
        c_err : CoordinateType
        *args
        **kwargs

        Returns
        -------
        object

        """
        # ----------
        # sample

        result = self._sampler(*args, **kwargs)
        self._sample_result = result

        # ----------
        # measure

        if self._measure is not None:
            result = self._measure(result, c_err=c_err, *args, **kwargs)
            self._measure_result = result

        # ----------
        # fit

        result = self._fitter(result, **kwargs)
        self._fit_result = result

        # ----------
        # residual

        if self._residual is not None:

            result = self._residual(result, **kwargs)
            self._residual_result = result

        # ----------
        # statistic

        if self._statistic is not None:

            result = self._statistic(result, **kwargs)
            self._statistic_result = result

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


# /class


# -------------------------------------------------------------------


##############################################################################
# END
