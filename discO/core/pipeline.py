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

    measurerr : `MeasurementErrorSampler` (optional)

    fitter : `PotentialFitter` (optional)

    residual


    Raises
    ------
    ValueError
        If give `residual` and not also `fitter`.

    """

    def __init__(
        self,
        sampler: PotentialSampler,
        measurerr: T.Optional[MeasurementErrorSampler] = None,
        fitter: T.Optional[PotentialFitter] = None,
        residual=None,
    ):
        if residual is not None and fitter is None:
            raise ValueError

        self._sampler = sampler
        self._measurerr = measurerr
        self._fitter = fitter
        # self._residual = residual  # FIXME

        self._sample_res = None
        self._fit_res = None
        self._residual_res = None

    # /def

    def __call__(self, c_err=None, *args, **kwargs):

        samples = self._sampler(*args, **kwargs)
        self._sample_res = samples

        if self._measurerr is not None:
            # FIXME! the error needs to be generated by the sampler!
            samples = self._measurerr(samples, c_err=0.01)

        fit = self._fitter(samples)
        self._fit_res = fit

        # return self.residual(fit)  # FIXME

        return fit

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
                    "can't pass measurerr when there's already a fitter",
                )
            elif self._measurerr is not None:
                raise ValueError("already have one of those")

            pipe = Pipeline(
                sampler=self._sampler,
                measurerr=other,
                fitter=None,
                residual=None,
            )

        elif isinstance(other, PotentialFitter):

            if self._fitter is not None:
                raise ValueError("already have one of those")

            pipe = Pipeline(
                sampler=self._sampler,
                measurerr=self._measurerr,
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
        #         measurerr=self._measurerr,
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
                    "can't pass measurerr when there's already a fitter",
                )
            elif self._measurerr is not None:
                raise ValueError("already have one of those")

            self._measurerr = other

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
        #         measurerr=self._measurerr,
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
