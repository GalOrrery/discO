# -*- coding: utf-8 -*-

"""Decorators."""


__all__ = [
    "frompyfunc",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T

# THIRD PARTY
import numpy as np

##############################################################################
# CODE
##############################################################################


def frompyfunc(nin: int, nout: int) -> T.Callable:
    """`~numpy.frompyfunc` decorator factory.

    Parameters
    ----------
    nin : int
    nout : int

    Returns
    -------
    frompyfunc_dec : callable
        `~numpy.frompyfunc` decorator.

    """

    def frompyfunc_dec(func: T.Callable) -> T.Callable:
        return np.frompyfunc(func, nin, nout)

    return frompyfunc_dec


# /def


# -------------------------------------------------------------------


##############################################################################
# END
