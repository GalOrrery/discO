# -*- coding: utf-8 -*-

"""Statistic."""

__all__ = [
    # functions
    "rms",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import numpy as np

# LOCAL
import discO.type_hints as TH

##############################################################################
# CODE
##############################################################################


def rms(resid, **kwargs) -> TH.QuantityType:
    """Root Mean Square.

    .. warning::

        Right now only works on vectorfields

    Parameters
    ----------
    resid
    **kwargs

    Returns
    -------
    Quantity

    """
    N: int = np.prod(resid.shape)

    return np.sqrt(np.nansum(np.square(resid.norm())) / N)


# /def

# -------------------------------------------------------------------


##############################################################################
# END
