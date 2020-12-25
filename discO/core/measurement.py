# -*- coding: utf-8 -*-

"""Sample a Potential.

.. todo::

    Stretch Goals:

    - plugin for registering classes


Registering a Measurement Sampler
*********************************
a


"""


__all__ = [
    "MeasurementErrorSampler",
    "GaussianMeasurementErrorSampler",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T
from abc import abstractmethod
from types import MappingProxyType

# THIRD PARTY
import numpy as np
from astropy.coordinates import (
    BaseCoordinateFrame,
    BaseRepresentation,
    SkyCoord,
)

# PROJECT-SPECIFIC
from .core import PotentialBase
from discO.common import FrameLikeType

##############################################################################
# PARAMETERS

MEASURE_REGISTRY = dict()  # package : measurer

##############################################################################
# CODE
##############################################################################


class MeasurementErrorSampler(PotentialBase):
    """Resample given observational errors.

    Parameters
    ----------
    potential
        The potential object.
    frame : frame-like or None (optional, keyword only)
        The preferred frame in which to sample.

    Returns
    -------
    `PotentialSampler` or subclass
        If `return_specific_class` is True, returns subclass.

    Other Parameters
    ----------------
    package : `~types.ModuleType` or str or None (optional, keyword only)
        The package to which the `potential` belongs.
        If not provided (None, default) tries to infer from `potential`.
    return_specific_class : bool (optional, keyword only)
        Whether to return a `PotentialSampler` or package-specific subclass.
        This only applies if instantiating a `PotentialSampler`.
        Default False.

    """

    _registry = MappingProxyType(MEASURE_REGISTRY)

    def __init_subclass__(cls):
        super().__init_subclass__()

        MEASURE_REGISTRY[cls.__name__] = cls

    # /defs

    def __init__(self, c_err=None):

        self.c_err = c_err

    # /def

    @abstractmethod
    def __call__(self, c: FrameLikeType, c_err: FrameLikeType):

        pass

    # /def


# /class

# -------------------------------------------------------------------


class GaussianMeasurementErrorSampler(MeasurementErrorSampler):
    """Gaussian."""

    def __call__(
        self,
        c: FrameLikeType,
        c_err: T.Union[FrameLikeType, float, None] = None,
        random: T.Union[int, np.random.Generator, None] = None,
    ):
        """Draw a realization given the errors.

        .. todo::

            - the velocities
            - confirm that units work nicely
            - figure out phase wrapping when draws over a wrap

        Parameters
        ----------
        c : SkyCoord
        c_err : SkyCoord

        random : `~numpy.random.Generator` or int or None
            The random number generator

        """
        # see 'default_rng' docs for details
        random = np.random.default_rng(random)

        nd = len(c.data._values.dtype)  # the shape

        units = c.data._units
        pos = c.data._values.view(dtype=np.float64).reshape(-1, nd)

        if isinstance(c_err, (BaseCoordinateFrame, SkyCoord)):
            d_pos = c_err.data._values.view(dtype=np.float64).reshape(-1, nd)
        elif isinstance(c_err, BaseRepresentation):
            raise NotImplementedError("Not yet")
        elif np.isscalar(c_err):
            d_pos = c_err
        else:
            raise NotImplementedError("Not yet")

        # draw realization
        # this will have no units. We will need to add those
        new_pos = random.normal(loc=pos, scale=d_pos, size=pos.shape)

        new_rep = c.data.__class__(
            **{n: p * unit for p, (n, unit) in zip(new_pos.T, units.items())}
        )

        new_c = c.realize_frame(new_rep)
        # need to transfer metadata.  TODO! more generally
        new_c.potential = c.potential
        new_c.mass = c.mass

        return new_c

        # return new_pos, units

    # /def


# /class


##############################################################################
# END
