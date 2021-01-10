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
import warnings
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
    """Draw a realization given measurement errors.

    Parameters
    ----------
    c_err : callable or None (optional)
        Callable with single mandatory positional argument -- coordinates
        ("c") -- that returns the absolute error.

    Returns
    -------
    `PotentialSampler` or subclass
        If `return_specific_class` is True, returns subclass.

    Other Parameters
    ----------------
    method : str or None (optional, keyword only)
        The method to use for resampling given measurement error.
        Only used if directly instantiating a MeasurementErrorSampler, not a
        subclass.
    return_specific_class : bool (optional, keyword only)
        Whether to return a `PotentialSampler` (if False, default) or
        package-specific subclass (if True).
        Only used if directly instantiating a MeasurementErrorSampler, not a
        subclass.

    Raises
    ------
    KeyError
        On class declaration if class registry information already exists,
        eg. if registering two classes with the same name.s

    ValueError
        - If directly instantiating a MeasurementErrorSampler (not a subclass)
          and `method` is not in the registry
        - If instantiating a MeasurementErrorSampler subclass and `method` is
          not None.

    """

    #################################################################
    # On the class

    _registry = MappingProxyType(MEASURE_REGISTRY)

    def __init_subclass__(cls):
        """Initialize subclass, adding to registry by class name.

        This method applies to all subclasses, not matter the
        inheritance depth, unless the MRO overrides.

        """
        super().__init_subclass__()

        key = cls.__name__
        if key in cls._registry:
            raise KeyError(f"`{key}` sampler already in registry.")

        MEASURE_REGISTRY[key] = cls

        # TODO? insist that define a __call__ method
        # this "abstractifies" the base-class even though it can be used
        # as a wrapper class.

    # /def

    #################################################################
    # On the instance

    def __new__(
        cls,
        c_err=None,
        *,
        method: T.Optional[str] = None,
        return_specific_class: bool = False,
    ):
        self = super().__new__(cls)

        # The class MeasurementErrorSampler is a wrapper for anything in its
        # registry If directly instantiating a MeasurementErrorSampler (not
        # subclass) we must also instantiate the appropriate subclass. Error
        # if can't find.
        if cls is MeasurementErrorSampler:

            # a cleaner error than KeyError on the actual registry
            if method is None or method not in cls._registry:
                raise ValueError(
                    f"{cls} has no registered measurement resampler '{method}'"
                )

            # from registry. Registered in __init_subclass__
            instance = cls[method](c_err=c_err)

            # Whether to return class or subclass
            # else continue, storing instance
            if return_specific_class:
                return instance

            self._instance = instance

        elif method is not None:
            raise ValueError(
                "Can't specify 'method' on MeasurementErrorSampler subclasses."
            )

        elif return_specific_class is not False:

            warnings.warn("Ignoring argument `return_specific_class`")

        return self

    # /def

    def __init__(self, c_err: T.Optional[callable] = None, **kwargs):
        self.c_err = c_err

    # /def

    #################################################################
    # Calling

    def __call__(
        self, c: FrameLikeType, c_err: FrameLikeType, *args, **kwargs
    ):
        """Draw a realization given Measurement error.

        Parameters
        ----------
        c : :class:`~astropy.coordinates.SkyCoord` instance
        c_err : :class:`~astropy.coordinates.SkyCoord` instance
        *args
            passed to underlying instance
        **kwargs
            passed to underlying instance

        Returns
        -------
        `SkyCoord`

        Notes
        -----
        If this is `MeasurementErrorSampler` then arguments are passed to the
        wrapped instance (see 'method' argument on initialization).

        """
        # call on instance
        return self._instance(c, c_err, *args, **kwargs)

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
        c : :class:`~astropy.coordinates.SkyCoord` instance
        c_err : :class:`~astropy.coordinates.SkyCoord` instance

        random : `~numpy.random.Generator` or int or None
            The random number generator

        """
        # see 'default_rng' docs for details
        random = np.random.default_rng(random)

        nd = len(c.data._values.dtype)  # the shape

        units = c.data._units
        pos = c.data._values.view(dtype=np.float64).reshape(-1, nd)

        if c_err is None:
            c_err = self.c_err

        if isinstance(c_err, (BaseCoordinateFrame, SkyCoord)):
            d_pos = c_err.data._values.view(dtype=np.float64).reshape(-1, nd)
        elif isinstance(c_err, BaseRepresentation):
            raise NotImplementedError("Not yet")
        elif np.isscalar(c_err):
            d_pos = c_err
        elif callable(c_err):
            d_pos = c_err(c)
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
        new_c.potential = getattr(c, "potential", None)
        new_c.mass = getattr(c, "mass", None)

        return new_c

        # return new_pos, units

    # /def


# /class


##############################################################################
# END
