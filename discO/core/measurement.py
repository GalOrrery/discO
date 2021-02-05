# -*- coding: utf-8 -*-

"""Sample a Potential.

Registering a Measurement Sampler
*********************************
a


"""


__all__ = [
    "MeasurementErrorSampler",
    # specific classes
    "GaussianMeasurementErrorSampler",
    # utilities
    "xpercenterror_factory",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T
import warnings
from collections.abc import Mapping
from functools import lru_cache
from types import MappingProxyType

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import (
    BaseCoordinateFrame,
    BaseRepresentation,
    SkyCoord,
)

# PROJECT-SPECIFIC
from .core import PotentialBase
from discO.type_hints import (
    CoordinateType,
    QuantityType,
    RepresentationType,
    SkyCoordType,
)

##############################################################################
# PARAMETERS

MEASURE_REGISTRY: T.Dict[str, PotentialBase] = dict()  # key : measurer

CERR_Type = T.Union[
    T.Callable,
    CoordinateType,
    RepresentationType,
    float,
    np.ndarray,
    T.Mapping,
    QuantityType,
]

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
    method : str or None (optional, keyword-only)
        The method to use for resampling given measurement error.
        Only used if directly instantiating a MeasurementErrorSampler, not a
        subclass.
    return_specific_class : bool (optional, keyword-only)
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

    def __init_subclass__(cls) -> None:
        """Initialize subclass, adding to registry by class name.

        This method applies to all subclasses, not matter the
        inheritance depth, unless the MRO overrides.

        """
        super().__init_subclass__(key=None)

        key = cls.__name__
        if key in cls._registry:
            raise KeyError(f"`{key}` sampler already in registry.")

        MEASURE_REGISTRY[key] = cls

        # TODO? insist that subclasses define a __call__ method
        # this "abstractifies" the base-class even though it can be used
        # as a wrapper class.

    # /def

    #################################################################
    # On the instance

    def __new__(
        cls,
        c_err: T.Optional[CERR_Type] = None,
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
                    "MeasurementErrorSampler has no registered "
                    f"measurement resampler '{method}'",
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
                "Can't specify 'method' on MeasurementErrorSampler subclasses.",
            )

        elif return_specific_class is not False:
            warnings.warn("Ignoring argument `return_specific_class`")

        return self

    # /def

    def __init__(self, c_err: T.Optional[CERR_Type] = None, **kwargs) -> None:
        super().__init__()
        self.c_err = c_err

    # /def

    #################################################################
    # Sampling

    def __call__(
        self,
        c: CoordinateType,
        c_err: T.Optional[CERR_Type] = None,
        *args,
        random: T.Union[int, np.random.Generator, None] = None,
        **kwargs,
    ) -> SkyCoordType:
        """Draw a realization given Measurement error.

        Parameters
        ----------
        c : :class:`~astropy.coordinates.SkyCoord` instance
        c_err : coord-like or callable or |Quantity| or None (optional)
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
        if c_err is None:
            c_err = self.c_err
        # call on instance
        return self._instance(c, c_err, *args, random=random, **kwargs)

    # /def

    def _parse_c_err(
        self,
        c_err: T.Optional[CERR_Type],
        c: CoordinateType,
    ) -> np.ndarray:
        """Parse ``c_err``, given ``c``.

        Parameters
        ----------
        c_err

        """
        nd = c.data.shape[0]
        if c_err is None:
            c_err = self.c_err

        if isinstance(c_err, (BaseCoordinateFrame, SkyCoord)):
            if c_err.representation_type != c.representation_type:
                raise TypeError(
                    "`c` & `c_err` must have matching `representation_type`.",
                )
            # calling ``represent_as`` fixes problems with missing components,
            # even when the representation types match, that happens when
            # something is eg Spherical, but has no distance.
            d_pos = (
                c_err.represent_as(c_err.representation_type)
                ._values.view(dtype=np.float64)
                .reshape(nd, -1)
            )
        elif isinstance(c_err, BaseRepresentation):
            if not isinstance(c_err, c.representation_type):
                raise TypeError(
                    "`c_err` must be the same Representation type as in `c`.",
                )
            d_pos = c_err._values.view(dtype=np.float64).reshape(nd, -1)
        elif isinstance(c_err, Mapping):
            raise NotImplementedError("TODO")
        # special case if has units of percent
        elif getattr(c_err, "unit", u.m) == u.percent:
            d_pos = xpercenterror_factory(c_err)(c)
        # just a number
        elif np.isscalar(c_err):
            d_pos = c_err
        # produce from callable
        elif callable(c_err):
            d_pos = c_err(c)
        else:
            raise NotImplementedError(f"{c_err} is not yet supported.")

        return d_pos

    # /def


# /class

# -------------------------------------------------------------------


class GaussianMeasurementErrorSampler(MeasurementErrorSampler):
    """Draw a realization given Gaussian measurement errors.

    Parameters
    ----------
    c_err : float or callable or None (optional)
        Callable with single mandatory positional argument -- coordinates
        ("c") -- that returns the absolute error.

    """

    def __call__(
        self,
        c: CoordinateType,
        c_err: T.Optional[CERR_Type] = None,
        *,
        random: T.Union[int, np.random.Generator, None] = None,
        representation_type: T.Optional[RepresentationType] = None,
    ) -> SkyCoordType:
        """Draw a realization given the errors.

        .. todo::

            - the velocities
            - confirm that units work nicely
            - figure out phase wrapping when draws over a wrap
            - make calling the function easier when inputting coordinates
            - ensure works on a shaped SkyCoord

        Parameters
        ----------
        c : :class:`~astropy.coordinates.SkyCoord` instance
            The coordinates at which to resample.
        c_err : SkyCoord or None (optional)
            The scale of the Gaussian errors.

        Returns
        -------
        new_c : :class:`~astropy.coordinates.SkyCoord`
            The resampled points.
            Has the same frame, representation_type, and shape and framas `c`.

        Other Parameters
        ----------------
        random : |RandomGenerator| or int or None (optional, keyword-only)
            The random number generator or generator seed.
        representation_type : |Representation| or None (optional, keyword-only)
            The representation type in which to calculate the errors.

        """
        # ----------------
        # Setup

        # see 'default_rng' docs for details
        random = np.random.default_rng(random)

        # get "c" into the correct representation type
        representation_type = representation_type or c.representation_type
        rep = c.data.represent_as(representation_type)
        cc = c.realize_frame(rep)

        # for re-building
        units = rep._units
        nd = rep.shape[0]  # the number of samples

        # ----------------
        # Resample

        # loc & error scale
        pos = rep._values.view(dtype=np.float64).reshape(nd, -1)  # shape=Nx3
        d_pos = self._parse_c_err(c_err, cc)

        # draw realization
        # this will have no units. We will need to add those
        new_pos = random.normal(loc=pos, scale=d_pos, size=pos.shape)

        # deal with wrapping!
        # TODO!

        # re-build representation
        new_rep = rep.__class__(
            **{n: p * unit for p, (n, unit) in zip(new_pos.T, units.items())}
        )

        # make SkyCoord from new realization, preserving shape
        new_c = SkyCoord(c.realize_frame(new_rep).reshape(c.shape))

        # ----------------
        # Cleanup

        # need to transfer metadata.
        # TODO! more generally, probably need different method for new_c
        new_c.potential = getattr(c, "potential", None)
        new_c.mass = getattr(c, "mass", None)

        return new_c

    # /def


# /class


######################################################################
# Utility Functions


@lru_cache()
def xpercenterror_factory(
    fractional_error: float,
) -> T.Callable[[CoordinateType], np.ndarray]:
    r"""Factory-function to build `xpercenterror` function.

    Parameters
    ----------
    fractional_error : float or |Quantity|
        Construct errors with X% error in each dimension
        If Quantity, must have units of percent.

        .. todo::

            Allow a mapping and / or list to apply separately to
            each dimension.

    Returns
    -------
    xpercenterror : callable
        function

    """
    frac_err: float
    if hasattr(fractional_error, "unit"):
        frac_err = fractional_error.to_value(u.dimensionless_unscaled)
    else:
        frac_err = fractional_error

    # X percent error function
    def xpercenterror(c: CoordinateType) -> np.ndarray:
        r"""Construct errors with {X}% error in each dimension.

        This function is made by
        :fun:`~discO.core.measurement.xpercenterror_factory`,
        which takes as argument ``fractional_error``, setting
        the percent-error.

        Parameters
        ----------
        c : coord-like

        Returns
        -------
        c_err : :class:`~numpy.ndarray`
            With {X}% error in each dimension.

        """
        # reshape "c" to Nx3 array
        nd = c.shape[0]  # the number of samples
        vals = c.data._values.view(dtype=np.float64).reshape(nd, -1)

        # get scaled error
        d_pos = np.abs(vals) * frac_err
        return d_pos

    # /def

    # edit the docs
    xpercenterror.__doc__ = xpercenterror.__doc__.format(X=frac_err * 100)

    return xpercenterror


# /def

##############################################################################
# END
