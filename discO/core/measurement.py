# -*- coding: utf-8 -*-

"""Sample a Potential.

Registering a Measurement Sampler
*********************************
a


"""


__all__ = [
    "MeasurementErrorSampler",
    # specific classes
    "RVS_Continuous",
    "GaussianMeasurementError",
    # utilities
    "xpercenterror_factory",
]


##############################################################################
# IMPORTS

# BUILT-IN
import abc
import copy
import inspect
import typing as T
from collections.abc import Mapping
from functools import lru_cache
from types import MappingProxyType

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import scipy.stats
from astropy.coordinates import (
    BaseCoordinateFrame,
    BaseRepresentation,
    SkyCoord,
    concatenate,
)
from astropy.utils.decorators import classproperty

# PROJECT-SPECIFIC
import discO.type_hints as TH
from .core import CommonBase
from .sample import RandomLike  # TODO move to type-hints
from discO.utils import resolve_framelike, resolve_representationlike

##############################################################################
# PARAMETERS

_MEASURE_REGISTRY: T.Dict[str, CommonBase] = dict()  # key : measurer

CERR_Type = T.Union[
    T.Callable,
    TH.CoordinateType,
    TH.RepresentationType,
    float,
    np.ndarray,
    T.Mapping,
    TH.QuantityType,
]

##############################################################################
# CODE
##############################################################################


class MeasurementErrorSampler(CommonBase, metaclass=abc.ABCMeta):
    """Draw a realization given measurement errors.

    Parameters
    ----------
    c_err : callable or None (optional)
        Callable with single mandatory positional argument -- coordinates
        ("c") -- that returns the absolute error.

    frame: frame-like or None (optional, keyword-only)
       The frame of the observational errors, ie the frame in which
        the error function should be applied along each dimension.
    representation_type: |Representation| or None (optional, keyword-only)
        The coordinate representation in which to resample along each
        dimension.

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

    _registry = MappingProxyType(_MEASURE_REGISTRY)

    def __init_subclass__(cls, method=None) -> None:
        """Initialize subclass, adding to registry by class name.

        This method applies to all subclasses, not matter the
        inheritance depth, unless the MRO overrides.

        """
        super().__init_subclass__(key=None)  # let's do it ourselves

        method = cls.__name__.lower() if method is None else method
        if method in cls._registry:
            raise KeyError(f"`{method}` sampler already in registry.")

        _MEASURE_REGISTRY[method] = cls
        cls._key = method

        # TODO? insist that subclasses define a __call__ method
        # this "abstractifies" the base-class even though it can be used
        # as a wrapper class.

    # /def

    @classproperty
    def _key(self):
        """The key. Overwritten in subclasses."""
        return None

    # /def

    #################################################################
    # On the instance

    def __new__(
        cls,
        c_err: T.Optional[CERR_Type] = None,
        *,
        method: T.Optional[str] = None,
        **kwargs,
    ):
        # The class MeasurementErrorSampler is a wrapper for anything in its
        # registry If directly instantiating a MeasurementErrorSampler (not
        # subclass) we must also instantiate the appropriate subclass. Error
        # if can't find.
        if cls is MeasurementErrorSampler:

            # a cleaner error than KeyError on the actual registry
            if method is None or not cls._in_registry(method):
                raise ValueError(
                    "MeasurementErrorSampler has no registered "
                    f"measurement resampler '{method}'",
                )

            # from registry. Registered in __init_subclass__
            return super().__new__(cls[method])

        elif method is not None:
            raise ValueError(
                f"Can't specify 'method' on {cls},"
                " only on MeasurementErrorSampler.",
            )

        return super().__new__(cls)

    # /def

    def __init__(
        self,
        c_err: T.Optional[CERR_Type] = None,
        *,
        frame: T.Optional[TH.FrameLikeType] = None,
        representation_type: T.Union[TH.RepresentationLikeType, None] = None,
        **kwargs,
    ) -> None:
        # kwargs are ignored
        super().__init__()
        self.c_err = c_err

        # store frame. If not None, resolve it.
        self.frame = resolve_framelike(frame) if frame is not None else None
        self.representation_type = (
            resolve_representationlike(representation_type)
            if representation_type is not None
            else None
        )

        # params (+ pop from ``__new__``)
        self.params = kwargs
        self.params.pop("method", None)

    # /def

    #################################################################
    # Sampling

    @abc.abstractmethod
    def __call__(
        self,
        c: TH.CoordinateType,
        c_err: T.Optional[CERR_Type] = None,
        *,
        frame: T.Optional[TH.FrameLikeType] = None,
        representation_type: T.Optional[TH.RepresentationType] = None,
        random: T.Optional[RandomLike] = None,
        **kwargs,
    ) -> TH.SkyCoordType:
        """Draw a realization given Measurement error.

        Parameters
        ----------
        c : :class:`~astropy.coordinates.SkyCoord` instance
        c_err : coord-like or callable or |Quantity| or None (optional)

        frame: frame-like or None (optional, keyword-only)
           The frame of the observational errors, ie the frame in which
            the error function should be applied along each dimension.
        representation_type: |Representation| or None (optional, keyword-only)
            The coordinate representation in which to resample along each
            dimension.

        **kwargs
            passed to underlying instance

        Returns
        -------
        :class:`~astropy.coordinates.SkyCoord`

        Other Parameters
        ----------------
        random : `~numpy.random.RandomState` or int (optional, keyword-only)
            The random number generator or generator seed.
            Unfortunately, scipy does not yet support `~numpy.random.Generator`

        Notes
        -----
        If this is `MeasurementErrorSampler` then arguments are passed to the
        wrapped instance (see 'method' argument on initialization).

        """
        raise NotImplementedError()

    # /def

    def resample(
        self,
        c: TH.SkyCoordType,
        c_err: TH.CoordinateType = None,
        *,
        frame: T.Optional[TH.FrameLikeType] = None,
        representation_type: T.Optional[TH.RepresentationType] = None,
        random: T.Optional[RandomLike] = None,
        **kwargs,
    ) -> TH.SkyCoordType:
        """Draw a realization given measurement error.

        .. todo::

            resolve `random` here

        Parameters
        ----------
        c : :class:`~astropy.coordinates.SkyCoord` instance
        c_err : :class:`~astropy.coordinates.SkyCoord` instance

        frame: frame-like or None (optional, keyword-only)
           The frame of the observational errors, ie the frame in which
            the error function should be applied along each dimension.
        representation_type: |Representation| or None (optional, keyword-only)
            The coordinate representation in which to resample along each
            dimension.

        **kwargs
            passed to ``__call__``

        Returns
        -------
        :class:`~astropy.coordinates.SkyCoord`

        Other Parameters
        ----------------
        random : `~numpy.random.RandomState` or int (optional, keyword-only)
            The random number generator or generator seed.
            Unfortunately, scipy does not yet support `~numpy.random.Generator`

        Notes
        -----
        If this is `MeasurementErrorSampler` then arguments are passed to the
        wrapped instance (see 'method' argument on initialization).

        """
        if c_err is None:
            c_err = self.c_err

        # depends on the shape of "c": (Nsamples,) or (Nsamples, Niter)?
        if len(c.shape) == 1:  # (Nsamples, )
            # TODO validate c_err shape etc.
            sample = self(
                c,
                c_err=c_err,
                frame=frame,
                representation_type=representation_type,
                random=random,
                **kwargs,
            )

        else:  # (Nsamples, Niter)

            N, niter = c.shape

            # need to determine how c_err should be distributed.
            if isinstance(
                c_err, (SkyCoord, BaseCoordinateFrame, BaseRepresentation),
            ):
                Nerr, *nerriter = c_err.shape
                nerriter = (nerriter or [1])[0]  # [] -> 1

                if nerriter == 1:
                    c_err = [c_err] * niter
                elif nerriter != niter:
                    raise ValueError("c & c_err shape mismatch")
                else:
                    c_err = c_err.T  # get in correct shape for iteration

            # mapping, or number, or callable
            # special case if has units of percent
            # all other units stuff will hit the error
            elif (
                isinstance(c_err, Mapping)
                or np.isscalar(c_err)
                or callable(c_err)
                or (getattr(c_err, "unit", u.m) == u.percent)
            ):
                c_err = [c_err] * niter  # distribute over `niter`

            # IDK what was passed
            else:
                raise NotImplementedError(f"{c_err} is not yet supported.")

            # /if

            samples = []
            for samp, err in zip(c.T, c_err):
                result = self(
                    samp,
                    c_err=err,
                    frame=frame,
                    representation_type=representation_type,
                    random=random,
                    **kwargs,
                )
                samples.append(result)

            sample = concatenate(samples).reshape(c.shape)

        return sample

    # /def

    def _parse_c_err(
        self, c_err: T.Optional[CERR_Type], c: TH.CoordinateType,
    ) -> np.ndarray:
        """Parse ``c_err``, given ``c``.

        Parameters
        ----------
        c_err : coord-like or callable or |Quantity| or None (optional)
        c : |CoordinateFrame| or |SkyCoord|

        Returns
        -------
        :class:`~numpy.ndarray`

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

    def _resolve_frame(
        self, frame: T.Optional[TH.RepresentationType], c: TH.SkyCoordType,
    ) -> TH.FrameType:
        """Resolve, given coordinate and passed value.

        .. todo::

            rename ``_preferred_frame_resolve``

        Uses :fun:`~discO.utils.resolve_framelike` for strings.

        Priority:
        1. frame, if not None
        2. frame set at initialization, if not None
        3. c.frame

        Returns
        -------
        |CoordinateFrame|

        """
        # prefer frame, if not None
        frame = frame if frame is not None else self.frame

        # prefer frame, if not None
        frame = frame if frame is not None else c.frame

        return resolve_framelike(frame)

    # /def

    def _resolve_representation_type(
        self,
        representation_type: T.Optional[TH.RepresentationLikeType],
        c: TH.SkyCoordType,
    ) -> TH.RepresentationType:
        """Resolve, given coordinate and passed value.

        Priority:
        1. representation_type, if not None
        2. representation_type set at initialization, if not None
        3. c.representation_type

        Returns
        -------
        :class:`~astropy.coordinates.Representation` class

        """
        # prefer representation_type, if not None
        rep_type = (
            representation_type
            if representation_type is not None
            else self.representation_type
        )

        # prefer representation_type, if not None
        rep_type = rep_type if rep_type is not None else c.representation_type

        return resolve_representationlike(rep_type)

    # /def

    @staticmethod
    def _fix_branch_cuts(
        array: TH.QuantityType,
        representation_type: TH.RepresentationType,
        units: T.Dict[str, TH.UnitType],
    ) -> TH.QuantityType:
        """Fix Branch Cuts.

        .. todo::

            In general w/out if statement for each rep type

        Parameters
        ----------
        array : (3, N) array |Quantity|
        representation_type : |Representation| class
        units : dict

        Returns
        -------
        array : |Quantity|

        """
        # First check if any of the components are angular
        if not any([v.physical_type == "angle" for v in units.values()]):
            return array

        elif representation_type is coord.UnitSphericalRepresentation:
            # longitude is not a problem, but latitude is restricted
            # to be between -90 and 90 degrees
            bound = 90 * u.deg.to(units["lat"])  # convert deg to X
            bad_lat_i = (array[1] < -bound) | (bound < array[1])
            bad_lat = array[1, bad_lat_i]

            # mirror the bad lats about the pole
            # and rotate the lons by 180 degrees
            array[1, bad_lat_i] = bound - np.mod(bad_lat + bound, 2 * bound)
            array[0, bad_lat_i] = array[0, bad_lat_i] + 180 * u.deg.to(
                units["lon"],
            )

        elif representation_type is coord.SphericalRepresentation:
            # longitude is not a problem, but latitude is restricted
            # to be between -90 and 90 degrees
            bound = 90 * u.deg.to(units["lat"])  # convert deg to X
            bad_lat_i = (array[1] < -bound) | (bound < array[1])
            bad_lat = array[1, bad_lat_i]

            # mirror the bad lats about the pole
            # and rotate the lons by 180 degrees
            array[1, bad_lat_i] = bound - np.mod(bad_lat + bound, 2 * bound)
            array[0, bad_lat_i] = array[0, bad_lat_i] + 180 * u.deg.to(
                units["lon"],
            )

            # the distance can also be problematic if less than 0
            # lat -> -lat,
            bad_d_i = array[2] < 0

            array[2, bad_d_i] = -array[2, bad_d_i]  # positive
            array[0, bad_d_i] = array[0, bad_d_i] + 180 * u.deg.to(
                units["lon"],  # + 180
            )
            array[1, bad_d_i] = -array[1, bad_d_i]  # flip

        elif representation_type is coord.CylindricalRepresentation:
            # the distance can also be problematic if less than 0
            # phi -> -phi,
            bad_rho_ind = array[0] < 0  # rho

            array[0, bad_rho_ind] = -array[0, bad_rho_ind]  # positive
            array[1, bad_rho_ind] = array[1, bad_rho_ind] + 180 * u.deg.to(
                units["phi"],
            )

        else:
            raise NotImplementedError(
                f"{representation_type} is not yet supported.",
            )

        return array

    # /def


# /class

# -------------------------------------------------------------------


class RVS_Continuous(MeasurementErrorSampler, method="rvs"):
    """Draw a realization given a :class:`scipy.stats.rv_continuous`.

    Parameters
    ----------
    rvs : `~scipy.stats.rv_continuous` subclass instance
        In the call method this is used to generate points by calling
        ``.rvs()`` with:

            - params set by kwarg
            - ``loc`` from the samples
            - ``scale`` from c_err
            - ``size`` as the samples' shape
            - ``random_state`` from the NumPy random generator.
    c_err : float or None (optional)
        The absolute error.

    **kwargs
        Stored as ``params``.

    Other Parameters
    ----------------
    method : str or None (optional, keyword-only)
        The method to use for resampling given measurement error.
        Only used if directly instantiating a MeasurementErrorSampler, not a
        subclass.

    """

    def __new__(
        cls,
        rvs: scipy.stats.rv_continuous,
        c_err: T.Optional[CERR_Type] = None,
        *,
        method: T.Optional[str] = None,
        **kwargs,
    ):
        # The class RVS_Continuous is a wrapper for anything in its
        # registry If directly instantiating a RVS_Continuous (not
        # subclass) we must also instantiate the appropriate subclass. Error
        # if can't find.
        if cls is RVS_Continuous and method is not None:

            # a cleaner error than KeyError on the actual registry
            if not cls._in_registry(method):
                raise ValueError(
                    "RVS_Continuous has no registered "
                    f"measurement resampler '{method}'",
                )

            # from registry. Registered in __init_subclass__
            return super().__new__(
                cls[method], c_err=c_err, method=None, **kwargs
            )

        elif method is not None:
            raise ValueError(
                f"Can't specify 'method' on {cls}," " only on RVS_Continuous.",
            )

        return super().__new__(cls, c_err, method=None, **kwargs)

    # /def

    def __init__(
        self,
        rvs: T.Callable,  # scipy.stats.rv_continuous
        c_err: T.Optional[CERR_Type] = None,
        *,
        frame: T.Optional[TH.FrameLikeType] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            c_err=c_err, frame=frame, **kwargs,
        )
        self._rvs = rvs
        self._rvs_sig = inspect.signature(rvs.rvs)

    # /def

    @property
    def rvs(self) -> scipy.stats.rv_continuous:
        """The random-variate sampler (rvs)."""
        return self._rvs

    # /def

    def __call__(
        self,
        c: TH.CoordinateType,
        c_err: T.Union[
            # T.Callable,  # not callable
            TH.CoordinateType,
            TH.RepresentationType,
            float,
            np.ndarray,
            T.Mapping,
            TH.QuantityType,
            None,
        ] = None,
        *,
        frame: T.Optional[TH.FrameLikeType] = None,
        representation_type: T.Optional[TH.RepresentationType] = None,
        random: T.Optional[RandomLike] = None,
        **params,
    ) -> TH.SkyCoordType:
        """Draw a realization given the errors.

        .. todo::

            - the velocities
            - make work on a shaped SkyCoord

        Steps:

            1. transforms ``c`` to frame and representation type.
            2. constructs RVS parameters, in particular ``scale``.
            3. (re)samples, on the representation array.
            4. reverse constructs the new ``c`` in the original frame and
               representation type.

        Parameters
        ----------
        c : :class:`~astropy.coordinates.SkyCoord` instance
            The coordinates at which to resample.
        c_err : SkyCoord or None (optional)
            The scale of the errors.
            Must be in the correct frame and representation type.

            ``d_pos`` is created from ``c_err``. It sets ``scale``.

        frame : frame-like or None (optional, keyword only)
            The frame of the observational errors, ie the frame in which
            the error function should be applied.
        representation_type : |Representation| or None (optional, keyword only)
            The representation type of the observational errors,
            ie the coordinates in which the error function should be applied.

        **params
            Parameters into the RVS. Scipy normally does these as arguments,
            but it also works as kwargs.

        Returns
        -------
        new_c : :class:`~astropy.coordinates.SkyCoord`
            The resampled points.
            Has the same frame, representation_type, and shape and framas `c`.

        Other Parameters
        ----------------
        random : `~numpy.random.RandomState` or int (optional, keyword-only)
            The random number generator or generator seed.
            Unfortunately, scipy does not yet support `~numpy.random.Generator`
        representation_type : |Representation| or None (optional, keyword-only)
            The representation type in which to calculate the errors.

        """
        # ----------------
        # Setup

        # pop from params, and set as RandomState
        # see 'RandomState' docs for details
        _random = self.params.pop("random", None)
        random = _random if random is None else random
        if not isinstance(random, np.random.RandomState):
            random = np.random.RandomState(random)

        # the rvs parameters
        ps = copy.deepcopy(self.params)
        ps.update(**params)

        # get "c" into the correct frame
        frame = self._resolve_frame(frame, c)
        cc = c.transform_to(frame)

        # get "cc" into the correct representation type
        representation_type = self._resolve_representation_type(
            representation_type, cc,
        )
        rep = cc.data.represent_as(representation_type)
        cc = cc.realize_frame(rep, representation_type=representation_type)

        # for re-building
        units = rep._units
        attr_classes = rep.attr_classes

        # ----------------
        # Resample

        # loc, must be ndarray
        pos = rep._values.view(dtype=np.float64).reshape(
            rep.shape[0], -1,
        )  # shape=Nx3
        # scale, from `c_err`
        scale = self._parse_c_err(c_err, cc)

        # draw realization
        # this will have no units. We will need to add those
        ba = self._rvs_sig.bind_partial(
            **ps, loc=pos, scale=scale, size=pos.shape, random_state=random
        )
        ba.apply_defaults()

        new_pos = self.rvs.rvs(*ba.args, **ba.kwargs)

        # deal with branch cuts
        new_posT = self._fix_branch_cuts(new_pos.T, rep.__class__, units)

        # ----------------
        # Cleanup

        # re-build representation
        new_rep = rep.__class__(
            **{
                n: attr_classes[n](p * unit)
                for p, (n, unit) in zip(new_posT, units.items())
            }
        )
        # transform back to `c`'s representation type
        new_rep = new_rep.represent_as(c.representation_type)

        # make coordinate
        new_cc = frame.realize_frame(
            new_rep, representation_type=c.representation_type,
        )

        # make SkyCoord from new realization, preserving original shape
        new_c = SkyCoord(
            new_cc.transform_to(c.frame).reshape(c.shape), copy=False,
        )

        # need to transfer metadata.
        # TODO! more generally, probably need different method for new_c
        new_c.potential = getattr(c, "potential", None)
        new_c.mass = getattr(c, "mass", None)

        return new_c

    # /def


# /class

# -------------------------------------------------------------------


class GaussianMeasurementError(RVS_Continuous, method="Gaussian"):
    """Draw a realization given Gaussian measurement errors.

    Parameters
    ----------
    rvs : callable (optional)
    c_err : float or callable or None (optional)
        Callable with single mandatory positional argument -- coordinates
        ("c") -- that returns the absolute error.

    frame: frame-like or None (optional, keyword-only)
       The frame of the observational errors, ie the frame in which
        the error function should be applied along each dimension.
    representation_type: |Representation| or None (optional, keyword-only)
        The coordinate representation in which to resample along each
        dimension.

    **params
        Stored as ``params``.

    Raises
    ------
    ValueError
        If "norm" not in the name in `rvs`.

    """

    def __new__(
        cls,
        rvs: T.Callable = scipy.stats.norm,
        c_err: T.Optional[CERR_Type] = None,
        *,
        method: T.Optional[str] = None,
        **kwargs,
    ):
        return super().__new__(
            cls, rvs, c_err=c_err, method=method, **kwargs,  # distribution
        )

    # /def

    def __init__(
        self,
        rvs: T.Callable = scipy.stats.norm,
        c_err: T.Optional[CERR_Type] = None,
        *,
        frame: T.Optional[TH.FrameLikeType] = None,
        **params,
    ) -> None:

        # TODO better validation that it's a "normal"
        if "norm" not in rvs.__class__.__name__:
            raise ValueError("rvs must be a Normal type.")

        super().__init__(
            rvs, c_err=c_err, frame=frame, **params,
        )

    # /def


# /class


######################################################################
# Utility Functions


@lru_cache()
def xpercenterror_factory(
    fractional_error: float,
) -> T.Callable[[TH.CoordinateType], np.ndarray]:
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
    def xpercenterror(c: TH.CoordinateType) -> np.ndarray:
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
