# -*- coding: utf-8 -*-

"""Sample a Potential.

Registering a Measurement Sampler
*********************************
TODO


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
from .common import CommonBase
from .sample import RandomLike  # TODO move to type-hints
from discO.utils import resolve_framelike, resolve_representationlike
from discO.utils.pbar import get_progress_bar

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
        *,
        c_err: T.Optional[CERR_Type] = None,
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
            kls = cls[method]
            return kls.__new__(kls, c_err=c_err, method=None, **kwargs)

        elif method is not None:
            raise ValueError(
                f"Can't specify 'method' on {cls},"
                " only on MeasurementErrorSampler.",
            )

        return super().__new__(cls)

    # /def

    def __init__(
        self,
        *,
        representation_type: TH.OptRepresentationLikeType,
        frame: TH.OptFrameLikeType = None,
        c_err: T.Optional[CERR_Type] = None,
        **kwargs,
    ) -> None:
        # kwargs are ignored
        super().__init__()
        self.c_err = c_err

        # store frame. If not None, resolve it.
        self._frame = resolve_framelike(frame)
        self._representation_type = resolve_representationlike(
            representation_type,
        )

        # params (+ pop from ``__new__``)
        # TODO protect
        self.params = kwargs
        self.params.pop("method", None)

    # /def

    @property
    def frame(self) -> TH.OptFrameLikeType:
        """The frame."""
        return self._frame

    # /def

    @property
    def representation_type(self) -> TH.OptRepresentationLikeType:
        """The representation type."""
        return self._representation_type

    # /def

    #################################################################
    # Sampling

    @abc.abstractmethod
    def __call__(
        self,
        c: TH.CoordinateType,
        c_err: T.Optional[CERR_Type] = None,
        *,
        random: T.Optional[RandomLike] = None,
        **kwargs,
    ) -> TH.SkyCoordType:
        """Draw a realization given Measurement error.

        Parameters
        ----------
        c : :class:`~astropy.coordinates.SkyCoord` instance
        c_err : coord-like or callable or |Quantity| or None (optional)

        representation_type: |Representation| or None (optional, keyword-only)
            The coordinate representation in which to resample along each
            dimension.
        random : `~numpy.random.RandomState` or int (optional, keyword-only)
            The random number generator or generator seed.
            Unfortunately, scipy does not yet support `~numpy.random.Generator`
        **kwargs
            passed to underlying instance

        Returns
        -------
        :class:`~astropy.coordinates.SkyCoord`

        Notes
        -----
        If this is `MeasurementErrorSampler` then arguments are passed to the
        wrapped instance (see 'method' argument on initialization).

        """
        raise NotImplementedError()

    # /def

    # ---------------------------------------------------------------

    def _run_iter(
        self,
        c: TH.SkyCoordType,
        c_err: TH.CoordinateType = None,
        *,
        random: T.Optional[RandomLike] = None,
        progress: bool = True,
        **kwargs,
    ) -> TH.SkyCoordType:
        """Draw a realization given measurement error.

        .. todo::

            - c can be generator object, eg from sampler._run_iter

        Parameters
        ----------
        c : :class:`~astropy.coordinates.SkyCoord` instance
        c_err : :class:`~astropy.coordinates.SkyCoord` instance
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
        progress : bool (optional, keyword-only)
            Whether to show a progress bar

        Notes
        -----
        If this is `MeasurementErrorSampler` then arguments are passed to the
        wrapped instance (see 'method' argument on initialization).

        """
        # need to resolve c_err
        c_err = self.c_err if c_err is None else c_err
        if c_err is None:
            raise ValueError

        N, *iterations = c.shape

        # TODO! fold this into the "with_progress bar" bit
        # depends on the shape of "c": (Nsamples,) or (Nsamples, Niter)?
        if not iterations:  # only (Nsamples, )
            # TODO validate c_err shape etc.
            yield self(
                c,
                c_err=c_err,
                random=random,
                **kwargs,
            )

            return  # prevent going to next thing

        iterations = iterations[0]  # TODO! check shape
        c_errs = self._distribute_c_err(c_err, iterations)

        with get_progress_bar(progress, iterations) as pbar:
            for samp, err in zip(c.T, c_errs):
                pbar.update(1)
                yield self(
                    samp,
                    c_err=err,
                    random=random,
                    **kwargs,
                )

    # /def

    # ---------------------------------------------------------------

    def _run_batch(
        self,
        c: TH.SkyCoordType,
        c_err: TH.CoordinateType = None,
        *,
        random: T.Optional[RandomLike] = None,
        # extra
        progress: bool = False,
        **kwargs,
    ) -> TH.SkyCoordType:
        """Draw a realization given measurement error.

        .. todo::

            - c can be generator object from sampler._run_iter

        Parameters
        ----------
        c : :class:`~astropy.coordinates.SkyCoord` instance
        c_err : :class:`~astropy.coordinates.SkyCoord` instance

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

        # TODO! fold this "if" into the "else" below bit bit
        # depends on the shape of "c": (Nsamples,) or (Nsamples, Niter)?
        if len(c.shape) == 1:  # (Nsamples, )
            # TODO validate c_err shape etc.
            sample = self(
                c,
                c_err=c_err,
                random=random,
                **kwargs,
            )

        else:  # (Nsamples, Niter)

            samples = list(
                self._run_iter(
                    c, c_err=c_err, random=random, progress=progress, **kwargs
                ),
            )

            sample = concatenate(samples).reshape(c.shape)
            # transfer mass & potential # TODO! better
            sample.mass = getattr(c, "mass", None)
            sample.potential = getattr(c, "potential", None)

        # /if

        return sample

    # /def

    def run(
        self,
        c: TH.SkyCoordType,
        c_err: TH.CoordinateType = None,
        *,
        random: T.Optional[RandomLike] = None,
        # extra
        batch: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> TH.SkyCoordType:
        """Draw a realization given measurement error.

        Parameters
        ----------
        c : :class:`~astropy.coordinates.SkyCoord` instance
        c_err : :class:`~astropy.coordinates.SkyCoord` instance

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
        run_func = self._run_batch if batch else self._run_iter

        # need to resolve RandomState
        if not isinstance(random, np.random.RandomState):
            random = np.random.RandomState(random)

        return run_func(
            c, c_err=c_err, random=random, progress=progress, **kwargs
        )

    # /def

    # ===============================================================
    # Utils

    def _distribute_c_err(self, c_err, iterations: int):

        # need to determine how c_err should be distributed.
        if isinstance(
            c_err,
            (SkyCoord, BaseCoordinateFrame, BaseRepresentation),
        ):
            Nerr, *nerriter = c_err.shape
            nerriter = (nerriter or [1])[0]  # [] -> 1

            if nerriter == 1:
                c_err = [c_err] * iterations
            elif nerriter != iterations:
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
            c_err = [c_err] * iterations  # distribute over `niter`

        # IDK what was passed
        else:
            raise NotImplementedError(f"{c_err} is not yet supported.")

        return c_err

    # /def

    def _parse_c_err(
        self,
        c_err: T.Optional[CERR_Type],
        c: TH.CoordinateType,
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

    @staticmethod
    def _fix_branch_cuts(
        array: TH.QuantityType,
        representation_type: TH.OptRepresentationLikeType,
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
        representation_type = resolve_representationlike(representation_type)

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
    rvs : `~scipy.stats.rv_continuous` subclass (keyword-only)
        In the call method this is used to generate points by calling
        ``.rvs()`` with:

            - params set by kwarg
            - ``loc`` from the samples
            - ``scale`` from c_err
            - ``size`` as the samples' shape
            - ``random_state`` from the NumPy random generator.
    c_err : float or None (optional, keyword-only)
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
        *,
        rvs: scipy.stats.rv_continuous,
        c_err: T.Optional[CERR_Type] = None,
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
            # don't pass rvs, b/c not all subclasses take it
            return super().__new__(
                cls[method], c_err=c_err, method=None, **kwargs
            )

        elif method is not None:
            raise ValueError(
                f"Can't specify 'method' on {cls}," " only on RVS_Continuous.",
            )

        # don't pass rvs, b/c not all subclasses take it
        return super().__new__(cls, c_err=c_err, method=None, **kwargs)

    # /def

    def __init__(
        self,
        rvs: T.Callable,  # scipy.stats.rv_continuous
        c_err: T.Optional[CERR_Type] = None,
        *,
        frame: TH.OptFrameLikeType = None,
        representation_type: TH.OptRepresentationLikeType = None,
        **kwargs,
    ) -> None:
        super().__init__(
            c_err=c_err,
            frame=frame,
            representation_type=representation_type,
            **kwargs,
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
        c_err: T.Optional[CERR_Type] = None,
        *,
        random: T.Optional[RandomLike] = None,
        **params,
    ) -> TH.SkyCoordType:
        """Draw a realization given the errors.

        .. todo::

            - the velocities
            - make work on a shaped SkyCoord
            - fold this machinery into parent

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

        """
        # ----------------
        # Setup

        # set as RandomState. see 'RandomState' docs for details
        if not isinstance(random, np.random.RandomState):
            random = np.random.RandomState(random)

        # the rvs parameters
        ps = copy.deepcopy(self.params)
        ps.update(**params)

        # get "c" into the correct frame
        cc = c.transform_to(self.frame)

        # get "cc" into the correct representation type
        rep = cc.data.represent_as(self.representation_type)
        cc = cc.realize_frame(
            rep,
            representation_type=self.representation_type,
        )

        # for re-building
        units = rep._units
        attr_classes = rep.attr_classes

        # ----------------
        # Resample

        # loc, must be ndarray (N, 3)
        pos = rep._values.view(dtype=np.float64).reshape(rep.shape[0], -1)
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
        # make coordinate
        new_cc = self.frame.realize_frame(
            new_rep,
            representation_type=self.representation_type,
        )
        # make SkyCoord from new realization, preserving original shape
        new_sc = SkyCoord(
            new_cc.reshape(c.shape),
            copy=False,
        )

        # need to transfer metadata.
        # TODO! more generally, probably need different method for new_c
        new_sc.potential = getattr(c, "potential", None)
        new_sc.mass = getattr(c, "mass", None)

        return new_sc

    # /def


# /class

# -------------------------------------------------------------------


class GaussianMeasurementError(RVS_Continuous, method="Gaussian"):
    """Draw a realization given Gaussian measurement errors.

    Parameters
    ----------
    rvs : callable (optional, keyword-only)
    c_err : float or callable or None (optional, keyword-only)
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
        *,
        c_err: T.Optional[CERR_Type] = None,
        method: T.Optional[str] = None,
        **kwargs,
    ):
        kwargs.pop("rvs", None)  # clear so no duplicate args
        return super().__new__(
            cls,
            rvs=scipy.stats.norm,
            c_err=c_err,
            method=method,
            **kwargs,  # distribution
        )

    # /def

    def __init__(
        self,
        c_err: T.Optional[CERR_Type] = None,
        *,
        representation_type: TH.OptRepresentationLikeType,
        frame: TH.OptFrameLikeType = None,
        **params,
    ) -> None:
        params.pop("rvs", None)  # clear so no duplicate args
        super().__init__(
            rvs=scipy.stats.norm,
            c_err=c_err,
            frame=frame,
            representation_type=representation_type,
            **params,
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
