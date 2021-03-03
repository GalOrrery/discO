# -*- coding: utf-8 -*-

"""Residuals.

.. todo::

    enforce that the frames are the same between the original and fit
    potential.

"""


__all__ = [
    "ResidualMethod",
    "GridResidual",
]


##############################################################################
# IMPORTS

# BUILT-IN
import abc
from collections.abc import Sequence
import typing as T
from types import MappingProxyType, ModuleType

# THIRD PARTY
import astropy.coordinates as coord
from discO.utils.pbar import get_progress_bar
import numpy as np

# PROJECT-SPECIFIC
import discO.type_hints as TH
from .common import CommonBase
from .wrapper import PotentialWrapper
from discO.utils.coordinates import (
    resolve_framelike,
    resolve_representationlike,
)

##############################################################################
# PARAMETERS

RESIDUAL_REGISTRY: T.Dict[str, object] = dict()  # key : sampler

##############################################################################
# CODE
##############################################################################


class ResidualMethod(CommonBase):
    """Calculate Residual.

    Parameters
    ----------
    original_potential : object or :class:`~discO.core.PotentialWrapper`
        The original potential.
        In order to evaluate on a grid with a frame, should be a
        :class:`~discO.core.wrapper.PotentialWrapper`.

    observable : str (optional)
        The quantity on which to calculate the residual.
        Must be method of :class:`~discO.core.wrapper.PotentialWrapper`.

    representation_type : representation-resolvable (optional, keyword-only)
        The output representation type.
        If None, resolves to ``PotentialWrapper.base_representation``

    """

    #################################################################
    # On the class

    _registry = RESIDUAL_REGISTRY

    def __init_subclass__(cls, key: T.Union[str, ModuleType, None] = None):
        """Initialize subclass, adding to registry by `package`.

        This method applies to all subclasses, not matter the
        inheritance depth, unless the MRO overrides.

        """
        super().__init_subclass__(key=key)

        if key is not None:  # same trigger as CommonBase
            # cls._package defined in super()
            cls.__bases__[0]._registry[cls._key] = cls
        else:  # key is None
            cls._key = None

        # TODO? insist that subclasses define a evaluate method
        # this "abstractifies" the base-class even though it can be used
        # as a wrapper class.

    # /def

    def __new__(cls, *args, method: T.Optional[str] = None, **kwargs):
        # The class ResidualMethod is a wrapper for anything in its registry
        # If directly instantiating a ResidualMethod (not subclass) we must
        # instead instantiate the appropriate subclass. Error if can't find.
        if cls is ResidualMethod:

            # a cleaner error than KeyError on the actual registry
            if method is None or not cls._in_registry(method):
                raise ValueError(
                    "ResidualMethod has no registered " f"method '{method}'",
                )

            # from registry. Registered in __init_subclass__
            kls = cls[method]
            return kls.__new__(kls, method=None, **kwargs)

        elif method is not None:
            raise ValueError(
                f"Can't specify 'method' on {cls.__name__}, "
                "only on ResidualMethod.",
            )

        return super().__new__(cls)

    # /def

    #################################################################
    # On the instance

    def __init__(
        self,
        original_potential: T.Optional[T.Any] = None,
        observable: str = "acceleration",  # TODO make None and have config
        *,
        representation_type: TH.OptRepresentationLikeType = None,
        **kwargs,
    ) -> None:
        kwargs.pop("method", None)  # pop from ``__new__``.

        self._observable: str = observable
        self._default_params: T.Dict[str, T.Any] = kwargs

        # representation type
        representation_type = (
            resolve_representationlike(representation_type)
            if not (
                representation_type is None or representation_type is Ellipsis
            )
            else representation_type
        )

        self._original_potential = PotentialWrapper(
            original_potential,
            # frame=frame,
            representation_type=representation_type,
        )

    # /def

    # ---------------------------------------------------------------

    @property
    def observable(self) -> str:
        """Observable."""
        return self._observable

    # /def

    @property
    def default_params(self) -> MappingProxyType:
        """Default parameters."""
        return MappingProxyType(self._default_params)

    # /def

    @property
    def original_potential(self) -> PotentialWrapper:
        """Original potential."""
        return self._original_potential

    # /def

    @property
    def frame(self) -> TH.OptFrameType:
        """Representation Type."""
        return self.original_potential.frame

    # /def

    @property
    def representation_type(self) -> TH.OptRepresentationLikeType:
        """Representation Type."""
        return self.original_potential.representation_type

    # /def

    #################################################################
    # evaluate

    @abc.abstractmethod
    def evaluate_potential(
        self,
        potential: T.Union[PotentialWrapper, T.Any],
        observable: T.Optional[str] = None,
        *,
        representation_type: TH.OptRepresentationLikeType = None,
        **kwargs,
    ) -> object:
        """Evaluate method on potential.

        Parameters
        ----------
        potential : object or :class:`~PotentialWrapper`
            The potential. If not already a PotentialWrapper, it is wrapped:
            this means that the frame is :class:`~discO.UnFrame`.
        observable : str or None (optional)
            name of method in :class:`~PotentialWrapper`.
            If None (default), uses default value -- ``.observable``.
        representation_type: representation-resolvable (optional, keyword-only)
            The output representation type. If None (default), uses default
            representation point.
        **kwargs
            Passed to method in :class:`~PotentialWrapper`.
            First mixed in with ``default_params`` (preferring ``kwargs``).

        Returns
        -------
        object

        Other Parameters
        ----------------
        points : frame-like or (optional, keyword-only)
            Not recommended, but allowed, to force the points on which the
            residual is calculated. The points can only be frame-like if the
            frame of the potentials is not :class:`~discO.UnFrame`.
            If not specified, and it shouldn't be, uses points determined
            by the class at initialization.

        """
        # -----------------------
        # Setup

        # observable.
        # None -> default. Error if default also None.
        observable = observable or self.observable  # None -> stored
        if observable is None:  # still None
            raise ValueError("Need to pass observable.")

        # points, default to default
        points = kwargs.pop("points", self.points)

        # representation type, resolve it here
        representation_type = (
            resolve_representationlike(representation_type)
            if representation_type is not None
            else self.representation_type
        )

        # # change the points to the potential`s reference frame
        # points = points.transform_to(potential.frame)

        # -----------------------
        # Evaluate

        # get class to evaluate
        evaluator_cls = PotentialWrapper(
            potential,
            # frame=frame,  # inherit
            representation_type=representation_type,
        )

        # get method from evaluator class
        evaluator: T.Callable = getattr(evaluator_cls, observable)

        # evaluate
        value = evaluator(
            points, representation_type=representation_type, **kwargs
        )

        # -----------------------
        # Return

        if representation_type is None:
            representation_type = value.base_representation
        representation_type = resolve_representationlike(representation_type)

        return value.represent_as(representation_type)

    # /def

    # -----------------------------------------------------

    def __call__(
        self,
        fit_potential: T.Any,
        original_potential: T.Optional[T.Any] = None,
        observable: T.Optional[str] = None,
        *,
        representation_type: TH.OptRepresentationLikeType = None,
        **kwargs,
    ) -> object:
        """Calculate Residual.

        Parameters
        ----------
        fit_potential : object or :class:`~PotentialWrapper`
            The fitted potential. If not already a PotentialWrapper, it is
            wrapped: this means that the frame is :class:`~discO.UnFrame`.
        original_potential : object or :class:`~PotentialWrapper` (optional)
            The original potential. If not already a PotentialWrapper, it is
            wrapped: this means that the frame is :class:`~discO.UnFrame`.

        observable : str or None (optional)
            The quantity on which to calculate the residual.
            Must be method of :class:`~discO.core.wrapper.PotentialWrapper`.
            None (default) becomes the default value at initialization.

        representation_type: representation-resolvable (optional, keyword-only)
            The output representation type. If None (default), uses default
            representation point.

        **kwargs
            Passed to method in :class:`~PotentialWrapper`.
            First mixed in with ``default_params`` (preferring ``kwargs``).
            See Other Parameters for more details.

        Returns
        -------
        residual : object
            In `representation_type`.

        Other Parameters
        ----------------
        points : frame-like or |Representation| (optional, keyword-only)
            Not recommended, but allowed, to force the points on which the
            residual is calculated. The points can only be frame-like if the
            frame of the potentials is not :class:`~discO.UnFrame`.
            If not specified, and it shouldn't be, uses points determined
            by the class at initialization.

        """
        # -----------------------
        # Setup

        # kwargs, mix in defaults, overriding with passed.
        kw = dict(self.default_params.items())
        kw.update(kwargs)

        # observable.
        # None -> default. Error if default also None.
        observable = observable or self.observable
        if observable is None:  # TODO get from config
            raise ValueError("`observable` not set. Need to pass.")

        # representation type
        # None -> default. Everything is resolved here.
        # both potentials will use this representation type
        # Note that this can still be None, which is we we need one more check
        # at the end
        representation_type = (
            resolve_representationlike(representation_type)
            if representation_type is not None
            else self.representation_type
        )

        # potential
        original_potential = original_potential or self.original_potential
        if original_potential is None:  # both passed and init are None
            raise ValueError("`original_potential` not set. Need to pass.")

        original_potential = PotentialWrapper(
            original_potential,
            # frame=frame,  # NOT SET ON PURPOSE
            representation_type=representation_type,
        )
        # now deal with Ellipsis to be threadsafe. resolve all frames here.
        if original_potential.frame is Ellipsis:
            original_potential._frame = resolve_framelike(Ellipsis)

        fit_potential = PotentialWrapper(
            fit_potential,
            # frame=frame,  # NOT SET ON PURPOSE
            representation_type=representation_type,
        )

        # -----------------------
        # Validate

        # now we confirm that the frames are the same.
        # we can do this since everything is in a PotentialWrapper
        if fit_potential.frame != original_potential.frame:
            raise ValueError(
                "original and fit potentials must have the same frames.\n"
                f"The original potential has:\n\t{original_potential.frame}\n"
                f"The fit potential has:\n\t{fit_potential.frame}\n",
            )

        # -----------------------
        # Evaluate

        # get value on original potential
        origval = self.evaluate_potential(
            original_potential,
            observable=observable,
            representation_type=coord.CartesianRepresentation,
            **kw,
        )
        # get value on fit potential
        fitval = self.evaluate_potential(
            fit_potential,
            observable=observable,
            representation_type=coord.CartesianRepresentation,
            **kw,
        )

        # get difference
        residual = fitval - origval  # TODO! weighting by errors

        # -----------------------
        # Return

        # output representation type
        # still None -> base_representation
        if representation_type is None:
            representation_type = residual.base_representation
        representation_type = resolve_representationlike(representation_type)

        return residual.represent_as(representation_type)

    # /def

    # -----------------------------------------------------

    def _run_iter(
        self,
        fit_potential: T.Union[PotentialWrapper, T.Sequence[PotentialWrapper]],
        original_potential: T.Optional[T.Any] = None,
        observable: T.Optional[str] = None,
        *,
        representation_type: TH.OptRepresentationLikeType = None,
        # extra
        progress: bool = True,
        **kwargs,
    ) -> object:
        """Calculate Residual.

        Parameters
        ----------
        fit_potential : :class:`~PotentialWrapper` or sequence thereof
            The fitted potential. If not already a PotentialWrapper, it is
            wrapped: this means that the frame is :class:`~discO.UnFrame`.
        original_potential : object or :class:`~PotentialWrapper` (optional)
            The original potential. If not already a PotentialWrapper, it is
            wrapped: this means that the frame is :class:`~discO.UnFrame`.

        observable : str or None (optional)
            The quantity on which to calculate the residual.
            Must be method of :class:`~discO.core.wrapper.PotentialWrapper`.
            None (default) becomes the default value at initialization.

        representation_type: representation-resolvable (optional, keyword-only)
            The output representation type. If None (default), uses default
            representation point.

        **kwargs
            Passed to method in :class:`~PotentialWrapper`.
            First mixed in with ``default_params`` (preferring ``kwargs``).

        Returns
        -------
        residual : object
            In `representation_type`.

        Other Parameters
        ----------------
        points : frame-like or |Representation| (optional, keyword-only)
            Not recommended, but allowed, to force the points on which the
            residual is calculated. The points can only be frame-like if the
            frame of the potentials is not :class:`~discO.UnFrame`.
            If not specified, and it shouldn't be, uses points determined
            by the class at initialization.

        """
        if not isinstance(fit_potential, (Sequence, np.ndarray)):
            fit_potential = [fit_potential]

        iterations = len(fit_potential)

        with get_progress_bar(progress, iterations) as pbar:

            for fpot in fit_potential:
                pbar.update(1)

                yield self(
                    fit_potential=fpot,
                    original_potential=original_potential,
                    observable=observable,
                    representation_type=representation_type,
                    **kwargs,
                )

    # /def

    # -----------------------------------------------------

    def _run_batch(
        self,
        fit_potential: T.Union[PotentialWrapper, T.Sequence[PotentialWrapper]],
        original_potential: T.Optional[T.Any] = None,
        observable: T.Optional[str] = None,
        *,
        representation_type: TH.OptRepresentationLikeType = None,
        # extra
        progress: bool = True,
        **kwargs,
    ) -> object:
        """Calculate Residual.

        Parameters
        ----------
        fit_potential : :class:`~PotentialWrapper` or sequence thereof
            The fitted potential. If not already a PotentialWrapper, it is
            wrapped: this means that the frame is :class:`~discO.UnFrame`.
        original_potential : object or :class:`~PotentialWrapper` (optional)
            The original potential. If not already a PotentialWrapper, it is
            wrapped: this means that the frame is :class:`~discO.UnFrame`.

        observable : str or None (optional)
            The quantity on which to calculate the residual.
            Must be method of :class:`~discO.core.wrapper.PotentialWrapper`.
            None (default) becomes the default value at initialization.

        representation_type: representation-resolvable (optional, keyword-only)
            The output representation type. If None (default), uses default
            representation point.

        **kwargs
            Passed to method in :class:`~PotentialWrapper`.
            First mixed in with ``default_params`` (preferring ``kwargs``).

        Returns
        -------
        residual : object
            In `representation_type`.

        Other Parameters
        ----------------
        points : frame-like or |Representation| (optional, keyword-only)
            Not recommended, but allowed, to force the points on which the
            residual is calculated. The points can only be frame-like if the
            frame of the potentials is not :class:`~discO.UnFrame`.
            If not specified, and it shouldn't be, uses points determined
            by the class at initialization.

        """
        resids = tuple(
            self._run_iter(
                fit_potential,
                original_potential=original_potential,
                observable=observable,
                representation_type=representation_type,
                progress=progress,
                **kwargs,
            )
        )

        if not isinstance(fit_potential, Sequence):
            return resids[0]
        else:
            return np.array(resids, dtype=object)

    # /def

    # -----------------------------------------------------

    def run(
        self,
        fit_potential: T.Union[PotentialWrapper, T.Sequence[PotentialWrapper]],
        original_potential: T.Optional[T.Any] = None,
        observable: T.Optional[str] = None,
        *,
        representation_type: TH.OptRepresentationLikeType = None,
        # extra
        batch: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> object:
        """Calculate Residual.

        Parameters
        ----------
        fit_potential : :class:`~PotentialWrapper` or sequence thereof
            The fitted potential. If not already a PotentialWrapper, it is
            wrapped: this means that the frame is :class:`~discO.UnFrame`.
        original_potential : object or :class:`~PotentialWrapper` (optional)
            The original potential. If not already a PotentialWrapper, it is
            wrapped: this means that the frame is :class:`~discO.UnFrame`.

        observable : str or None (optional)
            The quantity on which to calculate the residual.
            Must be method of :class:`~discO.core.wrapper.PotentialWrapper`.
            None (default) becomes the default value at initialization.

        representation_type: representation-resolvable (optional, keyword-only)
            The output representation type. If None (default), uses default
            representation point.

        **kwargs
            Passed to method in :class:`~PotentialWrapper`.
            First mixed in with ``default_params`` (preferring ``kwargs``).

        Returns
        -------
        residual : object
            In `representation_type`.

        Other Parameters
        ----------------
        points : frame-like or |Representation| (optional, keyword-only)
            Not recommended, but allowed, to force the points on which the
            residual is calculated. The points can only be frame-like if the
            frame of the potentials is not :class:`~discO.UnFrame`.
            If not specified, and it shouldn't be, uses points determined
            by the class at initialization.

        """
        run_func = self._run_batch if batch else self._run_iter

        return run_func(
            fit_potential,
            original_potential=original_potential,
            observable=observable,
            representation_type=representation_type,
            progress=progress,
            **kwargs,
        )

    # /def


# /class

##############################################################################


# class RandomGridResidual(ResidualMethod, key="random"):
#     """Residual at random points.

#     .. todo::

#         allow for distributions

#     Parameters
#     ----------
#     original_potential : PotentialWrapper or object or None (optional)

#     observable : str (optional)

#     Other Parameters
#     ----------------
#     representation_type : representation-resolvable (optional, keyword-only)
#         The output representation type.
#         If None, resolves to ``PotentialWrapper.base_representation``
#     **kwargs
#         default arguments into ``evaluate_potential``.

#     """

#     #################################################################
#     # On the instance

#     def __init__(
#         self,
#         grid: TH.RepresentationType,
#         original_potential: T.Union[PotentialWrapper, T.Any, None] = None,
#         observable: str = "acceleration",  # TODO make None and have config
#         *,
#         representation_type: TH.OptRepresentationLikeType = None,
#         **kwargs,
#     ) -> None:
#         # the points
#         self.points = grid
#         # initialize
#         super().__init__(
#             original_potential=original_potential,
#             observable=observable,
#             representation_type=representation_type,
#             **kwargs,
#         )

#     # /def

#     #################################################################
#     # evaluate

#     def evaluate_potential(
#         self,
#         potential: T.Union[PotentialWrapper, T.Any],
#         observable: T.Optional[str] = None,
#         points: T.Optional[TH.RepresentationType] = None,
#         *,
#         representation_type: TH.OptRepresentationLikeType = None,
#         **kwargs,
#     ) -> object:
#         """Evaluate method on potential.

#         Parameters
#         ----------
#         potential : object
#         observable : str
#             method in :class:`~PotentialWrapper`
#         points : `~astropy.coordinates.BaseRepresentation` or None (optional)
#             The points of the grid
#         **kwargs
#             Passed to method on :class:`~PotentialWrapper`

#         Returns
#         -------
#         object

#         """
#         observable = observable or self.observable  # None -> stored
#         if observable is None:  # still None
#             raise ValueError("Need to pass observable.")

#         if points is None:  # default to default
#             points = self.points

#         # get class to evaluate
#         evaluator_cls = PotentialWrapper(
#             potential,
#             # frame=frame,
#             representation_type=representation_type,
#         )

#         # get method from evaluator class
#         evaluator = getattr(evaluator_cls, observable)

#         # evaluate
#         value = evaluator(
#             points, representation_type=representation_type, **kwargs
#         )

#         # output representation type
#         if representation_type is None:
#             representation_type = value.base_representation
#         representation_type = resolve_representationlike(representation_type)

#         return value.represent_as(representation_type)

#     # /def


# # /class

##############################################################################


class GridResidual(ResidualMethod, key="grid"):
    """Residual on a grid.

    Parameters
    ----------
    grid : |CoordinateFrame| or |SkyCoord|
        The grid on which to evaluate the residual. Mandatory.

    original_potential: object or :class:`~PotentialWrapper` or None (optional)
        The potential. If not already a PotentialWrapper, it is wrapped:
        this means that the frame is :class:`~discO.UnFrame`. When evaluating
        the residual, the grid can only be frame-like if this has a frame
        (is not :class:`~discO.UnFrame`).

    observable : str (optional)
        name of method in :class:`~PotentialWrapper`.

    Other Parameters
    ----------------
    representation_type : representation-resolvable (optional, keyword-only)
        The output representation type.
        If None, resolves to ``PotentialWrapper.base_representation``
    **kwargs
        default arguments into ``evaluate_potential``.

    """

    #################################################################
    # On the instance

    def __init__(
        self,
        grid: TH.CoordinateType,
        original_potential: T.Union[PotentialWrapper, T.Any, None] = None,
        observable: str = "acceleration",  # TODO make None and have config
        *,
        representation_type: TH.OptRepresentationLikeType = None,
        **kwargs,
    ) -> None:
        # the points
        self.points = grid
        # initialize
        super().__init__(
            original_potential=original_potential,
            observable=observable,
            representation_type=representation_type,
            **kwargs,
        )

    # /def

    #################################################################
    # evaluate

    def evaluate_potential(
        self,
        potential: T.Union[PotentialWrapper, T.Any],
        observable: T.Optional[str] = None,
        *,
        representation_type: TH.OptRepresentationLikeType = None,
        **kwargs,
    ) -> object:
        """Evaluate method on potential.

        Parameters
        ----------
        potential : object or :class:`~PotentialWrapper`
            The potential. If not already a PotentialWrapper, it is wrapped:
            this means that the frame is :class:`~discO.UnFrame`.
        observable : str
            name of method in :class:`~PotentialWrapper`.
            If None (default), uses default value -- ``.observable``.
        representation_type: representation-resolvable (optional, keyword-only)
            The output representation type. If None (default), uses default
            representation point.
        **kwargs
            Passed to method in :class:`~PotentialWrapper`.
            First mixed in with ``default_params`` (preferring ``kwargs``).

        Returns
        -------
        object

        Other Parameters
        ----------------
        points : frame-like or |Representation| (optional, keyword-only)
            Not recommended, but allowed, to force the points on which the
            residual is calculated. The points can only be frame-like if the
            frame of the potentials is not :class:`~discO.UnFrame`.
            If not specified, and it shouldn't be, uses points determined
            by the class at initialization.

        """
        return super().evaluate_potential(
            potential,
            observable=observable,
            representation_type=representation_type,
            **kwargs,
        )

    # /def


# /class


##############################################################################
# END
