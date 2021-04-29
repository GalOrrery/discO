# -*- coding: utf-8 -*-

"""For classes that interact with a Potential."""

__all__ = [
    "PotentialWrapper",
]


##############################################################################
# IMPORTS

# BUILT-IN
import inspect
import typing as T
from abc import ABCMeta, abstractmethod
from types import ModuleType

# THIRD PARTY
import astropy.coordinates as coord
import typing_extensions as TE
from astropy.utils import sharedmethod
from astropy.utils.misc import indent

# PROJECT-SPECIFIC
import discO.type_hints as TH
from discO.utils import resolve_framelike, resolve_representationlike
from discO.utils.coordinates import UnFrame

##############################################################################
# PARAMETERS

WRAPPER_REGISTRY: T.Dict[str, object] = dict()

SCorF = T.Union[TH.SkyCoordType, TE.Literal[False]]

##############################################################################
# CODE
##############################################################################


class PotentialWrapperMeta(ABCMeta):
    """Meta-class for Potential Wrapper."""

    def _convert_to_frame(
        cls,
        points: TH.PositionType,
        frame: T.Optional[TH.FrameLikeType],
        representation_type: TH.OptRepresentationLikeType = None,
    ) -> T.Tuple[TH.CoordinateType, T.Union[TH.FrameType, str, None]]:
        """Convert points to the given coordinate frame.

        Parameters
        ----------
        points : |SkyCoord| or |CoordinateFrame| or |Representation|
            The points at which to evaluate the potential.
        frame : |CoordinateFrame| or None
            The frame in to which `points` are transformed.
            If None, then no transformation is applied.

        Returns
        -------
        points : |CoordinateFrame| or |SkyCoord|
            Same type as `points`, in the potential's frame.
        from_frame : |CoordinateFrame|
            The frame of the input points,

        """
        resolved_frame = resolve_framelike(frame)  # works with None and ...

        # -----------
        # from_frame

        # "from_frame" is the frame of the input points
        if isinstance(points, coord.SkyCoord):
            from_frame = points.frame.replicate_without_data()
            from_rep = points.representation_type
        elif isinstance(points, coord.BaseCoordinateFrame):
            from_frame = points.replicate_without_data()
            from_rep = points.representation_type
        elif isinstance(points, coord.BaseRepresentation):  # (frame is None)
            from_frame = None
            from_rep = points.__class__
        else:
            raise TypeError(
                f"points is <{type(points)}> not "
                "<SkyCoord, CoordinateFrame, or Representation>.",
            )

        if isinstance(resolved_frame, UnFrame) and not (
            from_frame is None or isinstance(from_frame, UnFrame)
        ):
            raise TypeError(
                "To pass points as SkyCoord or CoordinateFrame, "
                "the potential must have a frame.",
            )

        # -----------
        # parse rep

        if representation_type is None:
            rep_type = from_rep
        else:
            rep_type = resolve_representationlike(representation_type)

        # -----------
        # to frame

        # potential doesn't have a frame
        if frame is None:
            p = points
        elif from_frame is None:  # but frame is not None
            p = resolved_frame.realize_frame(
                points,
                representation_type=rep_type,
            )
        # don't need to transform
        elif (
            isinstance(frame, coord.BaseCoordinateFrame)  # catch comp error
            and frame == from_frame  # equivalent frames
        ):
            p = points
        else:
            p = points.transform_to(resolved_frame)

        # -----------
        # to rep

        if isinstance(p, coord.BaseRepresentation):
            p = p.represent_as(rep_type)
        elif isinstance(p, coord.BaseCoordinateFrame) and rep_type is not None:
            p._data = p._data.represent_as(rep_type)
        elif isinstance(p, coord.SkyCoord) and rep_type is not None:
            p.frame._data = p.frame._data.represent_as(rep_type)

        return p, from_frame

    # /def

    ########################
    # Methods

    @abstractmethod
    def total_mass(self, potential):
        """Evaluate the total mass.

        Parameters
        ----------
        potential : object
            The potential.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError("Please use the appropriate subpackage.")

    # /def

    @abstractmethod
    def potential(
        self,
        potential: T.Any,
        points: TH.PositionType,
        *,
        frame: T.Optional[TH.FrameType] = None,
        representation_type: TH.OptRepresentationLikeType = None,
        **kwargs,
    ):
        """Evaluate the specific potential.

        Parameters
        ----------
        potential : object
            The potential.
        points : coord-array or |Representation| or None (optional)
            The points at which to evaluate the potential.
        frame : |CoordinateFrame| or None (optional, keyword-only)
            The frame of the potential. Potentials do not have an intrinsic
            reference frame, but if one is assigned, then anything needs to be
            converted to that frame.
        representation_type : |Representation| or None (optional, keyword-only)
            The representation type in which to return data.
        **kwargs
            Arguments into the potential.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError("Please use the appropriate subpackage.")

    # /def

    @abstractmethod
    def specific_force(
        self,
        potential: T.Any,
        points: TH.PositionType,
        *,
        frame: T.Optional[TH.FrameType] = None,
        representation_type: TH.OptRepresentationLikeType = None,
        **kwargs,
    ):
        """Evaluate the specific force.

        Parameters
        ----------
        potential : object
            The potential.
        points : coord-array or |Representation| or None (optional)
            The points at which to evaluate the potential.
        frame : |CoordinateFrame| or None (optional, keyword-only)
            The frame of the potential. Potentials do not have an intrinsic
            reference frame, but if one is assigned, then anything needs to be
            converted to that frame.
        representation_type : |Representation| class or None (optional)
            The representation type in which to return data.
        **kwargs
            Arguments into the potential.

        Returns
        -------
        `~discO.utils.vectorfield.BaseVectorField` subclass instance
            Type set by `representation_type`

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError("Please use the appropriate subpackage.")

    # /def

    @abstractmethod
    def acceleration(
        self,
        potential: T.Any,
        points: TH.PositionType,
        *,
        frame: T.Optional[TH.FrameType] = None,
        representation_type: TH.OptRepresentationLikeType = None,
    ):
        """Evaluate the acceleration.

        Parameters
        ----------
        potential : object
            The potential.
        points : coord-array or |Representation| or None (optional)
            The points at which to evaluate the potential.
        frame : |CoordinateFrame| or None (optional, keyword-only)
            The frame of the potential. Potentials do not have an intrinsic
            reference frame, but if one is assigned, then anything needs to be
            converted to that frame.
        representation_type : |Representation| class or None (optional)
            The representation type in which to return data.

        Returns
        -------
        `~discO.utils.vectorfield.BaseVectorField` subclass instance


        """
        raise NotImplementedError("Please use the appropriate subpackage.")

    # /def

    # -----------------------------------------------------

    def coefficients(self, potential) -> T.Optional[T.Dict[str, T.Any]]:
        """Coefficients of the potential.

        Parameters
        ----------
        potential : object
            The potential.

        Returns
        -------
        None or dict
            None if there aren't coefficients, a dict of the coefficients
            if there are.

        """
        raise NotImplementedError("Please use the appropriate subpackage.")

    # /def


# /class

#####################################################################


class PotentialWrapper(metaclass=PotentialWrapperMeta):
    """Base-class for evaluating potentials.

    Parameters
    ----------
    potential : object
        The potential.
    frame : frame-like or None or Ellipsis (optional, keyword-only)
        The natural frame of the potential.
    representation_type : representation-like or None or Ellipsis

    """

    ####################################################
    # On the class

    def __init_subclass__(
        cls,
        key: T.Union[str, ModuleType, None] = None,
    ) -> None:
        """Initialize subclass, optionally adding to registry.

        Parameters
        ----------
        key : str or Module or None (optional)
            Optionally register subclasses by key.
            Not registered if key is None.

        """
        if inspect.ismodule(key):  # module -> str
            key = key.__name__

        cls._key: T.Optional[str] = key

        if key is not None:  # same trigger as CommonBase
            # cls._key defined in super()
            WRAPPER_REGISTRY[cls._key] = cls

    # /def

    def __class_getitem__(cls, key: str) -> PotentialWrapperMeta:
        """Get class from registry.

        Parameters
        ----------
        key : str
            Get class from registry.

        Returns
        -------
        class
            class from registry.

        """
        return WRAPPER_REGISTRY[key]

    # /def

    def __new__(
        cls,
        potential: T.Any,
        *,
        frame: T.Optional[TH.FrameLikeType] = None,
        representation_type: TH.OptRepresentationLikeType = None,
    ):
        # A wrapped potential goes thru
        if isinstance(potential, PotentialWrapper):
            # get class
            kls = potential.__class__
            # get frame and rep-type, defaulting to wrapper
            frame = potential.frame if frame is None else frame
            representation_type = (
                potential.representation_type
                if representation_type is None
                else representation_type
            )

            # we return using kls's __new__, in-case it does something special.
            return kls.__new__(
                kls,
                potential.__wrapped__,
                frame=frame,
                representation_type=representation_type,
            )

        # PotentialWrapper can become any of it's registered subclasses.
        if cls is PotentialWrapper:
            # try to infer the package of the potential.
            key = cls._infer_key(potential, package=None)

            # if key in wrapper, return that class.
            if key in WRAPPER_REGISTRY:
                kls = WRAPPER_REGISTRY[key]
                return kls.__new__(
                    kls,
                    potential,
                    frame=frame,
                    representation_type=representation_type,
                )
        return super().__new__(cls)

    # /def

    ####################################################
    # On the instance

    def __init__(
        self,
        potential: T.Any,
        *,
        frame: TH.OptFrameLikeType = None,
        representation_type: TH.OptRepresentationLikeType = None,
    ):
        # if it's a wrapper, have to pop back
        if isinstance(potential, PotentialWrapper):
            # frame
            frame = potential.frame if frame is None else frame
            # representation type
            representation_type = (
                potential.representation_type
                if representation_type is None
                else representation_type
            )
            # potential
            potential = potential.__wrapped__

        # Initialize wrapper for potential.
        self.__wrapped__: object = potential  # a la decorator

        # the "intrinsic" frame of the potential.
        # resolve else-wise (None -> UnFrame)
        self._frame = (
            resolve_framelike(frame) if frame is not Ellipsis else frame
        )
        self._default_representation = (
            resolve_representationlike(representation_type)
            if representation_type not in (None, Ellipsis)
            else representation_type
        )
        if frame is not Ellipsis and self._default_representation not in (
            Ellipsis,
            None,
        ):
            self._frame.representation_type = self._default_representation

    # /def

    # -----------------------------------------------------

    @property
    def wrapped(self) -> object:
        """The wrapped potential."""
        return self.__wrapped__

    # /def

    @property
    def frame(self) -> T.Optional[TH.FrameType]:
        """The |CoordinateFrame| of the potential."""
        return self._frame

    # /def

    @property
    def default_representation(self) -> TH.RepresentationType:
        """The default |Representation| of the potential."""
        return self._default_representation
        # TODO? should this resolve?
        # return resolve_representationlike(self._default_representation)

    # /def

    @property
    def representation_type(
        self,
    ) -> T.Union[TH.RepresentationType, TH.EllipsisType]:
        """The |Representation| of the potential."""
        if self.frame is Ellipsis or (
            # UnFrame should not have a representation_type, but check
            isinstance(self.frame, UnFrame)
            and getattr(self.frame, "representation_type", None) is None
        ):
            return self._default_representation

        return self.frame.representation_type

    # /def

    # -----------------------------------------------------

    def __call__(
        self,
        points: TH.PositionType,
        *,
        representation_type: TH.OptRepresentationLikeType = None,
        **kwargs,
    ):
        """Evaluate the specific potential.

        Parameters
        ----------
        points : coord-array or |Representation| or None (optional)
            The points at which to evaluate the potential.
            Potentials do not have an intrinsic reference frame, but if we
            have assigned one, then anything needs to be converted to that
            frame.
        representation_type : |Representation| or None (optional, keyword-only)
            The representation type in which to return data.
            None means no representation is forced.
        **kwargs
            Arguments into the potential.

        Returns
        -------
        points : |CoordinateFrame| or |SkyCoord|
        values : |Quantity|

        """
        return self.potential(
            points, representation_type=representation_type, **kwargs
        )

    # /def

    ####################################################
    # Methods

    @sharedmethod  # TODO as property
    def total_mass(self):
        """Evaluate the total mass.

        Returns
        -------
        mtot : |Quantity|
            The total mass.

        """
        return self.__class__.total_mass(self.__wrapped__)

    # /def

    # -----------------------------------------------------

    @sharedmethod
    def potential(
        self,
        points: TH.PositionType,
        *,
        representation_type: TH.OptRepresentationLikeType = None,
        **kwargs,
    ) -> T.Tuple[TH.CoordinateType, TH.QuantityType]:
        """Evaluate the potential.

        Parameters
        ----------
        points : coord-array or |Representation| or None (optional)
            The points at which to evaluate the potential.
            Potentials do not have an intrinsic reference frame, but if we
            have assigned one, then anything needs to be converted to that
            frame.
        representation_type : |Representation| or None (optional, keyword-only)
            The representation type in which to return data.
            None means no representation is forced.
        **kwargs
            Arguments into the potential.

        Returns
        -------
        points : |CoordinateFrame| or |SkyCoord|
        values : |Quantity|

        """
        # if representation type is None, use default
        representation_type = (
            self.representation_type
            if representation_type is None
            else representation_type
        )

        return self.__class__.potential(
            self.__wrapped__,  # potential
            points=points,
            frame=self.frame,
            representation_type=representation_type,
            **kwargs,
        )

    # /def

    # -----------------------------------------------------

    @sharedmethod
    def specific_force(
        self,
        points: TH.PositionType,
        *,
        representation_type: TH.OptRepresentationLikeType = None,
        **kwargs,
    ) -> T.Tuple[TH.CoordinateType, TH.QuantityType]:
        """Evaluate the specific force.

        Parameters
        ----------
        points : coord-array or |Representation| or None (optional)
            The points at which to evaluate the force.
            Potentials do not have an intrinsic reference frame, but if we
            have assigned one, then anything needs to be converted to that
            frame.
        representation_type : |Representation| or None (optional, keyword-only)
            The representation type in which to return data.
        **kwargs
            Arguments into the potential.

        Returns
        -------
        `~discO.utils.vectorfield.BaseVectorField` subclass instance
            Type set by `representation_type`

        """
        # if representation type is None, use default
        representation_type = (
            self.representation_type
            if representation_type is None
            else representation_type
        )

        return self.__class__.specific_force(
            self.__wrapped__,
            points=points,
            frame=self.frame,
            representation_type=representation_type,
            **kwargs,
        )

    acceleration = specific_force
    # /def

    # -----------------------------------------------------

    @sharedmethod
    def coefficients(self):
        """Coefficients of the potential.

        .. todo::

            Make this an attribute

        Returns
        -------
        None or dict
            None if there aren't coefficients, a dict of the coefficients
            if there are.

        """
        return self.__class__.coefficients(self.__wrapped__)

    # /def

    ####################################################
    # Misc

    @staticmethod
    def _infer_key(
        obj: T.Any,
        package: T.Union[ModuleType, str, None] = None,
    ) -> str:
        """Figure out the package name of an object.

        .. todo::

            Break out as utility function.
            Combine with the one in CommonBase.

        Parameters
        ----------
        obj : object
            The object. What's its package?
        package : :class:`~types.ModuleType` or str of None (optional)
            Any hints about the package.
            If module, then the module's name is returned.
            If string, then this is returned.
            If None (default), `obj` is inspected.

        Returns
        -------
        key : str
            The inferred key name.

        """
        if isinstance(package, str):
            key = package
        elif inspect.ismodule(package):
            key = package.__name__
        elif package is None:  # Need to get package from obj
            info = inspect.getmodule(obj)

            if info is None:  # fails for c-compiled things
                package = obj.__class__.__module__
            else:
                package = info.__package__

            key = package.split(".")[0]

        else:
            raise TypeError("package must be <module, str, or None>.")

        return key

    # /def

    def __repr__(self) -> str:
        """String representation."""
        r = super().__repr__()
        r = r[r.index("at") + 3 :]  # noqa: E203  # includes ">"

        # the potential
        ps = repr(self.__wrapped__).strip()
        ps = ps[1:] if ps.startswith("<") else ps
        ps = ps[:-1] if ps.endswith(">") else ps

        # the frame
        fs = repr(self.frame).strip()
        fs = fs[1:] if fs.startswith("<") else fs
        fs = fs[:-1] if fs.endswith(">") else fs

        return "\n".join(
            (
                self.__class__.__name__ + ": at <" + r,
                indent("potential : " + ps),
                indent("frame     : " + fs),
            ),
        )

    # /def


# /class

# -------------------------------------------------------------------


##############################################################################
# END
