# -*- coding: utf-8 -*-

"""For classes that interact with a Potential."""

__all__ = [
    "CommonBase",
    "PotentialWrapper",
]


##############################################################################
# IMPORTS

# BUILT-IN
import inspect
import typing as T
from abc import ABCMeta, abstractmethod
from collections import Sequence
from types import MappingProxyType, ModuleType

# THIRD PARTY
import astropy.coordinates as coord
import typing_extensions as TE
from astropy.utils import sharedmethod
from astropy.utils.decorators import classproperty
from astropy.utils.introspection import resolve_name

# PROJECT-SPECIFIC
import discO.type_hints as TH
from discO.utils import resolve_framelike

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
        # "from_frame" is the frame of the input points
        if isinstance(points, coord.SkyCoord):
            from_frame = points.frame.replicate_without_data()
        elif isinstance(points, coord.BaseCoordinateFrame):
            from_frame = points.replicate_without_data()
        elif isinstance(points, coord.BaseRepresentation):
            points = frame.realize_frame(points)
            from_frame = None
            frame = None
        else:
            raise TypeError(
                f"points is <{type(points)}> not "
                "<SkyCoord, CoordinateFrame, or Representation>.",
            )

        if (frame is None) or (
            isinstance(frame, coord.BaseCoordinateFrame)
            and frame == from_frame  # errors if not frame instance
        ):  # don't need to xfm
            return points, from_frame

        return points.transform_to(resolve_framelike(frame)), from_frame

    # /def

    def _return_points(
        cls,
        points: TH.CoordinateType,
        rep: TH.RepresentationType,
        representation_type: T.Optional[TH.RepresentationType],
        frame: TH.FrameType,
    ) -> TH.CoordinateType:
        """Helper method for making sure points are returned correctly.

        Parameters
        ----------
        points : |CoordinateFrame| or |SkyCoord|
        frame : |CoordinateFrame|
            Frame of the data
        rep : |Representation|
        representation_type: |Representation|

        Returns
        -------
        point : |CoordinateFrame| or |SkyCoord|

        """
        # if need to change representation type
        # convert rep -> build frame -> (?) make SkyCoord
        if representation_type is not None:
            is_sc: SCorF = (
                True if isinstance(points, coord.SkyCoord) else False
            )

            rep: TH.RepresentationType = rep.represent_as(representation_type)
            points: TH.FrameType = frame.realize_frame(
                rep,
                representation_type=representation_type,
            )

            if is_sc:  # it was & should be a SkyCoord
                points: TH.SkyCoordType = coord.SkyCoord(points, copy=False)

        return points

    # /def

    ########################
    # Methods

    @abstractmethod
    def specific_potential(
        self,
        potential: T.Any,
        points: TH.PositionType,
        *,
        frame: T.Optional[TH.FrameType] = None,
        representation_type: T.Optional[TH.RepresentationType] = None,
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
        raise NotImplementedError()

    # /def

    @abstractmethod
    def specific_force(
        self,
        potential: T.Any,
        points: TH.PositionType,
        *,
        frame: T.Optional[TH.FrameType] = None,
        representation_type: T.Optional[TH.RepresentationType] = None,
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
        raise NotImplementedError()

    # /def

    @abstractmethod
    def acceleration(
        self,
        potential: T.Any,
        points: TH.PositionType,
        *,
        frame: T.Optional[TH.FrameType] = None,
        representation_type: T.Optional[TH.RepresentationType] = None,
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
        raise NotImplementedError()

    # /def


# /class

# -------------------------------------------------------------------


class PotentialWrapper(metaclass=PotentialWrapperMeta):
    """Base-class for evaluating potentials.

    Parameters
    ----------
    potential : object
        The potential
    frame : frame-like (optional, keyword-only)
        The natural frame of the potential.

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
        cls, potential: T.Any, *, frame: T.Optional[TH.FrameLikeType] = None
    ):
        # PotentialWrapper can become any of it's registered subclasses.
        if cls is PotentialWrapper:
            # try to infer the package of the potential.
            key = cls._infer_package(potential, package=None)

            # if key in wrapper, return that class.
            if key in WRAPPER_REGISTRY:
                return super().__new__(WRAPPER_REGISTRY[key])

        return super().__new__(cls)

    # /def

    ####################################################
    # On the instance

    def __init__(
        self, potential: T.Any, *, frame: T.Optional[TH.FrameLikeType] = None
    ):
        # Initialize wrapper for potential.
        self.__wrapped__: object = potential  # a la decorator

        # the "intrinsic" frame of the potential.
        # keep None as None, resolve else-wise.
        self._frame = resolve_framelike(frame) if frame is not None else frame

    # /def

    @property
    def frame(self) -> T.Optional[TH.FrameType]:
        """The |CoordinateFrame| of the potential."""
        return self._frame

    # /def

    ####################################################
    # Methods

    @sharedmethod
    def specific_potential(
        self,
        points: TH.PositionType,
        *,
        representation_type: T.Optional[TH.RepresentationType] = None,
        **kwargs,
    ) -> T.Tuple[TH.CoordinateType, TH.QuantityType]:
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

        """
        return self.__class__.specific_potential(
            self.__wrapped__,  # potential
            points=points,
            frame=self.frame,
            representation_type=representation_type,
            **kwargs,
        )

    # /def

    @sharedmethod
    def specific_force(
        self,
        points: TH.PositionType,
        *,
        representation_type: T.Optional[TH.RepresentationType] = None,
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
        return self.__class__.specific_force(
            self.__wrapped__,
            points=points,
            frame=self.frame,
            representation_type=representation_type,
            **kwargs,
        )

    acceleration = specific_force
    # /def

    ####################################################
    # Utilities

    @staticmethod
    def _infer_package(
        obj: T.Any,
        package: T.Union[ModuleType, str, None] = None,
    ) -> str:

        if inspect.ismodule(package):
            package = package.__name__
        elif isinstance(package, str):
            pass
        elif package is None:  # Need to get package from obj
            info = inspect.getmodule(obj)

            if info is None:  # fails for c-compiled things
                package = obj.__class__.__module__
            else:
                package = info.__package__

            package = package.split(".")[0]

        else:
            raise TypeError("package must be <module> or <str> or None.")

        return package

    # /def


# /class


##############################################################################


class CommonBase(metaclass=ABCMeta):
    """Base-class for interfacing with a Potential.

    Raises
    ------
    TypeError
        On class declaration if metaclass argument 'package' is not a string
        or module.

    KeyError
        On class declaration if class registry information already exists,
        eg. if registering two things under the same name

    """

    #######################################################
    # On the class

    def __init_subclass__(
        cls,
        key: T.Union[
            str,
            ModuleType,
            T.Sequence[T.Union[ModuleType, str]],
            None,
        ] = None,
    ) -> None:
        """Initialize a subclass.

        This method applies to all subclasses, not matter the
        inheritance depth, unless the MRO overrides.

        Parameters
        ----------
        key : str or `~types.ModuleType` or Sequence thereof or None (optional)

            If the key is not None, resolves key module
            and stores it in attribute ``_key``.

        """
        super().__init_subclass__()

        if key is not None:
            key = cls._parse_registry_path(key)

            if key in cls._registry:
                raise KeyError(f"`{key}` sampler already in registry.")

            cls._key = key

    # /def

    def __class_getitem__(cls, key: T.Union[str, T.Sequence]) -> object:
        if isinstance(key, str):
            item = cls._registry[key]
        elif len(key) == 1:
            item = cls._registry[key[0]]
        else:
            item = cls._registry[key[0]][key[1:]]

        return item

    # /def

    @classproperty
    def registry(self) -> T.Mapping:
        """The class registry."""
        return MappingProxyType(self._registry)

    # /def

    @property
    @abstractmethod
    def _registry(self):
        """The class registry. Need to override."""

    # /def

    #################################################################
    # On the instance

    # just write it here to show it's nothing.
    def __init__(self, *args, **kwarg) -> None:
        super().__init__()

    # /def

    #################################################################
    # Running

    @abstractmethod
    def __call__(self):
        """Call. Must be overwritten."""

    # /def

    #################################################################
    # utils

    @staticmethod
    def _infer_package(
        obj: T.Any,
        package: T.Union[ModuleType, str, None] = None,
    ) -> ModuleType:

        if inspect.ismodule(package):
            pass
        elif isinstance(package, str):
            package = resolve_name(package)
        elif package is None:  # Need to get package from obj
            info = inspect.getmodule(obj)

            if info is None:  # fails for c-compiled things
                package = obj.__class__.__module__
            else:
                package = info.__package__

            package = resolve_name(package.split(".")[0])

        else:
            raise TypeError("package must be <module> or <str> or None.")

        return package

    # /def

    @staticmethod
    def _parse_registry_path(
        path: T.Union[str, ModuleType, T.Sequence[T.Union[str, ModuleType]]],
    ) -> str:

        if isinstance(path, str):
            parsed = path
        elif isinstance(path, ModuleType):
            parsed = path.__name__
        elif isinstance(path, Sequence):
            parsed = []
            for p in path:
                if isinstance(p, str):
                    parsed.append(p)
                elif isinstance(p, ModuleType):
                    parsed.append(p.__name__)
                else:
                    raise TypeError(
                        f"{path} is not <str, ModuleType, Sequence>",
                    )
        else:
            raise TypeError(
                f"{path} is not <str, ModuleType, or Sequence thereof>",
            )

        return parsed

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
