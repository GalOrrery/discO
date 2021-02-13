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
from collections.abc import Sequence
from types import MappingProxyType, ModuleType

# THIRD PARTY
import astropy.coordinates as coord
import typing_extensions as TE
from astropy.coordinates.representation import (
    REPRESENTATION_CLASSES as _REP_CLSs,
)
from astropy.utils import sharedmethod
from astropy.utils.decorators import classproperty
from astropy.utils.introspection import resolve_name
from astropy.utils.misc import indent

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
        representation_type: T.Optional[TH.RepresentationType] = None,
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
        if (frame is None) and isinstance(
            points,
            (coord.SkyCoord, coord.BaseCoordinateFrame),
        ):
            raise TypeError(
                "To pass points as SkyCoord or CoordinateFrame, "
                "the potential must have a frame.",
            )

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

        # -----------
        # parse rep

        if representation_type is None:
            rep_type = from_rep
        elif inspect.isclass(representation_type) and issubclass(
            representation_type,
            coord.BaseRepresentation,
        ):
            rep_type = representation_type
        elif isinstance(representation_type, str):
            rep_type = _REP_CLSs[representation_type]
        else:
            raise TypeError(
                f"representation_type is <{type(representation_type)}> not "
                "<Representation, str, or None>.",
            )

        # -----------
        # to frame

        # potential doesn't have a frame
        if frame is None:
            p = points
        elif from_frame is None:  # but frame is not
            p = resolve_framelike(frame).realize_frame(
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
            p = points.transform_to(resolve_framelike(frame))

        # -----------
        # to rep

        if isinstance(p, coord.BaseRepresentation):
            p = p.represent_as(rep_type)
        elif isinstance(p, coord.BaseCoordinateFrame):
            p._data = p._data.represent_as(rep_type)
        else:  # SkyCoord
            p.frame._data = p.frame._data.represent_as(rep_type)

        return p, from_frame

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
        raise NotImplementedError("Please use the appropriate subpackage.")

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
        raise NotImplementedError("Please use the appropriate subpackage.")

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
        raise NotImplementedError("Please use the appropriate subpackage.")

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
            key = cls._infer_key(potential, package=None)

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
        # if it's a wrapper, have to pop back
        if isinstance(potential, PotentialWrapper):
            potential = potential.__wrapped__

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

    def __call__(
        self,
        points: TH.PositionType,
        *,
        representation_type: T.Optional[TH.RepresentationType] = None,
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
        return self.specific_potential(
            points, representation_type=representation_type, **kwargs
        )

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

        Returns
        -------
        points : |CoordinateFrame| or |SkyCoord|
        values : |Quantity|

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
            item = cls.registry[key]
        elif len(key) == 1:
            item = cls.registry[key[0]]
        else:
            item = cls.registry[key[0]][key[1:]]

        return item

    # /def

    @classmethod
    def _in_registry(cls, key: T.Union[str, T.Tuple[str]]):
        """Is it in the registry?"""
        # make iterable
        if isinstance(key, str):
            key = [key]

        # start with the whole shebang
        registry = cls.registry
        # iterate through registry
        for k in key:
            if (registry is None) or (k not in registry):
                return False
            # update what we call the registry
            registry = cls.registry[k].registry

        return True

    # /def

    @property
    @abstractmethod
    def _registry(self):
        """The class registry. Need to override."""

    # /def

    @classproperty
    def registry(cls) -> T.Mapping:
        """The class registry."""
        if isinstance(cls._registry, property):
            return None

        return MappingProxyType(
            {
                k: v
                for k, v in cls._registry.items()
                if issubclass(v, cls) and v is not cls
            },
        )

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
