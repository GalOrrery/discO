# -*- coding: utf-8 -*-

"""For classes that interact with a Potential."""

__all__ = [
    "CommonBase",
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
import typing_extensions as TE
from astropy.utils.decorators import classproperty
from astropy.utils.introspection import resolve_name

# PROJECT-SPECIFIC
import discO.type_hints as TH

##############################################################################
# PARAMETERS

WRAPPER_REGISTRY: T.Dict[str, object] = dict()

SCorF = T.Union[TH.SkyCoordType, TE.Literal[False]]

##############################################################################
# CODE
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
        # registry is a property on own class
        if isinstance(cls._registry, property):
            return None

        # else, filter registry by subclass
        return MappingProxyType(
            {k: v for k, v in cls._registry.items() if issubclass(v, cls) and v is not cls},
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
