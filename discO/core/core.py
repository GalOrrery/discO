# -*- coding: utf-8 -*-

"""Baseclass for classes that interact with a Potential."""

__all__ = [
    "PotentialBase",
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
from astropy.utils.decorators import classproperty
from astropy.utils.introspection import resolve_name

##############################################################################
# CODE
##############################################################################


class PotentialBase(metaclass=ABCMeta):
    """Sample a Potential.

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
            None,
            T.Sequence[T.Union[ModuleType, str]],
        ] = None,
    ):
        """Initialize a subclass.

        This method applies to all subclasses, not matter the
        inheritance depth, unless the MRO overrides.

        Parameters
        ----------
        key : str or `~types.ModuleType` or None (optional)

            If the key is not None, resolves key module
            and stores it in attribute ``_key``.

            .. todo::

                Maybe store as a string instead.

        """
        super().__init_subclass__()

        if key is not None:
            key = cls._parse_registry_path(key)

            if key in cls._registry:
                raise KeyError(f"`{key}` sampler already in registry.")

            cls._key = key

    # /def

    def __class_getitem__(cls, key):
        if isinstance(key, str):
            item = cls._registry[key]
        elif len(key) == 1:
            item = cls._registry[key[0]]
        else:
            item = cls._registry[key[0]][key[1:]]

        return item

    # /def

    @classproperty
    def registry(self):
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
    def __init__(self, *args, **kwarg):
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
    ):

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
    def _parse_registry_path(path):

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
            raise TypeError(f"{path} is not <str, ModuleType, Sequence>")

        return parsed

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
