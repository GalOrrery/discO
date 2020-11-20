# -*- coding: utf-8 -*-

"""**DOCSTRING**.

Description.

"""

__all__ = [
    "PotentialBase",
]


##############################################################################
# IMPORTS

# BUILT-IN
import inspect
import typing as T
from abc import ABC, abstractmethod
from types import ModuleType

# THIRD PARTY
from astropy.utils.introspection import resolve_name

##############################################################################
# PARAMETERS


##############################################################################
# CODE
##############################################################################


class PotentialBase(ABC):
    """Sample a Potential.

    Raises
    ------
    TypeError
        On class declaration if metaclass argument 'package' is not a string
        or module.

    """

    def __init_subclass__(cls, package: T.Union[str, ModuleType, None] = None):
        super().__init_subclass__()

        if package is not None:

            if isinstance(package, str):
                package = resolve_name(package)
            elif not isinstance(package, ModuleType):
                raise TypeError

            if package in cls._registry:
                raise Exception(f"`{package}` sampler already in registry.")

            cls._package = package

    # /def

    @property
    @abstractmethod
    def _registry(self):
        """The class registry. Need to override."""
        pass

    # /def

    #################################################################
    # utils

    @staticmethod
    def _infer_package(
        obj: T.Any, package: T.Union[ModuleType, str, None] = None
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


# /class


# -------------------------------------------------------------------


##############################################################################
# END
