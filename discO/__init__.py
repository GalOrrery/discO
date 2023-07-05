# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Disk Expansion Project."""

__author__ = ("Nathaniel Starkman", "Christopher Carr")

__credits__ = ["Jo Bovy"]
__maintainer__ = "Nathaniel Starkman"
__status__ = "In Progress"

from importlib.metadata import version as _get_version

from . import core, data, plugin
from .config import conf
from .core import *  # noqa: F401, F403
from .plugin import *  # noqa: F401, F403
from .utils import UnFrame

__all__ = [
    # class
    "UnFrame",
    # data
    "data",
    # configuration
    "conf",
]
__all__ += core.__all__
__all__ += plugin.__all__
__version__ = _get_version("discO")
