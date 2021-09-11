# -*- coding: utf-8 -*-
"""Configure Test Suite.

This file is used to configure the behavior of pytest when using the Astropy
test infrastructure. It needs to live inside the package in order for it to
get picked up when running the tests inside an interpreter using
packagename.test

"""

##############################################################################
# IMPORTS

# STDLIB
import os

# THIRD PARTY
import pytest

# LOCAL
from .setup_package import HAS_AGAMA, HAS_GALA, HAS_GALPY

try:
    # THIRD PARTY
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS

    ASTROPY_HEADER = True
except ImportError:
    ASTROPY_HEADER = False

# /if

##############################################################################
# CODE
##############################################################################


# ------------------------------------------------------
# Test collection: ignore patterns

collect_ignore = ["setup.py"]

# AGAMA
SKIP_NO_AGAMA = pytest.mark.skipif(not HAS_AGAMA, reason="needs agama")
if not HAS_AGAMA:
    collect_ignore.append("plugin/agama/")

# Gala
SKIP_NO_AGAMA = pytest.mark.skipif(not HAS_GALA, reason="needs gala")
if not HAS_GALA:
    collect_ignore.append("plugin/gala/")

# Galpy
SKIP_NO_GALPY = pytest.mark.skipif(not HAS_GALPY, reason="needs galpy")
if not HAS_GALPY:
    collect_ignore.append("plugin/galpy/")
else:
    from galpy.util import config as galpy_config
    galpy_config.__config__["astropy"]["astropy-units"] = "True"
    galpy_config.__config__["astropy"]["astropy-coords"] = "True"


# ------------------------------------------------------


def pytest_configure(config):
    """Configure Pytest with Astropy.

    Parameters
    ----------
    config : pytest configuration

    """
    if ASTROPY_HEADER:

        config.option.astropy_header = True

        # Customize the following lines to add/remove entries from the list of
        # packages for which version numbers are displayed when running the
        # tests.
        PYTEST_HEADER_MODULES.pop("Pandas", None)
        PYTEST_HEADER_MODULES["scikit-image"] = "skimage"

        # LOCAL
        from . import __version__

        packagename = os.path.basename(os.path.dirname(__file__))
        TESTED_VERSIONS[packagename] = __version__


# /def


# ------------------------------------------------------


@pytest.fixture(autouse=True)
def add_numpy(doctest_namespace):
    """Add NumPy to Pytest.

    Parameters
    ----------
    doctest_namespace : namespace

    """
    # THIRD PARTY
    import numpy

    # add to namespace
    doctest_namespace["np"] = numpy

    return


# def


@pytest.fixture(autouse=True)
def add_astropy(doctest_namespace):
    """Add Astropy stuff to Pytest.

    Parameters
    ----------
    doctest_namespace : namespace

    """
    # THIRD PARTY
    import astropy.coordinates as coord
    import astropy.units

    # add to namespace
    doctest_namespace["coord"] = coord
    doctest_namespace["u"] = astropy.units

    return


# def

##############################################################################
# END
