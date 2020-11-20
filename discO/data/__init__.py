# -*- coding: utf-8 -*-
# see LICENSE.rst

# ----------------------------------------------------------------------------
#
# TITLE   : Data
# PROJECT : `discO`
#
# ----------------------------------------------------------------------------

"""Data Loaders."""


__all__ = [
    "load_Milky_Way_Sim_100",
    "load_Milky_Way_Sim_100_bulge",
    "load_Milky_Way_Sim_100_disc",
    "load_Milky_Way_Sim_100_halo",
]


##############################################################################
# IMPORTS

# BUILT IN
from typing_extensions import Literal

# THIRD PARTY
from astropy.table import QTable
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.decorators import format_doc

##############################################################################
# CODE
##############################################################################

#####################################################################
# Milky Way Sim 100


def load_Milky_Way_Sim_100(
    component: Literal["disc", "bulge", "halo"] = "disc"
) -> QTable:
    """Load Data from Milky Way Sim 100.

    Parameters
    ----------
    component: str
        One of "disc", "bulge", or "halo"

    Returns
    -------
    `~astropy.table.QTable`
        With columns:

        - 'ID'
        - 'mass' : solMass
        - 'x' : kpc
        - 'y' : kpc
        - 'z' : kpc
        - 'vx' : km / s
        - 'vy' : km / s
        - 'vz' : km / s

    """
    fname = get_pkg_data_filename(
        f"data/sim_CC_100/{component}_100.ecsv", package="discO"
    )

    return QTable.read(fname, format="ascii.ecsv")


# /def

# -------------------------------------------------------------------

_docstring: str = """Load {component} Data from Milky Way Sim 100.

    Returns
    -------
    `~astropy.table.QTable`
        With columns:

        - 'ID'
        - 'mass' : solMass
        - 'x' : kpc
        - 'y' : kpc
        - 'z' : kpc
        - 'vx' : km / s
        - 'vy' : km / s
        - 'vz' : km / s

    """


@format_doc(_docstring, component="Bulge")
def load_Milky_Way_Sim_100_bulge() -> QTable:
    """Load Bulge Data from Milky Way Sim 100."""
    return load_Milky_Way_Sim_100("bulge")


# /def


@format_doc(_docstring, component="Disc")
def load_Milky_Way_Sim_100_disc() -> QTable:
    """Load Disc Data from Milky Way Sim 100."""
    return load_Milky_Way_Sim_100("disc")


# /def


@format_doc(_docstring, component="Dark Matter")
def load_Milky_Way_Sim_100_halo() -> QTable:
    """Load Dark Matter Halo Data from Milky Way Sim 100."""
    return load_Milky_Way_Sim_100("halo")


# /def

#####################################################################


##############################################################################
# END
