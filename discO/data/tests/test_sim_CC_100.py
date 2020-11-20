# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.data`."""

__all__ = [
    "test_load_Milky_Way_Sim_100",
]


##############################################################################
# IMPORTS

# BUILT-IN
import pathlib
import shutil

# THIRD PARTY
import pytest

# PROJECT-SPECIFIC
from discO import data as loaddata

##############################################################################
# PARAMETERS

schema = r"""# %ECSV 0.9
# ---
# datatype:
# - {name: ID, datatype: int64}
# - {name: mass, unit: solMass, datatype: float64}
# - {name: x, unit: kpc, datatype: float64}
# - {name: y, unit: kpc, datatype: float64}
# - {name: z, unit: kpc, datatype: float64}
# - {name: vx, unit: km / s, datatype: float64}
# - {name: vy, unit: km / s, datatype: float64}
# - {name: vz, unit: km / s, datatype: float64}
# meta: !!omap
# - __serialized_columns__:
#     mass:
#       __class__: astropy.units.quantity.Quantity
#       unit: !astropy.units.Unit {unit: solMass}
#       value: !astropy.table.SerializedColumn {name: mass}
#     vx:
#       __class__: astropy.units.quantity.Quantity
#       unit: !astropy.units.Unit {unit: km / s}
#       value: !astropy.table.SerializedColumn {name: vx}
#     vy:
#       __class__: astropy.units.quantity.Quantity
#       unit: !astropy.units.Unit {unit: km / s}
#       value: !astropy.table.SerializedColumn {name: vy}
#     vz:
#       __class__: astropy.units.quantity.Quantity
#       unit: !astropy.units.Unit {unit: km / s}
#       value: !astropy.table.SerializedColumn {name: vz}
#     x:
#       __class__: astropy.units.quantity.Quantity
#       unit: &id001 !astropy.units.Unit {unit: pc}
#       value: !astropy.table.SerializedColumn {name: x}
#     y:
#       __class__: astropy.units.quantity.Quantity
#       unit: *id001
#       value: !astropy.table.SerializedColumn {name: y}
#     z:
#       __class__: astropy.units.quantity.Quantity
#       unit: *id001
#       value: !astropy.table.SerializedColumn {name: z}
# schema: astropy-2.0
ID mass x y z vx vy vz
"""

# Make Test Directory
# skip if it exists, since this is a large file and we only want to test on
# the server
fdir = pathlib.Path(__file__).parent.joinpath("../sim_CC_100").resolve()
if fdir.is_dir():
    pytestmark = pytest.mark.skip("Directory Exists")

##############################################################################
# PYTEST


def setup_module(module):
    """Setup module for testing."""
    if not fdir.is_dir():
        fdir.mkdir()

    module.testdir = fdir
    print(fdir)

    # Make bulge_100.ecsv
    fp = fdir.joinpath("bulge_100.ecsv")
    if not fp.is_file():
        with fp.open("w") as file:
            file.write(schema + "0 9862 -98 43 -99 -30 -25 -50")

    # Make disc_100.ecsv
    fp = fdir.joinpath("disc_100.ecsv")
    if not fp.is_file():
        with fp.open("w") as file:
            file.write(schema + "0 44170744 -0 -0 0 -281 -50 182")

    # Make halo_100.ecsv
    fp = fdir.joinpath("halo_100.ecsv")
    if not fp.is_file():
        with fp.open("w") as file:
            file.write(schema + "0 44170744 -0 -0 0 -281 -50 182")


# /def


def teardown_module(module):
    """Tear-down module for testing."""
    shutil.rmtree(module.testdir)


# /def


##############################################################################
# TESTS
##############################################################################


@pytest.mark.parametrize("component", ["disc", "bulge", "halo"])
def test_load_Milky_Way_Sim_100(component):
    """Test read method."""
    data = loaddata.load_Milky_Way_Sim_100(component)

    assert data.colnames == ["ID", "mass", "x", "y", "z", "vx", "vy", "vz"]


# /def


# -------------------------------------------------------------------


def test_load_Milky_Way_Sim_100_bulge():
    """Test read method."""
    data = loaddata.load_Milky_Way_Sim_100_bulge()
    assert data.colnames == ["ID", "mass", "x", "y", "z", "vx", "vy", "vz"]


# /def


# -------------------------------------------------------------------


def test_load_Milky_Way_Sim_100_disc():
    """Test read method."""
    data = loaddata.load_Milky_Way_Sim_100_disc()
    assert data.colnames == ["ID", "mass", "x", "y", "z", "vx", "vy", "vz"]


# /def


# -------------------------------------------------------------------


def test_load_Milky_Way_Sim_100_halo():
    """Test read method."""
    data = loaddata.load_Milky_Way_Sim_100_halo()
    assert data.colnames == ["ID", "mass", "x", "y", "z", "vx", "vy", "vz"]


# /def

# -------------------------------------------------------------------

##############################################################################
# END
