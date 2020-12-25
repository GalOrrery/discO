# -*- coding: utf-8 -*-

"""Common code."""


# __all__ = [
# ]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u

##############################################################################
# PARAMETERS

EllipsisType = type(Ellipsis)

UnitType = T.Union[
    T.TypeVar("Unit", bound=u.UnitBase),
    T.TypeVar("FunctionUnit", bound=u.FunctionUnitBase),
]
QuantityType = T.TypeVar("Quantity", bound=u.Quantity)


FrameType = T.TypeVar("CoordinateFrame", bound=coord.BaseCoordinateFrame)
SkyCoordType = T.TypeVar("SkyCoord", bound=coord.SkyCoord)
CoordinateType = T.Union[FrameType, SkyCoordType]

FrameLikeType = T.Union[CoordinateType, str]

##############################################################################
# END
