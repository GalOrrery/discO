# -*- coding: utf-8 -*-

"""Type hints.

This project extensively uses :mod:`~typing` hints.
Note that this is not (necessarily) static typing.


**TypeVar**

Most of the types are :class:`~typing.TypeVar` with a standard format: for an
object X, the variable name and TypeVar name are "{X}Type" and the TypeVar is
bound to X such that all subclasses of X permit the same type hint.

As a trivial example,

    >>> import typing as T
    >>> IntType = T.TypeVar("Int", bound=int)

``IntType`` works on any subclass (inclusive) of int.

"""

__all__ = [
    "NoneType",
    "EllipsisType",
    # Astropy types
    # coordinates
    "RepresentationOrDifferentialType",
    "RepresentationType",
    "OptRepresentationType",
    "RepresentationLikeType",
    "OptRepresentationLikeType",
    "DifferentialType",
    "FrameType",
    "OptFrameType",
    "SkyCoordType",
    "CoordinateType",
    "PositionType",
    "GenericPositionType",
    "FrameLikeType",
    "OptFrameLikeType",
    # tables
    "TableType",
    "QTableType",
    # units
    "UnitType",
    "UnitLikeType",
    "QuantityType",
    "QuantityLikeType",
]

__credits__ = ["Astropy"]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
from astropy import table

##############################################################################
# TYPES
##############################################################################

NoneType = type(None)
EllipsisType = type(Ellipsis)

# -------------------------------------
# Astropy types

# -----------------
# coordinates

RepresentationOrDifferentialType = T.TypeVar(
    "BaseRepresentationOrDifferential",
    bound=coord.BaseRepresentationOrDifferential,
)

RepresentationType = T.TypeVar(
    "BaseRepresentation",
    bound=coord.BaseRepresentation,
)

OptRepresentationType = T.Union[RepresentationType, None, EllipsisType]

RepresentationLikeType = T.Union[RepresentationType, str]

OptRepresentationLikeType = T.Union[RepresentationLikeType, None, EllipsisType]

DifferentialType = T.TypeVar("BaseDifferential", bound=coord.BaseDifferential)

FrameType = T.TypeVar("CoordinateFrame", bound=coord.BaseCoordinateFrame)

OptFrameType = T.Union[FrameType, None, EllipsisType]

SkyCoordType = T.TypeVar("SkyCoord", bound=coord.SkyCoord)

CoordinateType = T.Union[FrameType, SkyCoordType]

PositionType = T.Union[RepresentationType, CoordinateType]

GenericPositionType = T.Union[RepresentationOrDifferentialType, CoordinateType]

FrameLikeType = T.Union[CoordinateType, str]

OptFrameLikeType = T.Union[FrameLikeType, None, EllipsisType]

# -----------------
# table

TableType = T.TypeVar("Table", bound=table.Table)

QTableType = T.TypeVar("QTable", bound=table.QTable)

# -----------------
# units

UnitType = T.Union[
    T.TypeVar("Unit", bound=u.UnitBase),
    T.TypeVar("FunctionUnit", bound=u.FunctionUnitBase),
]

UnitLikeType = T.Union[UnitType, str]

QuantityType = T.TypeVar("Quantity", bound=u.Quantity)

QuantityLikeType = T.Union[QuantityType, str]

##############################################################################
# END
