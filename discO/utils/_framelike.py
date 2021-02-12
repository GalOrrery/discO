# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Utilities."""

__all__ = [
    "resolve_framelike",
    "resolve_representationlike",
]


##############################################################################
# IMPORTS

# BUILT-IN
import inspect
import typing as T

# THIRD PARTY
from astropy.coordinates import (
    BaseCoordinateFrame,
    SkyCoord,
    sky_coordinate_parsers,
    BaseRepresentation,
)
from astropy.coordinates.representation import (
    REPRESENTATION_CLASSES as _REP_CLSs,
)

# PROJECT-SPECIFIC
from discO.config import conf
import discO.type_hints as TH

##############################################################################
# CODE
##############################################################################


def resolve_framelike(
    frame: T.Union[TH.FrameLikeType, None, T.Any],
    error_if_not_type: bool = True,
) -> T.Union[TH.FrameType, T.Any]:
    """Determine the frame and return a blank instance.

    Parameters
    ----------
    frame : frame-like instance or None
        If BaseCoordinateFrame, replicates without data.
        If str, uses astropy parsers to determine frame class
        If None (default), gets default frame name from config, and parses.

    error_if_not_type : bool
        Whether to raise TypeError if `frame` is not one of the allowed types.
        If False, pass through unchanged.

    Returns
    -------
    frame : `~astropy.coordinates.BaseCoordinateFrame` instance
        Replicated without data.

    """
    # If no frame is specified, assume that the input footprint is in a
    # frame specified in the configuration
    if frame is None:
        frame: str = conf.default_frame

    if isinstance(frame, str):
        frame = sky_coordinate_parsers._get_frame_class(frame.lower())()
    elif isinstance(frame, BaseCoordinateFrame):
        frame = frame.replicate_without_data()
    elif isinstance(frame, SkyCoord):
        frame = frame.frame.replicate_without_data()
    elif inspect.isclass(frame) and issubclass(frame, BaseCoordinateFrame):
        frame = frame()

    elif error_if_not_type:
        raise TypeError(
            "Input coordinate frame must be an astropy "
            "coordinates frame subclass *instance*, not a "
            "'{}'".format(frame.__class__.__name__),
        )

    return frame


# /def


def resolve_representationlike(
    representation: T.Union[TH.RepresentationLikeType, T.Any],
    error_if_not_type: bool = True,
) -> T.Union[TH.RepresentationType, T.Any]:
    """Determine the representation and return the class.

    Parameters
    ----------
    representation : |Representation| or str
        If Representation (instance or class), return the class.
        If str, uses astropy to determine class

    error_if_not_type : bool
        Whether to raise TypeError if `representation` is not one of the
        allowed types. If False, pass through unchanged.

    Returns
    -------
    frame : `~astropy.coordinates.BaseCoordinateFrame` instance
        Replicated without data.

    """
    if isinstance(representation, str):
        representation = _REP_CLSs[representation]
    elif isinstance(representation, BaseRepresentation):
        representation = representation.__class__
    elif inspect.isclass(representation) and issubclass(
        representation, BaseRepresentation
    ):
        pass

    elif error_if_not_type:
        raise TypeError(
            "Input representation must be an astropy representation subclass, "
            f"not a '{representation}'",
        )

    return representation


# /def


##############################################################################
# END
