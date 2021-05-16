# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Utilities."""

__all__ = [
    "resolve_framelike",
    "resolve_representationlike",
    #
    "UnFrame",
]


##############################################################################
# IMPORTS

# BUILT-IN
import inspect
import typing as T

# THIRD PARTY
from astropy.coordinates import (
    BaseCoordinateFrame,
    BaseRepresentation,
    SkyCoord,
)
from astropy.coordinates import representation as r
from astropy.coordinates import sky_coordinate_parsers

# PROJECT-SPECIFIC
import discO.type_hints as TH
from discO.config import conf

##############################################################################
# CODE
##############################################################################


class UnFrame(BaseCoordinateFrame):
    """Unconnected Coordinate Frame. Does not support transformations."""

    default_differential = None
    default_representation = None

    def __init__(
        self,
        *args,
        copy=True,
        representation_type=None,
        differential_type=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            copy=copy,
            representation_type=representation_type,
            differential_type=differential_type,
            **kwargs,
        )

        # set the representation and differential default type info from the
        # initialized values.
        if hasattr(self, "_data") and self._data is not None:
            self._default_representation = self._data.__class__
            self.representation_type = self._data.__class__

            if "s" in self.data.differentials:
                self.differential_type = self.data.differentials["s"].__class__

    # /def


# /class


##############################################################################


def resolve_framelike(
    frame: T.Union[TH.FrameLikeType, None, TH.EllipsisType, T.Any],
    error_if_not_type: bool = True,
) -> T.Union[TH.FrameType, T.Any]:
    """Determine the frame and return a blank instance.

    Parameters
    ----------
    frame : frame-like instance or None
        If BaseCoordinateFrame, replicates without data.
        If str, uses astropy parsers to determine frame class
        If None (default), return UnFrame.
        If Ellipsis, return default frame.

    error_if_not_type : bool
        Whether to raise TypeError if `frame` is not one of the allowed types.
        If False, pass through unchanged.

    Returns
    -------
    frame : `~astropy.coordinates.BaseCoordinateFrame` instance
        Replicated without data.

    """
    if frame is None:
        frame: TH.FrameType = UnFrame()
    elif frame is Ellipsis:
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
    representation: T.Union[TH.RepresentationLikeType, TH.EllipsisType, T.Any],
    error_if_not_type: bool = True,
) -> T.Union[TH.RepresentationType, T.Any]:
    """Determine the representation and return the class.

    Parameters
    ----------
    representation : |Representation| or str
        If Representation (instance or class), return the class.
        If str, uses astropy to determine class
        If Ellipsis, return default representation type

    error_if_not_type : bool
        Whether to raise TypeError if `representation` is not one of the
        allowed types. If False, pass through unchanged.

    Returns
    -------
    frame : `~astropy.coordinates.BaseCoordinateFrame` instance
        Replicated without data.

    """
    if representation is Ellipsis:
        representation = conf.default_representation_type

    if isinstance(representation, str):
        representation = r.REPRESENTATION_CLASSES[representation]
    elif isinstance(representation, BaseRepresentation):
        representation = representation.__class__
    elif inspect.isclass(representation) and issubclass(
        representation,
        BaseRepresentation,
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
