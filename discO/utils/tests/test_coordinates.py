# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.utils`."""

__all__ = [
    "Test_resolve_framelike",
    "Test_resolve_representationlike",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import pytest

# PROJECT-SPECIFIC
from discO.config import conf
from discO.utils.coordinates import (
    resolve_framelike,
    resolve_representationlike,
)

##############################################################################
# TESTS
##############################################################################


class Test_resolve_framelike:
    """Test function :func:`~discO.utils.resolve_framelike`."""

    @staticmethod
    def test_frame_is_none():
        """Test when frame is None."""
        # basic usage
        assert resolve_framelike(frame=Ellipsis) == resolve_framelike(
            frame=conf.default_frame,
        )

        # test changes with conf
        with conf.set_temp("default_frame", "galactocentric"):

            assert resolve_framelike(frame=Ellipsis) == resolve_framelike(
                frame=conf.default_frame,
            )

    # /def

    @staticmethod
    def test_frame_is_str():
        """Test when frame is a string."""
        # basic usage
        assert resolve_framelike(frame="icrs") == coord.ICRS()

    # /def

    @staticmethod
    def test_frame_is_BaseCoordinateFrame():
        """Test when frame is a BaseCoordinateFrame."""
        # basic usage
        assert resolve_framelike(frame=coord.ICRS) == coord.ICRS()
        assert resolve_framelike(frame=coord.ICRS()) == coord.ICRS()

        # replicates without data
        c = coord.ICRS(ra=1 * u.deg, dec=2 * u.deg)
        assert resolve_framelike(frame=c) == coord.ICRS()

    # /def

    @staticmethod
    def test_frame_is_SkyCoord():
        """Test when frame is a SkyCoord."""
        c = coord.ICRS(ra=1 * u.deg, dec=2 * u.deg)
        sc = coord.SkyCoord(c)

        # basic usage
        assert resolve_framelike(frame=sc) == coord.ICRS()

    # /def

    @staticmethod
    def test_error_if_not_type():
        """Test when frame is not the right type."""
        # raise error if pass bad argument type
        with pytest.raises(TypeError):
            resolve_framelike(Exception, error_if_not_type=True)

        # check this is the default behavior
        with pytest.raises(TypeError):
            resolve_framelike(Exception)

        # check it doesn't error if
        assert (
            resolve_framelike(Exception, error_if_not_type=False) is Exception
        )

    # /def


# /class


# -------------------------------------------------------------------


class Test_resolve_representationlike:
    """Test function :func:`~discO.utils.resolve_representationlike`."""

    @staticmethod
    def test_representation_is_str():
        """Test when representation is a string."""
        # basic usage
        assert (
            resolve_representationlike(representation="cartesian")
            == coord.CartesianRepresentation
        )

    # /def

    @staticmethod
    def test_representation_is_BaseCoordinateFrame():
        """Test when representation is a BaseCoordinateFrame."""
        # basic usage
        assert (
            resolve_representationlike(
                representation=coord.CartesianRepresentation,
            )
            == coord.CartesianRepresentation
        )
        assert (
            resolve_representationlike(
                representation=coord.CartesianRepresentation(x=(1, 2, 3)),
            )
            == coord.CartesianRepresentation
        )

        # replicates without data
        c = coord.SphericalRepresentation(
            lon=1 * u.deg,
            lat=2 * u.deg,
            distance=3 * u.kpc,
        )
        assert (
            resolve_representationlike(representation=c)
            == coord.SphericalRepresentation
        )

    # /def

    @staticmethod
    def test_error_if_not_type():
        """Test when representation is not the right type."""
        # raise error if pass bad argument type
        with pytest.raises(TypeError):
            resolve_representationlike(Exception, error_if_not_type=True)

        # check this is the default behavior
        with pytest.raises(TypeError):
            resolve_representationlike(Exception)

        # check it doesn't error if
        assert (
            resolve_representationlike(Exception, error_if_not_type=False)
            is Exception
        )

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
