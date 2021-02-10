# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.plugin.agama` wrappers."""

__all__ = [
    "Test_AGAMAPotentialWrapperMeta",
    "Test_AGAMAPotentialWrapper",
]


##############################################################################
# IMPORTS

# BUILT-IN
from abc import abstractmethod

# THIRD PARTY
import agama
import astropy.coordinates as coord
import astropy.units as u
import pytest

# PROJECT-SPECIFIC
from discO.core.tests.test_core import (
    Test_PotentialWrapper as PotentialWrapper_Test,
)
from discO.core.tests.test_core import (
    Test_PotentialWrapperMeta as PotentialWrapperMeta_Test,
)
from discO.plugin import agama as plugin
from discO.utils import resolve_framelike, vectorfield

##############################################################################
# TESTS
##############################################################################


class Test_AGAMAPotentialWrapperMeta(
    PotentialWrapperMeta_Test,
    obj=plugin.AGAMAPotentialMeta,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        super().setup_class()

        # now agama stuff
        # override super
        agama.setUnits(mass=1, length=1, velocity=1)  # FIXME! bad
        cls.potential = agama.Potential(type="Plummer")

    # /def

    #################################################################
    # Method Tests

    def test_specific_potential(self):
        """Test method ``specific_force``."""
        # ---------------
        # basic

        points, values = self.subclass.specific_potential(
            self.potential,
            self.points,
        )

        # the points are unchanged
        assert points is self.points
        # check data types
        assert isinstance(points, coord.SkyCoord)
        assert isinstance(points.frame, self.frame)
        # and on the values
        assert isinstance(values, u.Quantity)
        assert values.unit == u.km ** 2 / u.s ** 2

        # TODO! test the specific values

        # ---------------
        # frame
        # test the different inputs

        for frame in (
            coord.Galactocentric,
            coord.Galactocentric(),
            "galactocentric",
        ):

            points, values = self.subclass.specific_potential(
                self.potential,
                self.points,
                frame=frame,
            )
            assert points is self.points
            assert isinstance(points, coord.SkyCoord)
            assert isinstance(points.frame, self.frame)
            assert isinstance(values, u.Quantity)
            assert values.unit == u.km ** 2 / u.s ** 2

            # TODO! test the specific values

        # ---------------
        # representation_type

        points, values = self.subclass.specific_potential(
            self.potential,
            self.points,
            representation_type=coord.CartesianRepresentation,
        )
        assert points is not self.points
        assert isinstance(points, coord.SkyCoord)
        assert isinstance(points.frame, self.frame)
        assert isinstance(points.data, coord.CartesianRepresentation)
        assert isinstance(values, u.Quantity)
        assert values.unit == u.km ** 2 / u.s ** 2

        # TODO! test the specific values

    # /def

    def test_specific_force(self):
        """Test method ``specific_force``."""
        # ---------------
        # basic

        vf = self.subclass.specific_force(self.potential, self.points)
        assert isinstance(vf, vectorfield.BaseVectorField)
        assert isinstance(vf.points, coord.CartesianRepresentation)
        assert hasattr(vf, "vf_x")
        assert hasattr(vf, "vf_y")
        assert hasattr(vf, "vf_z")
        assert vf.frame is None

        # TODO! test the specific values

        # ---------------
        # frame
        # test the different inputs

        for frame in (
            coord.Galactocentric,
            coord.Galactocentric(),
            "galactocentric",
        ):

            vf = self.subclass.specific_force(
                self.potential,
                self.points,
                frame=frame,
            )

            assert isinstance(vf, vectorfield.BaseVectorField)
            assert isinstance(vf.points, coord.CartesianRepresentation)
            assert hasattr(vf, "vf_x")
            assert hasattr(vf, "vf_y")
            assert hasattr(vf, "vf_z")
            assert isinstance(vf.frame, resolve_framelike(frame).__class__)

            # TODO! test the specific values

        # ---------------
        # representation_type

        vf = self.subclass.specific_force(
            self.potential,
            self.points,
            representation_type=coord.CylindricalRepresentation,
        )

        assert isinstance(vf, vectorfield.BaseVectorField)
        assert isinstance(vf.points, coord.CylindricalRepresentation)
        assert hasattr(vf, "vf_rho")
        assert hasattr(vf, "vf_phi")
        assert hasattr(vf, "vf_z")
        assert vf.frame is None

        # TODO! test the specific values

    # /def

    def test_acceleration(self):
        """Test method ``acceleration``."""
        assert self.subclass.acceleration == self.subclass.specific_force

    # /def

    #################################################################
    # Usage Tests


# /class


#####################################################################


class Test_AGAMAPotentialWrapper(
    PotentialWrapper_Test,
    obj=plugin.AGAMAPotentialWrapper,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        agama.setUnits(mass=1, length=1, velocity=1)  # FIXME! bad
        cls.potential = agama.Potential(type="Plummer")

        # set up the rest
        super().setup_class()

    # /def

    #################################################################
    # Method Tests

    def test_specific_potential(self):
        """Test method ``specific_force``."""
        # ---------------
        # basic

        points, values = self.inst.specific_potential(self.points)

        # the points are unchanged
        assert points is self.points
        # check data types
        assert isinstance(points, coord.SkyCoord)
        assert isinstance(points.frame, self.frame)
        # and on the values
        assert isinstance(values, u.Quantity)
        assert values.unit == u.km ** 2 / u.s ** 2

        # TODO! test the specific values

        # ---------------
        # can't pass frame
        # test the different inputs

        with pytest.raises(TypeError) as e:

            points, values = self.inst.specific_potential(
                self.points,
                frame=coord.Galactocentric(),
            )

        assert "multiple values for keyword argument 'frame'" in str(e.value)

        # ---------------
        # representation_type

        points, values = self.inst.specific_potential(
            self.points,
            representation_type=coord.CartesianRepresentation,
        )
        assert points is not self.points
        assert isinstance(points, coord.SkyCoord)
        assert isinstance(points.frame, self.frame)
        assert isinstance(points.data, coord.CartesianRepresentation)
        assert isinstance(values, u.Quantity)
        assert values.unit == u.km ** 2 / u.s ** 2

        # TODO! test the specific values

    # /def

    @abstractmethod
    def test_specific_force(self):
        """Test method ``specific_force``."""
        # ---------------
        # basic

        vf = self.inst.specific_force(self.points)
        assert isinstance(vf, vectorfield.BaseVectorField)
        assert isinstance(vf.points, coord.CartesianRepresentation)
        assert hasattr(vf, "vf_x")
        assert hasattr(vf, "vf_y")
        assert hasattr(vf, "vf_z")
        assert isinstance(vf.frame, coord.Galactocentric)

        # TODO! test the specific values

        # ---------------
        # frame
        # test the different inputs

        with pytest.raises(TypeError) as e:

            vf = self.inst.specific_force(
                self.points,
                frame=coord.Galactocentric(),
            )

        assert "multiple values for keyword argument 'frame'" in str(e.value)

        # TODO! test the specific values

        # ---------------
        # representation_type

        vf = self.inst.specific_force(
            self.points,
            representation_type=coord.CylindricalRepresentation,
        )

        assert isinstance(vf, vectorfield.BaseVectorField)
        assert isinstance(vf.points, coord.CylindricalRepresentation)
        assert hasattr(vf, "vf_rho")
        assert hasattr(vf, "vf_phi")
        assert hasattr(vf, "vf_z")
        assert isinstance(vf.frame, coord.Galactocentric)

        # TODO! test the specific values

    # /def

    @abstractmethod
    def test_acceleration(self):
        """Test method ``acceleration``."""
        assert self.subclass.acceleration == self.subclass.specific_force

    # /def


# /class


##############################################################################
# END