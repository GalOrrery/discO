# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.plugin.agama.wrapper`."""

__all__ = [
    "Test_AGAMAPotentialWrapperMeta",
    "Test_AGAMAPotentialWrapper",
]


##############################################################################
# IMPORTS

# STDLIB
from abc import abstractmethod

# THIRD PARTY
import agama
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest

# LOCAL
from discO.core.tests.test_wrapper import Test_PotentialWrapper as PotentialWrapper_Test
from discO.core.tests.test_wrapper import Test_PotentialWrapperMeta as PotentialWrapperMeta_Test
from discO.plugin.agama import wrapper
from discO.utils import resolve_framelike, vectorfield

##############################################################################
# TESTS
##############################################################################


class Test_AGAMAPotentialWrapperMeta(
    PotentialWrapperMeta_Test,
    obj=wrapper.AGAMAPotentialMeta,
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

    def test_total_mass(self):
        """Test method ``total_mass``."""
        assert np.allclose(
            self.subclass.total_mass(self.potential),
            1 * u.solMass,
        )

    # /def

    def test_density(self):
        """Test method ``density``."""
        # ---------------
        # when there isn't a frame

        with pytest.raises(TypeError, match="must have a frame."):
            self.subclass.density(self.potential, self.points)

        # ---------------
        # basic

        points, values = self.subclass.density(
            self.potential,
            self.points.data,
        )

        # the points are unchanged
        assert points is self.points.data
        # check data types
        assert isinstance(points, coord.BaseRepresentation)
        # and on the values
        assert isinstance(values, u.Quantity)
        assert values.unit == u.solMass / u.pc ** 3

        # TODO! test the specific values

        # ---------------
        # frame
        # test the different inputs

        for frame in (
            coord.Galactocentric,
            coord.Galactocentric(),
            "galactocentric",
        ):

            points, values = self.subclass.density(
                self.potential,
                self.points,
                frame=frame,
            )
            assert isinstance(points, coord.SkyCoord)
            assert isinstance(points.frame, resolve_framelike(frame).__class__)
            assert isinstance(values, u.Quantity)
            assert values.unit == u.solMass / u.pc ** 3

            # TODO! test the specific values

        # ---------------
        # representation_type

        points, values = self.subclass.density(
            self.potential,
            self.points,
            frame=self.points.frame.replicate_without_data(),
            representation_type=coord.CartesianRepresentation,
        )
        assert isinstance(points, coord.SkyCoord)
        assert isinstance(points.frame, self.frame)
        assert isinstance(points.data, coord.CartesianRepresentation)
        assert isinstance(values, u.Quantity)
        assert values.unit == u.solMass / u.pc ** 3

        # TODO! test the specific values

    # /def

    def test_potential(self):
        """Test method ``potential``."""
        # ---------------
        # when there isn't a frame

        with pytest.raises(TypeError, match="must have a frame."):
            self.subclass.potential(self.potential, self.points)

        # ---------------
        # basic

        points, values = self.subclass.potential(
            self.potential,
            self.points.data,
        )

        # the points are unchanged
        assert points is self.points.data
        # check data types
        assert isinstance(points, coord.BaseRepresentation)
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

            points, values = self.subclass.potential(
                self.potential,
                self.points,
                frame=frame,
            )
            assert isinstance(points, coord.SkyCoord)
            assert isinstance(points.frame, resolve_framelike(frame).__class__)
            assert isinstance(values, u.Quantity)
            assert values.unit == u.km ** 2 / u.s ** 2

            # TODO! test the specific values

        # ---------------
        # representation_type

        points, values = self.subclass.potential(
            self.potential,
            self.points,
            frame=self.points.frame.replicate_without_data(),
            representation_type=coord.CartesianRepresentation,
        )
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
        # when there isn't a frame

        with pytest.raises(TypeError, match="must have a frame."):
            self.subclass.specific_force(self.potential, self.points)

        # ---------------
        # basic

        vf = self.subclass.specific_force(self.potential, self.points.data)
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
            frame=self.points.frame.replicate_without_data(),
            representation_type=coord.CylindricalRepresentation,
        )

        assert isinstance(vf, vectorfield.BaseVectorField)
        assert isinstance(vf.points, coord.CylindricalRepresentation)
        assert hasattr(vf, "vf_rho")
        assert hasattr(vf, "vf_phi")
        assert hasattr(vf, "vf_z")
        assert isinstance(vf.frame, self.frame)

        # TODO! test the specific values

    # /def

    def test_acceleration(self):
        """Test method ``acceleration``."""
        assert self.subclass.acceleration == self.subclass.specific_force

    # /def

    @pytest.mark.skip("TODO")
    def test_coefficients(self):
        """Test method ``coefficients``."""
        assert False

    # /def

    #################################################################
    # Usage Tests


# /class


#####################################################################


class Test_AGAMAPotentialWrapper(
    PotentialWrapper_Test,
    obj=wrapper.AGAMAPotentialWrapper,
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

    def test_total_mass(self):
        """Test method ``total_mass``."""
        assert np.allclose(self.inst.total_mass(), 1 * u.solMass)

    # /def

    def test_density(self):
        """Test method ``density``."""
        # ---------------
        # basic

        points, values = self.inst.density(self.points.data)

        # check data types
        assert isinstance(points, self.inst.frame.__class__)
        # and on the values
        assert isinstance(values, u.Quantity)
        assert values.unit == u.solMass / u.pc ** 3

        # TODO! test the specific values

        # ---------------
        # with a frame

        points, values = self.inst.density(self.points)

        # check data types
        assert isinstance(points, coord.SkyCoord)
        assert isinstance(points.frame, self.inst.frame.__class__)
        # and on the values
        assert isinstance(values, u.Quantity)
        assert values.unit == u.solMass / u.pc ** 3

        # TODO! test the specific values

        # ---------------
        # can't pass frame
        # test the different inputs

        with pytest.raises(TypeError, match="multiple values for keyword"):

            points, values = self.inst.density(
                self.points,
                frame=coord.Galactocentric(),
            )

        # ---------------
        # representation_type

        points, values = self.inst.density(
            self.points,
            representation_type=coord.CartesianRepresentation,
        )
        assert points is not self.points
        assert isinstance(points, coord.SkyCoord)
        assert isinstance(points.frame, self.inst.frame.__class__)
        assert isinstance(points.data, coord.CartesianRepresentation)
        assert isinstance(values, u.Quantity)
        assert values.unit == u.solMass / u.pc ** 3

        # TODO! test the specific values

    # /def

    def test_potential(self):
        """Test method ``potential``."""
        # ---------------
        # basic

        points, values = self.inst.potential(self.points.data)

        # check data types
        assert isinstance(points, self.inst.frame.__class__)
        # and on the values
        assert isinstance(values, u.Quantity)
        assert values.unit == u.km ** 2 / u.s ** 2

        # TODO! test the specific values

        # ---------------
        # with a frame

        points, values = self.inst.potential(self.points)

        # check data types
        assert isinstance(points, coord.SkyCoord)
        assert isinstance(points.frame, self.inst.frame.__class__)
        # and on the values
        assert isinstance(values, u.Quantity)
        assert values.unit == u.km ** 2 / u.s ** 2

        # TODO! test the specific values

        # ---------------
        # can't pass frame
        # test the different inputs

        with pytest.raises(TypeError, match="multiple values for keyword"):

            points, values = self.inst.potential(
                self.points,
                frame=coord.Galactocentric(),
            )

        # ---------------
        # representation_type

        points, values = self.inst.potential(
            self.points,
            representation_type=coord.CartesianRepresentation,
        )
        assert points is not self.points
        assert isinstance(points, coord.SkyCoord)
        assert isinstance(points.frame, self.inst.frame.__class__)
        assert isinstance(points.data, coord.CartesianRepresentation)
        assert isinstance(values, u.Quantity)
        assert values.unit == u.km ** 2 / u.s ** 2

        # TODO! test the specific values

    # /def

    def test___call__(self):
        """Test method ``__call__``."""
        # ---------------
        # basic

        points, values = self.inst(self.points.data)

        # check data types
        assert isinstance(points, self.inst.frame.__class__)
        # and on the values
        assert isinstance(values, u.Quantity)
        assert values.unit == u.km ** 2 / u.s ** 2

        # TODO! test the specific values

        # ---------------
        # with a frame

        points, values = self.inst(self.points)

        # check data types
        assert isinstance(points, coord.SkyCoord)
        assert isinstance(points.frame, self.inst.frame.__class__)
        # and on the values
        assert isinstance(values, u.Quantity)
        assert values.unit == u.km ** 2 / u.s ** 2

        # TODO! test the specific values

        # ---------------
        # can't pass frame
        # test the different inputs

        with pytest.raises(TypeError, match="multiple values for keyword"):

            points, values = self.inst(
                self.points,
                frame=coord.Galactocentric(),
            )

        # ---------------
        # representation_type

        points, values = self.inst(
            self.points,
            representation_type=coord.CartesianRepresentation,
        )
        assert points is not self.points
        assert isinstance(points, coord.SkyCoord)
        assert isinstance(points.frame, self.inst.frame.__class__)
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

        with pytest.raises(TypeError, match="multiple values for keyword"):

            vf = self.inst.specific_force(
                self.points,
                frame=coord.Galactocentric(),
            )

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

    @pytest.mark.skip("TODO")
    def test_coefficients(self):
        """Test method ``coefficients``."""
        assert False

    # /def


# /class


##############################################################################
# END
