# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.plugin.gala.wrapper`."""

__all__ = [
    "Test_GalaPotentialWrapperMeta",
    "Test_GalaPotentialWrapper",
]


##############################################################################
# IMPORTS

# BUILT-IN
from abc import abstractmethod

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import gala.potential as gpot
import numpy as np
import pytest
from gala.units import galactic

# PROJECT-SPECIFIC
from discO.core.tests.test_wrapper import (
    Test_PotentialWrapper as PotentialWrapper_Test,
)
from discO.core.tests.test_wrapper import (
    Test_PotentialWrapperMeta as PotentialWrapperMeta_Test,
)
from discO.plugin.gala import wrapper
from discO.utils import resolve_framelike, vectorfield

##############################################################################
# TESTS
##############################################################################


class Test_GalaPotentialWrapperMeta(
    PotentialWrapperMeta_Test,
    obj=wrapper.GalaPotentialMeta,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        super().setup_class()

        # now gala stuff
        # override super
        cls.potential = gpot.KeplerPotential(
            m=1 * u.solMass,
            units=galactic,
        )

    # /def

    #################################################################
    # Method Tests

    def test_total_mass(self):
        """Test method ``total_mass``."""
        # TODO! upstream fix
        assert np.isnan(self.subclass.total_mass(self.potential))
        # assert np.allclose(
        #     self.subclass.total_mass(self.potential),
        #     1 * u.solMass,
        # )

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
        assert u.allclose(values, [0.0, 0.0, 0.0] * u.solMass / u.pc**3)

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
            assert u.allclose(values, [0.0, 0.0, 0.0] * u.solMass / u.pc**3)

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
        assert u.allclose(values, [0.0, 0.0, 0.0] * u.solMass / u.pc**3)

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
        assert values.unit == u.kpc**2 / u.Myr**2

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
            assert values.unit == u.kpc**2 / u.Myr**2

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
        assert values.unit == u.kpc**2 / u.Myr**2

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
            representation_type=coord.CartesianRepresentation,
        )

        assert isinstance(vf, vectorfield.BaseVectorField)
        assert isinstance(vf.points, coord.CartesianRepresentation)
        assert hasattr(vf, "vf_x")
        assert hasattr(vf, "vf_y")
        assert hasattr(vf, "vf_z")
        assert isinstance(vf.frame, self.frame)

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


class Test_GalaPotentialWrapper(
    PotentialWrapper_Test,
    obj=wrapper.GalaPotentialWrapper,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.potential = gpot.KeplerPotential(
            m=1 * u.solMass,
            units=galactic,
        )

        # set up the rest
        super().setup_class()

        cls.inst = cls.obj(
            cls.potential,
            frame="galactocentric",
            representation_type="cylindrical",
        )

    # /def

    #################################################################
    # Method Tests

    def test_total_mass(self):
        """Test method ``total_mass``."""
        # TODO! upstream fix
        assert np.isnan(self.subclass.total_mass(self.potential))
        # assert np.allclose(self.inst.total_mass(), 1 * u.solMass)

    # /def

    def test_density(self):
        """Test method ``specific_force``."""
        # ---------------
        # basic

        points, values = self.inst.density(self.points.data)

        # check data types
        assert isinstance(points, self.inst.frame.__class__)
        # and on the values
        assert u.allclose(values, [0.0, 0.0, 0.0] * u.solMass / u.pc**3)

        # TODO! test the specific values

        # ---------------
        # with a frame

        points, values = self.inst.density(self.points)

        # check data types
        assert isinstance(points, coord.SkyCoord)
        assert isinstance(points.frame, self.inst.frame.__class__)
        # and on the values
        assert u.allclose(values, [0.0, 0.0, 0.0] * u.solMass / u.pc**3)

        # TODO! test the specific values

        # ---------------
        # can't pass frame
        # test the different inputs

        with pytest.raises(TypeError, match="keyword argument 'frame'"):
            points, values = self.inst.potential(
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
        assert u.allclose(values, [0.0, 0.0, 0.0] * u.solMass / u.pc**3)

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
        assert values.unit == u.kpc**2 / u.Myr**2

        # TODO! test the specific values

        # ---------------
        # with a frame

        points, values = self.inst.potential(self.points)

        # check data types
        assert isinstance(points, coord.SkyCoord)
        assert isinstance(points.frame, self.inst.frame.__class__)
        # and on the values
        assert isinstance(values, u.Quantity)
        assert values.unit == u.kpc**2 / u.Myr**2

        # TODO! test the specific values

        # ---------------
        # can't pass frame
        # test the different inputs

        with pytest.raises(TypeError, match="keyword argument 'frame'"):
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
        assert values.unit == u.kpc**2 / u.Myr**2

        # TODO! test the specific values

    # /def

    @abstractmethod
    def test___call__(self):
        """Test method ``specific_force``."""
        # ---------------
        # basic

        points, values = self.inst(self.points.data)

        # check data types
        assert isinstance(points, self.inst.frame.__class__)
        # and on the values
        assert isinstance(values, u.Quantity)
        assert values.unit == u.kpc**2 / u.Myr**2

        # ---------------
        # with a frame

        points, values = self.inst(self.points)

        # check data types
        assert isinstance(points, coord.SkyCoord)
        assert isinstance(points.frame, self.inst.frame.__class__)
        # and on the values
        assert isinstance(values, u.Quantity)
        assert values.unit == u.kpc**2 / u.Myr**2

        # TODO! test the specific values

        # ---------------
        # can't pass frame
        # test the different inputs

        with pytest.raises(TypeError, match="keyword argument 'frame'"):
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
        assert values.unit == u.kpc**2 / u.Myr**2

        # TODO! test the specific values

    # /def

    def test_specific_force(self):
        """Test method ``specific_force``."""
        # ---------------
        # basic

        vf = self.inst.specific_force(self.points)
        assert isinstance(vf, vectorfield.BaseVectorField)
        assert isinstance(vf.points, coord.CylindricalRepresentation)
        assert hasattr(vf, "vf_rho")
        assert hasattr(vf, "vf_phi")
        assert hasattr(vf, "vf_z")
        assert vf.vf_rho.unit == u.kpc / u.Myr**2
        assert isinstance(vf.frame, coord.Galactocentric)

        # TODO! test the specific values

        # ---------------
        # frame
        # test the different inputs

        with pytest.raises(TypeError, match="keyword argument 'frame'"):
            vf = self.inst.specific_force(
                self.points,
                frame=coord.Galactocentric(),
            )

        # TODO! test the specific values

        # ---------------
        # representation_type

        vf = self.inst.specific_force(
            self.points,
            representation_type=coord.CartesianRepresentation,
        )

        assert isinstance(vf, vectorfield.BaseVectorField)
        assert isinstance(vf.points, coord.CartesianRepresentation)
        assert hasattr(vf, "vf_x")
        assert hasattr(vf, "vf_y")
        assert hasattr(vf, "vf_z")
        assert vf.vf_x.unit == u.kpc / u.Myr**2
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
