# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.plugin.galpy.wrapper`."""

__all__ = [
    "Test_GalpyPotentialWrapperMeta",
    "Test_GalpyPotentialWrapper",
]


##############################################################################
# IMPORTS

# BUILT-IN
from abc import abstractmethod

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import galpy.potential as gpot
import numpy as np
import pytest

# PROJECT-SPECIFIC
from discO.core.tests.test_wrapper import (
    Test_PotentialWrapper as PotentialWrapper_Test,
)
from discO.core.tests.test_wrapper import (
    Test_PotentialWrapperMeta as PotentialWrapperMeta_Test,
)
from discO.plugin.galpy import wrapper
from discO.utils import resolve_framelike, vectorfield

##############################################################################
# TESTS
##############################################################################


class Test_GalpyPotentialWrapperMeta(
    PotentialWrapperMeta_Test,
    obj=wrapper.GalpyPotentialMeta,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        super().setup_class()

        # now galpy stuff
        # override super
        cls.potential = gpot.KeplerPotential(
            amp=1 * u.solMass,
            ro=8 * u.kpc,
            vo=220 * u.km / u.s,
        )
        cls.potential.turn_physical_on(ro=8 * u.kpc, vo=220 * u.km / u.s)

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

    def test_specific_potential(self):
        """Test method ``specific_force``."""
        # ---------------
        # when there isn't a frame

        with pytest.raises(TypeError) as e:
            self.subclass.specific_potential(self.potential, self.points)

        assert "the potential must have a frame." in str(e.value)

        # ---------------
        # basic

        points, values = self.subclass.specific_potential(
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

            points, values = self.subclass.specific_potential(
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

        points, values = self.subclass.specific_potential(
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

        with pytest.raises(TypeError) as e:
            self.subclass.specific_force(self.potential, self.points)

        assert "the potential must have a frame." in str(e.value)

        # ---------------
        # basic

        vf = self.subclass.specific_force(self.potential, self.points.data)
        assert isinstance(vf, vectorfield.BaseVectorField)
        assert isinstance(vf.points, coord.CylindricalRepresentation)
        assert hasattr(vf, "vf_rho")
        assert hasattr(vf, "vf_phi")
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
            assert isinstance(vf.points, coord.CylindricalRepresentation)
            assert hasattr(vf, "vf_rho")
            assert hasattr(vf, "vf_phi")
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

    def test_coefficients(self):
        """Test method ``coefficients``."""
        # No coefficients
        pot = gpot.KeplerPotential()
        assert self.subclass.coefficients(pot) is None

        # SCF
        pot = gpot.SCFPotential()
        coeffs = self.subclass.coefficients(pot)
        assert coeffs["type"] == "SCF"
        assert np.all(np.equal(coeffs["Acos"], pot._Acos))
        assert np.all(np.equal(coeffs["Asin"], pot._Asin))

        # DiskSCFPotential
        pot = gpot.DiskSCFPotential()
        coeffs = self.subclass.coefficients(pot)
        assert coeffs["type"] == "diskSCF"
        assert np.all(np.equal(coeffs["Acos"], pot._scf._Acos))
        assert np.all(np.equal(coeffs["Asin"], pot._scf._Asin))

    # /def

    #################################################################
    # Usage Tests


# /class


#####################################################################


class Test_GalpyPotentialWrapper(
    PotentialWrapper_Test,
    obj=wrapper.GalpyPotentialWrapper,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.potential = gpot.KeplerPotential(
            amp=1 * u.solMass,
            ro=8 * u.kpc,
            vo=220 * u.km / u.s,
        )
        cls.potential.turn_physical_on(ro=8 * u.kpc, vo=220 * u.km / u.s)

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
        assert np.allclose(self.inst.total_mass(), 1 * u.solMass)

    # /def

    def test_specific_potential(self):
        """Test method ``specific_force``."""
        # ---------------
        # basic

        points, values = self.inst.specific_potential(self.points.data)

        # check data types
        assert isinstance(points, self.inst.frame.__class__)
        # and on the values
        assert isinstance(values, u.Quantity)
        assert values.unit == u.km ** 2 / u.s ** 2

        # TODO! test the specific values

        # ---------------
        # with a frame

        points, values = self.inst.specific_potential(self.points)

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
        assert isinstance(points.frame, self.inst.frame.__class__)
        assert isinstance(points.data, coord.CartesianRepresentation)
        assert isinstance(values, u.Quantity)
        assert values.unit == u.km ** 2 / u.s ** 2

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
        assert values.unit == u.km ** 2 / u.s ** 2

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

        with pytest.raises(TypeError) as e:

            points, values = self.inst(
                self.points,
                frame=coord.Galactocentric(),
            )

        assert "multiple values for keyword argument 'frame'" in str(e.value)

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
        assert vf.vf_rho.unit == u.km / u.s ** 2
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
            representation_type=coord.CartesianRepresentation,
        )

        assert isinstance(vf, vectorfield.BaseVectorField)
        assert isinstance(vf.points, coord.CartesianRepresentation)
        assert hasattr(vf, "vf_x")
        assert hasattr(vf, "vf_y")
        assert hasattr(vf, "vf_z")
        assert vf.vf_x.unit == u.km / u.s ** 2
        assert isinstance(vf.frame, coord.Galactocentric)

        # TODO! test the specific values

    # /def

    @abstractmethod
    def test_acceleration(self):
        """Test method ``acceleration``."""
        assert self.subclass.acceleration == self.subclass.specific_force

    # /def

    def test_coefficients(self):
        """Test method ``coefficients``."""
        # No coefficients
        pot = gpot.KeplerPotential()
        inst = self.obj(pot)
        assert inst.coefficients() is None

        # SCF
        pot = gpot.SCFPotential()
        inst = self.obj(pot)
        coeffs = inst.coefficients()
        assert coeffs["type"] == "SCF"
        assert np.all(np.equal(coeffs["Acos"], pot._Acos))
        assert np.all(np.equal(coeffs["Asin"], pot._Asin))

        # DiskSCFPotential
        pot = gpot.DiskSCFPotential()
        inst = self.obj(pot)
        coeffs = inst.coefficients()
        assert coeffs["type"] == "diskSCF"
        assert np.all(np.equal(coeffs["Acos"], pot._scf._Acos))
        assert np.all(np.equal(coeffs["Asin"], pot._scf._Asin))

    # /def


# /class


##############################################################################
# END
