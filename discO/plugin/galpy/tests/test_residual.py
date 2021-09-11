# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.core.residual` with galpy potentials."""

__all__ = ["Test_GridResidual_Galpy"]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import galpy.potential as gpot
import numpy as np
import pytest

# LOCAL
from discO.core import residual
from discO.core.tests.test_residual import Test_GridResidual as GridResidual_Test
from discO.plugin.galpy.wrapper import GalpyPotentialWrapper
from discO.utils import vectorfield

##############################################################################
# PYTEST


##############################################################################
# TESTS
##############################################################################


class Test_GridResidual_Galpy(GridResidual_Test, obj=residual.GridResidual):
    """Docstring for ClassName."""

    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        super().setup_class()

        cls.representation_type = None  # Whatevs. Need to prevent Cartesian.

        # TODO!! actual potential that properly evaluates
        cls.original_potential = gpot.NFWPotential(amp=2e12 * u.solMass)
        cls.original_potential.turn_physical_on(ro=8 * u.kpc, vo=220 * u.km / u.s)

        cls.klass = cls.obj

        cls.inst = cls.klass(
            grid=cls.points,
            original_potential=cls.original_potential,
            observable=cls.observable,
            representation_type=cls.representation_type,
            **cls.kwargs
        )

    # /def

    #################################################################
    # Method Tests

    @pytest.mark.parametrize(
        "representation_type, expected",
        [
            (None, None),
            (Ellipsis, Ellipsis),
            (coord.CartesianRepresentation, coord.CartesianRepresentation),
            (coord.CylindricalRepresentation, coord.CylindricalRepresentation),
            ("cartesian", coord.CartesianRepresentation),
        ],
    )
    def test___init__(self, representation_type, expected):
        """Test method ``__init__``."""
        inst = self.klass(
            self.points,
            original_potential=self.original_potential,
            observable=self.observable,
            representation_type=representation_type,
            **self.kwargs
        )

        assert inst.points is self.points
        assert inst._observable is self.observable
        assert inst._default_params == self.kwargs
        assert isinstance(inst._original_potential, GalpyPotentialWrapper)
        assert inst._original_potential.wrapped is self.original_potential
        assert inst._original_potential.representation_type is expected

    # /def

    # -------------------------------

    def test_evaluate_potential(self):
        """Test method ``evaluate_potential``."""
        # evaluate_potential
        val = self.inst.evaluate_potential(self.original_potential)

        assert isinstance(val, vectorfield.CylindricalVectorField)
        assert u.allclose(
            val.represent_as(
                coord.CylindricalRepresentation,
            ).points._values.view(dtype=float),
            self.inst.points.represent_as(
                coord.CylindricalRepresentation,
            )._values.view(dtype=float),
        )
        # TODO! test expected value

    # /def

    # -------------------------------

    def test___call__(self):
        """Test method ``__call__``."""
        # -------------------
        # defaults

        resid = self.inst(
            fit_potential=self.original_potential,
            original_potential=None,
            observable=None,
            representation_type=None,
            **self.kwargs
        )

        assert isinstance(resid, vectorfield.CartesianVectorField)
        assert np.allclose(
            (resid.points - self.inst.points.to_cartesian()).norm(),
            0,
        )
        assert np.allclose(resid.norm(), 0)

        # -------------------
        # passing in values

        resid = self.inst(
            fit_potential=self.original_potential,
            original_potential=self.original_potential,
            observable=self.observable,
            representation_type="spherical",
        )

        assert isinstance(resid, vectorfield.SphericalVectorField)
        assert np.allclose(resid.to_cartesian().norm(), 0)

    # /def

    # -------------------------------

    def test_run(self):
        """Test method ``run``."""
        # -------------------
        # defaults

        resid = self.inst.run(
            fit_potential=self.original_potential,
            original_potential=None,
            observable=None,
            representation_type=None,
            batch=True,
            **self.kwargs
        )

        assert isinstance(resid, vectorfield.CartesianVectorField)
        assert np.allclose(
            (resid.points - self.inst.points.to_cartesian()).norm(),
            0,
        )
        assert np.allclose(resid.norm(), 0)

        # -------------------
        # passing in values

        resid = self.inst.run(
            fit_potential=self.original_potential,
            original_potential=self.original_potential,
            observable=self.observable,
            representation_type="spherical",
            batch=True,
        )

        assert isinstance(resid, vectorfield.SphericalVectorField)
        assert np.allclose(resid.to_cartesian().norm(), 0)

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
