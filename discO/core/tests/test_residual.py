# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.core.residual`."""

__all__ = ["Test_ResidualMethod", "Test_GridResidual"]


##############################################################################
# IMPORTS

# BUILT-IN
from types import MappingProxyType

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest

# PROJECT-SPECIFIC
from discO.core import residual
from discO.core.tests.test_common import Test_CommonBase as CommonBase_Test
from discO.core.wrapper import PotentialWrapper

##############################################################################
# PYTEST


##############################################################################
# TESTS
##############################################################################


def test_RESIDUAL_REGISTRY():
    """Test constant :obj:`~discO.core.residual.RESIDUAL_REGISTRY`."""
    assert isinstance(residual.RESIDUAL_REGISTRY, dict)

    for key, val in residual.RESIDUAL_REGISTRY.items():
        assert issubclass(val, residual.ResidualMethod)
        assert residual.ResidualMethod[key] is val


# /def

#####################################################################


class Test_ResidualMethod(CommonBase_Test, obj=residual.ResidualMethod):
    """Docstring for ClassName."""

    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.representation_type = coord.CartesianRepresentation
        cls.observable = "acceleration"
        cls.kwargs = {}

        _r = np.linspace(0.1, 10, num=50)
        _lon = np.linspace(0, 360, num=10)
        _lat = np.linspace(-90, 90, num=10)
        r, lon, lat = np.meshgrid(_r, _lon, _lat)
        cls.points = coord.SphericalRepresentation(
            lon=lon * u.deg,
            lat=lat * u.deg,
            distance=r * u.kpc,
        )

        if cls.obj is residual.ResidualMethod:
            cls.original_potential = object()

            class SubClass(cls.obj):
                def evaluate_potential(self, *args, **kwargs):
                    return coord.CartesianRepresentation(x=1, y=2, z=3)

            cls.klass = SubClass
            # /class

            # have to go the long way around
            cls.inst = cls.klass(
                original_potential=cls.original_potential,
                observable=cls.observable,
                representation_type=cls.representation_type,
                **cls.kwargs
            )
            # need to assign points for run
            cls.inst.points = cls.points
        else:
            pass  # need to do in subclass

    # /def

    #################################################################
    # Method Tests

    def test___init_subclass__(self):
        """Test subclassing."""
        # --------------------
        # When key is None

        class SubClasss1(self.obj):
            pass

        assert hasattr(SubClasss1, "_key")
        assert SubClasss1._key is None

        # --------------------
        # When key is str

        class SubClasss2(self.obj, key="pytest"):
            pass

        assert SubClasss2._key == "pytest"

        # clean it up
        SubClasss2._registry.pop("pytest")

        # --------------------
        # test error

        with pytest.raises(TypeError):

            class SubClasss3(self.obj, key=Exception):
                pass

    # /def

    def test___new__(self):
        """Test method ``__new__``."""
        if self.obj is residual.ResidualMethod:
            with pytest.raises(ValueError, match="has no registered"):
                residual.ResidualMethod(method=None)

            # doesn't fail
            rm = residual.ResidualMethod(method="grid", grid=10)
            assert isinstance(rm, residual.GridResidual)

        else:
            with pytest.raises(ValueError, match="only on ResidualMethod."):
                self.obj(method="not in registry")

    # /def

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
            original_potential=self.original_potential,
            observable=self.observable,
            representation_type=representation_type,
            **self.kwargs
        )

        assert inst._observable is self.observable
        assert inst._default_params == self.kwargs
        assert isinstance(inst._original_potential, PotentialWrapper)
        assert inst._original_potential.wrapped is self.original_potential
        assert inst._original_potential.representation_type is expected

    # /def

    # -------------------------------

    def test_observable(self):
        """Test property ``observable``."""
        assert self.inst.observable is self.inst._observable

    # /def

    def test_default_params(self):
        """Test property ``default_params``."""
        assert isinstance(self.inst.default_params, MappingProxyType)
        assert self.inst.default_params == self.inst._default_params

    # /def

    def test_original_potential(self):
        """Test property ``original_potential``."""
        assert self.inst.original_potential is self.inst._original_potential

    # /def

    def test_frame(self):
        """Test property ``frame``."""
        assert self.inst.frame is self.inst.original_potential.frame

    # /def

    def test_representation_type(self):
        """Test property ``representation_type``."""
        assert (
            self.inst.representation_type
            is self.inst.original_potential.representation_type
        )

    # /def

    # -------------------------------

    def test_evaluate_potential(self):
        """Test method ``evaluate_potential``."""
        with pytest.raises(
            NotImplementedError,
            match="ppropriate subpackage.",
        ):
            self.obj.evaluate_potential(self.inst, self.original_potential)

        # evaluate_potential
        assert self.inst.evaluate_potential(
            self.original_potential,
        ) == coord.CartesianRepresentation(x=1, y=2, z=3)

    # /def

    # -------------------------------

    def test___call__no_observable(self):
        """Test method ``__call__`` without a set observable."""
        original = self.inst.observable
        self.inst._observable = None

        try:
            with pytest.raises(ValueError, match="`observable` not set"):
                self.inst(
                    fit_potential=self.original_potential,
                    observable=None,
                )

        except Exception:
            raise
        finally:
            self.inst._observable = original

    # /def

    def test___call__no_original_potential(self):
        """Test method ``__call__`` without a set original_potential."""
        original = self.inst.original_potential
        self.inst._original_potential = None

        try:
            with pytest.raises(
                ValueError,
                match="`original_potential` not set",
            ):
                self.inst(
                    fit_potential=self.original_potential,
                    original_potential=None,
                )

        except Exception:
            raise
        finally:
            self.inst._original_potential = original

    # /def

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

        assert resid == coord.CartesianRepresentation((0, 0, 0))

        # -------------------
        # passing in values

        resid = self.inst(
            fit_potential=self.original_potential,
            original_potential=self.original_potential,
            observable=self.observable,
            representation_type="spherical",
        )

        assert resid == coord.SphericalRepresentation(0 * u.rad, 0 * u.rad, 0)

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

        assert resid == coord.CartesianRepresentation((0, 0, 0))

        # -------------------
        # passing in values

        resid = self.inst.run(
            fit_potential=self.original_potential,
            original_potential=self.original_potential,
            observable=self.observable,
            representation_type="spherical",
            batch=True,
        )

        assert resid == coord.SphericalRepresentation(0 * u.rad, 0 * u.rad, 0)

    # /def


# /class


#####################################################################


class Test_GridResidual(Test_ResidualMethod, obj=residual.GridResidual):
    """Docstring for ClassName."""

    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        super().setup_class()

        if cls is Test_GridResidual:
            # TODO!! actual potential that properly evaluates
            cls.original_potential = object()

            class SubClass(cls.obj):
                def evaluate_potential(self, *args, **kwargs):
                    return coord.CartesianRepresentation(x=1, y=2, z=3)

            cls.klass = SubClass
            # /class

            cls.inst = cls.klass(
                grid=cls.points,
                original_potential=cls.original_potential,
                observable=cls.observable,
                representation_type=cls.representation_type,
                **cls.kwargs
            )

        else:
            pass  # need to do in subclass

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
        assert isinstance(inst._original_potential, PotentialWrapper)
        assert inst._original_potential.wrapped is self.original_potential
        assert inst._original_potential.representation_type is expected

    # /def

    # -------------------------------

    def test_evaluate_potential(self):
        """Test method ``evaluate_potential``."""
        # evaluate_potential
        assert self.inst.evaluate_potential(
            self.original_potential,
        ) == coord.CartesianRepresentation(x=1, y=2, z=3)

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
