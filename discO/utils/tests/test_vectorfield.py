# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.utils.vectorfield`."""

__all__ = [
    "test_VECTORFIELD_CLASSES",
    "Test_BaseVectorField",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import pytest

# PROJECT-SPECIFIC
from discO.tests.helper import ObjectTest
from discO.utils import vectorfield

##############################################################################
# PARAMETERS


##############################################################################
# TESTS
##############################################################################


def test__VECTORFIELD_CLASSES():
    """Test :mod:`~discO.utils.VECTORFIELD_CLASSES`."""
    assert isinstance(vectorfield._VECTORFIELD_CLASSES, dict)

    for key, val in vectorfield._VECTORFIELD_CLASSES.items():
        assert isinstance(key, str)
        assert issubclass(val, vectorfield.BaseVectorField)


# /def

# -------------------------------------------------------------------


def test_VECTORFIELD_REPRESENTATIONS():
    """Test :mod:`~discO.utils.VECTORFIELD_REPRESENTATIONS`."""
    assert isinstance(vectorfield.VECTORFIELD_REPRESENTATIONS, dict)

    for key, val in vectorfield.VECTORFIELD_REPRESENTATIONS.items():
        assert issubclass(key, coord.BaseRepresentation)
        assert isinstance(val, object)


# /def

#####################################################################


class Test_BaseVectorField(ObjectTest, obj=vectorfield.BaseVectorField):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        # create a test vector field
        class TestVectorField(vectorfield.BaseVectorField):
            base_representation = coord.UnitSphericalRepresentation

        cls.klass = TestVectorField

        cls.points = coord.UnitSphericalRepresentation(1 * u.deg, 2 * u.deg)
        cls.kwargs = dict(vf_lon=2 * u.km / u.s, vf_lat=4 * u.km / u.s)
        cls.inst = TestVectorField(cls.points, **cls.kwargs)

    # /def

    # @classmethod
    # def teardown_class(cls):
    #     """Teardown fixtures for testing."""
    #     del cls.klass
    #     del cls.points
    #     del cls.kwargs
    #     del cls.inst

    #     vectorfield._VECTORFIELD_CLASSES.pop("testvectorfield")
    #     vectorfield.VECTORFIELD_REPRESENTATIONS.pop(
    #         coord.UnitSphericalRepresentation
    #     )

    # # /def

    #######################################################
    # Methods Tests

    def test___init_subclass__(self):
        """Test method ``__init_subclass__``.

        ``__init_subclass__`` also registers into ``_VECTORFIELD_CLASSES`` and
        ``VECTORFIELD_REPRESENTATIONS``. We will need to pop the tests after.

        """
        # since no ``attr_classes``, this is built from ``base_representation``
        assert self.klass.attr_classes.keys() == {
            "vf_" + c for c in self.klass.base_representation.attr_classes
        }
        assert list(self.klass.attr_classes.values()) == [
            u.Quantity for _ in self.klass.base_representation.attr_classes
        ]

        # -------------------
        # name
        assert self.klass.get_name() == "testvectorfield"

        # -------------------
        # an error is raised if the vectorfield name already exists
        with pytest.raises(ValueError) as e:

            class TestVectorField(vectorfield.BaseVectorField):
                base_representation = coord.UnitSphericalRepresentation

        assert "VectorField class 'testvectorfield' already exists." in str(
            e.value,
        )

        # -------------------
        # another error is raised if the vectorfield re-uses a Representation
        with pytest.raises(ValueError) as e:

            class FailedVectorField(vectorfield.BaseVectorField):
                base_representation = coord.UnitSphericalRepresentation

        # -------------------
        # check caches
        assert "testvectorfield" in vectorfield._VECTORFIELD_CLASSES
        assert (
            vectorfield._VECTORFIELD_CLASSES["testvectorfield"] is self.klass
        )
        assert (
            coord.UnitSphericalRepresentation
            in vectorfield.VECTORFIELD_REPRESENTATIONS
        )
        assert (
            vectorfield.VECTORFIELD_REPRESENTATIONS[
                coord.UnitSphericalRepresentation
            ]
            is self.klass
        )

        # -------------------
        # Check attributes

        for component in self.klass.attr_classes:
            assert hasattr(self.klass, component)

    # /def

    def test___init__(self):
        """Test method ``__init__``."""
        # frame
        inst = self.klass(self.points, **self.kwargs)
        assert inst.frame is None

        # points are converted
        inst = self.klass(
            self.points.represent_as(coord.CartesianRepresentation),
            frame="icrs",
            **self.kwargs
        )

        assert isinstance(inst.points, coord.UnitSphericalRepresentation)

        # resolve_frame
        with pytest.raises(TypeError):  # not instance
            self.klass(self.points, frame=coord.ICRS, **self.kwargs)

        inst = self.klass(self.points, frame=coord.ICRS(), **self.kwargs)
        assert inst.frame == coord.ICRS()

        inst = self.klass(self.points, frame="icrs", **self.kwargs)
        assert inst.frame == coord.ICRS()

        # -------------------
        # errors
        # if different units
        with pytest.raises(u.UnitsError) as e:
            self.klass(self.points, vf_lon=2 * u.km / u.s, vf_lat=4 * u.one)

        assert "components should have equivalent units." in str(e.value)

        # if not base-representation
        with pytest.raises(TypeError) as e:
            self.klass(object(), **self.kwargs)

        assert "points is not <BaseRepresentation>." in str(e.value)

    # /def

    def test_frame(self):
        """Test method ``frame``."""
        assert self.inst.frame is self.inst._frame
        assert self.inst.frame is None

    # /def

    def test_to_cartesian(self):
        """Test method ``to_cartesian``."""
        inst = self.inst.to_cartesian()

        assert isinstance(inst.points, coord.CartesianRepresentation)
        for component in inst.components:
            assert hasattr(inst, component)
            assert getattr(inst, component).unit == u.km / u.s

    # /def

    def test_from_cartesian(self):
        """Test method ``from_cartesian``."""
        intermediate = self.inst.to_cartesian()
        newinst = self.inst.from_cartesian(intermediate)

        # test equivalence to original
        for component in self.inst.points.components:
            assert u.allclose(
                getattr(newinst.points, component),
                getattr(self.inst.points, component),
            )
        for component in self.inst.components:
            assert u.allclose(
                getattr(newinst, component),
                getattr(self.inst, component),
            )

    # /def

    def test_represent_as(self):
        """Test method ``represent_as``."""
        # -------------------
        # other class is self

        inst = self.inst.represent_as(self.inst.__class__)
        assert inst is self.inst

        # -------------------
        # convert thru BaseVectorField

        inst = self.inst.represent_as(vectorfield.CartesianVectorField)

        assert isinstance(inst, vectorfield.CartesianVectorField)
        # TODO? more tests

        # -------------------
        # convert thru Representation

        inst = self.inst.represent_as(coord.CartesianRepresentation)

        assert isinstance(inst, vectorfield.CartesianVectorField)
        # TODO? more tests

        # -------------------
        # failed

        with pytest.raises(TypeError):
            self.inst.represent_as(object)
        # TODO? more tests

    # /def

    @pytest.mark.skip("TODO")
    def test_from_field(self):
        """Test method ``from_field``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test__scale_operation(self):
        """Test method ``_scale_operation``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test__combine_operation(self):
        """Test method ``_combine_operation``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test_norm(self):
        """Test method ``norm``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test_unit_vectors(self):
        """Test method ``unit_vectors``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test_scale_factors(self):
        """Test method ``scale_factors``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test___repr__(self):
        """Test method ``__repr__``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test__apply(self):
        """Test method ``_apply``."""
        assert False

    # /def

    #######################################################
    # Usage Tests


# /class


# -------------------------------------------------------------------


class Test_CartesianVectorField(
    Test_BaseVectorField,
    obj=vectorfield.CartesianVectorField,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        pass

    # /def

    #######################################################
    # Methods Tests

    @pytest.mark.skip("TODO")
    def test_attributes(self):
        """Test class attributes."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test___init__(self):
        """Test method ``__init__``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test_get_xyz(self):
        """Test method ``get_xyz``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test_get_vf_xyz(self):
        """Test method ``get_vf_xyz``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test_dot(self):
        """Test method ``dot``."""
        assert False

    # /def

    #######################################################
    # Usage Tests


# /class

# -------------------------------------------------------------------


class Test_CylindricalVectorField(
    Test_BaseVectorField,
    obj=vectorfield.CylindricalVectorField,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        pass

    # /def

    #######################################################
    # Methods Tests

    @pytest.mark.skip("TODO")
    def test_attributes(self):
        """Test class attributes."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test___init__(self):
        """Test method ``__init__``."""
        assert False

    # /def

    #######################################################
    # Usage Tests


# /class

# -------------------------------------------------------------------


class Test_SphericalVectorField(
    Test_BaseVectorField,
    obj=vectorfield.SphericalVectorField,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        pass

    # /def

    #######################################################
    # Methods Tests

    @pytest.mark.skip("TODO")
    def test_attributes(self):
        """Test class attributes."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test___init__(self):
        """Test method ``__init__``."""
        assert False

    # /def

    #######################################################
    # Usage Tests


# /class

# -------------------------------------------------------------------


class Test_PhysicsSphericalVectorField(
    Test_BaseVectorField,
    obj=vectorfield.PhysicsSphericalVectorField,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        pass

    # /def

    #######################################################
    # Methods Tests

    @pytest.mark.skip("TODO")
    def test_attributes(self):
        """Test class attributes."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test___init__(self):
        """Test method ``__init__``."""
        assert False

    # /def

    #######################################################
    # Usage Tests


# /class

##############################################################################
# END
