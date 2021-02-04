# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.utils.vectorfield`."""

__all__ = [
    "test__VECTORFIELD_CLASSES",
    "test_VECTORFIELD_REPRESENTATIONS",
    # ting classes
    "Test_BaseVectorField",
    "Test_CartesianVectorField",
    "Test_CylindricalVectorField",
    "Test_PhysicsSphericalVectorField",
    "Test_SphericalVectorField",
]


##############################################################################
# IMPORTS

# BUILT-IN
import operator

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
        cls.rep_cls = coord.UnitSphericalRepresentation

        # create a test vector field
        class TestVectorField(vectorfield.BaseVectorField):
            base_representation = cls.rep_cls

        cls.klass = TestVectorField
        cls.klass_name = cls.klass.__name__.lower()

        cls.points = cls.rep_cls(1 * u.deg, 2 * u.deg)
        cls.kwargs = dict(vf_lon=2 * u.km / u.s, vf_lat=4 * u.km / u.s)
        cls.inst = TestVectorField(cls.points, **cls.kwargs)

    # /def

    @classmethod
    def teardown_class(cls):
        """Teardown fixtures for testing."""
        # vectorfield._VECTORFIELD_CLASSES.pop(cls.klass_name, None)
        # vectorfield.VECTORFIELD_REPRESENTATIONS.pop(cls.rep_cls, None)

        del cls.klass
        del cls.points
        del cls.kwargs
        del cls.inst
        del cls.rep_cls
        del cls.klass_name

    # /def

    #######################################################
    # Methods Tests

    def test___init_subclass__(self):
        """Test method ``__init_subclass__``.

        ``__init_subclass__`` also registers into ``_VECTORFIELD_CLASSES`` and
        ``VECTORFIELD_REPRESENTATIONS``. We will need to pop the tests after.

        """
        # -------------------
        # failure

        with pytest.raises(NotImplementedError):

            class FailedTest(vectorfield.BaseVectorField):
                """no ``base_representation``."""

        # -------------------
        # testing a previous success from ``setup_class``.

        # since no ``attr_classes``, this is built from ``base_representation``
        assert self.klass.attr_classes.keys() == {
            "vf_" + c for c in self.klass.base_representation.attr_classes
        }
        assert list(self.klass.attr_classes.values()) == [
            u.Quantity for _ in self.klass.base_representation.attr_classes
        ]

        # -------------------
        # name
        assert self.klass.get_name() == self.klass_name

        # -------------------
        # an error is raised if the vectorfield name already exists
        with pytest.raises(ValueError) as e:

            type(
                self.klass_name,
                (self.obj,),
                dict(base_representation=self.rep_cls),
            )

        assert f"VectorField class '{self.klass_name}' already exists." in str(
            e.value,
        )

        # -------------------
        # another error is raised if the vectorfield re-uses a Representation
        with pytest.raises(ValueError) as e:

            class FailedVectorField(self.obj):
                base_representation = self.rep_cls

        # -------------------
        # check caches
        assert self.klass_name in vectorfield._VECTORFIELD_CLASSES
        assert vectorfield._VECTORFIELD_CLASSES[self.klass_name] is self.klass
        assert self.rep_cls in vectorfield.VECTORFIELD_REPRESENTATIONS
        assert (
            vectorfield.VECTORFIELD_REPRESENTATIONS[self.rep_cls] is self.klass
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
            **self.kwargs,
        )

        assert isinstance(inst.points, self.rep_cls)

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
            inp = [list(itm) for itm in self.kwargs.items()]
            inp[0][1] = 1 * u.one  # assign wrong unit

            self.klass(self.points, **dict(inp))

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
                atol=1e-15 * u.km / u.s,
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
        assert inst.points == self.inst.points.represent_as(
            coord.CartesianRepresentation,
        )
        # TODO! more tests, of the specific vectorfield values.

        # -------------------
        # convert thru Representation

        inst = self.inst.represent_as(coord.CartesianRepresentation)

        assert isinstance(inst, vectorfield.CartesianVectorField)
        assert inst.points == self.inst.points.represent_as(
            coord.CartesianRepresentation,
        )
        # TODO! more tests, of the specific vectorfield values.

        # -------------------
        # failed

        with pytest.raises(TypeError):
            self.inst.represent_as(object)

    # /def

    def test_from_field(self):
        """Test method ``from_field``.

        We will do this by round-trip.

        """
        intermediate = self.inst.represent_as(vectorfield.CartesianVectorField)
        inst = self.inst.from_field(intermediate)

        # test inst is same as self.inst
        for comp in self.inst.components:
            assert u.allclose(
                getattr(inst, comp),
                getattr(self.inst, comp),
                atol=1e-15 * u.km / u.s,
            )

        # -------------------
        # failed

        with pytest.raises(TypeError):
            self.inst.from_field(object())

    # /def

    def test__scale_operation(self):
        """Test method ``_scale_operation``.

        only the vectorfield values are scaled.

        """
        newinst = self.inst._scale_operation(operator.mul, 2)

        for comp in self.inst.components:
            assert getattr(newinst, comp) == 2 * getattr(self.inst, comp)

    # /def

    @pytest.mark.skip("TODO")
    def test__combine_operation(self):
        """Test method ``_combine_operation``."""
        assert False

    # /def

    def test_norm(self):
        """Test method ``norm``."""
        # sqrt(4^2 + 2^2)
        assert u.allclose(self.inst.norm() ** 2, 20 * u.km ** 2 / u.s ** 2)

    # /def

    def test_unit_vectors(self):
        """Test method ``unit_vectors``."""
        assert self.inst.unit_vectors() == self.inst.points.unit_vectors()

    # /def

    def test_scale_factors(self):
        """Test method ``scale_factors``."""
        assert self.inst.scale_factors() == self.inst.points.scale_factors()

    # /def

    def test___repr__(self):
        """Test method ``__repr__``."""
        s = self.inst.__repr__()

        assert isinstance(s, str)
        # TODO! more tests

    # /def

    def test__apply(self):
        """Test method ``_apply``.

        As the documentation implies, ``reshape`` uses ``_apply``.

        """
        newinst = self.inst.reshape(1, -1)
        assert newinst.shape == (1, 1)

        # TODO more tests

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
        cls.rep_cls = coord.CartesianRepresentation
        cls.klass = cls.obj
        cls.klass_name = "cartesianvectorfield"

        cls.points = cls.rep_cls(1 * u.kpc, 2 * u.kpc, 20 * u.pc)
        cls.kwargs = dict(
            vf_x=2 * u.km / u.s,
            vf_y=4 * u.km / u.s,
            vf_z=0 * u.km / u.s,
        )
        cls.inst = cls.klass(cls.points, **cls.kwargs)

    # /def

    #######################################################
    # Methods Tests

    def test_attributes(self):
        """Test class attributes."""
        assert self.klass.base_representation is coord.CartesianRepresentation

        assert self.inst.x == self.points.x
        assert self.inst.y == self.points.y
        assert self.inst.z == self.points.z

        assert u.allclose(self.inst.xyz, self.inst.points.xyz)
        assert u.allclose(self.inst.vf_xyz, self.inst.get_vf_xyz())

    # /def

    def test_get_xyz(self):
        """Test method ``get_xyz``."""
        assert u.allclose(self.inst.get_xyz(), self.inst.points.get_xyz())

    # /def

    def test_get_vf_xyz(self):
        """Test method ``get_vf_xyz``."""
        vf_xyz = self.inst.get_vf_xyz()
        assert u.allclose(vf_xyz, [2.0, 4.0, 0.0] * u.km / u.s)

        # specifying the axis
        vf_xyz = self.inst.get_vf_xyz(vf_xyz_axis=-1)
        assert u.allclose(vf_xyz, [2.0, 4.0, 0.0] * u.km / u.s)

        # TODO! tests for self._vf_xyz is not None

    # /def

    def test_dot(self):
        """Test method ``dot``."""
        # First, a failure
        with pytest.raises(TypeError):
            self.inst.dot(object())

        # now, a success
        val = self.inst.dot(self.inst)
        assert val == 20 * u.km ** 2 / u.s ** 2

        # and dot with a CartesianRepresentation
        for key, base_e in self.inst.points.unit_vectors().items():
            val = self.inst.dot(base_e)
            assert val == getattr(self.inst, "vf_" + key)

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
        cls.rep_cls = coord.CylindricalRepresentation
        cls.klass = cls.obj
        cls.klass_name = "cylindricalvectorfield"

        cls.points = cls.rep_cls(1 * u.kpc, 2 * u.deg, 20 * u.pc)
        cls.kwargs = dict(
            vf_rho=2 * u.km / u.s,
            vf_phi=4 * u.km / u.s,
            vf_z=0 * u.km / u.s,
        )
        cls.inst = cls.klass(cls.points, **cls.kwargs)

    # /def

    #######################################################
    # Methods Tests

    def test_attributes(self):
        """Test class attributes."""
        assert (
            self.klass.base_representation is coord.CylindricalRepresentation
        )

        assert self.inst.rho == self.points.rho
        assert self.inst.phi == self.points.phi
        assert self.inst.z == self.points.z

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
        cls.rep_cls = coord.SphericalRepresentation
        cls.klass = cls.obj
        cls.klass_name = "sphericalvectorfield"

        cls.points = cls.rep_cls(1 * u.deg, 2 * u.deg, 20 * u.pc)
        cls.kwargs = dict(
            vf_lon=2 * u.km / u.s,
            vf_lat=0 * u.km / u.s,
            vf_distance=4 * u.km / u.s,
        )
        cls.inst = cls.klass(cls.points, **cls.kwargs)

    # /def

    #######################################################
    # Methods Tests

    def test_attributes(self):
        """Test class attributes."""
        assert self.klass.base_representation is coord.SphericalRepresentation

        assert self.inst.lon == self.points.lon
        assert self.inst.lat == self.points.lat
        assert self.inst.distance == self.points.distance

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
        cls.rep_cls = coord.PhysicsSphericalRepresentation
        cls.klass = cls.obj
        cls.klass_name = "physicssphericalvectorfield"

        cls.points = cls.rep_cls(1 * u.deg, 2 * u.deg, 20 * u.pc)
        cls.kwargs = dict(
            vf_phi=2 * u.km / u.s,
            vf_theta=0 * u.km / u.s,
            vf_r=4 * u.km / u.s,
        )
        cls.inst = cls.klass(cls.points, **cls.kwargs)

    # /def

    #######################################################
    # Methods Tests

    def test_attributes(self):
        """Test class attributes."""
        assert (
            self.klass.base_representation
            is coord.PhysicsSphericalRepresentation
        )

        assert self.inst.phi == self.points.phi
        assert self.inst.theta == self.points.theta
        assert self.inst.r == self.points.r

    # /def

    #######################################################
    # Usage Tests


# /class

##############################################################################
# END
