# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.core.wrapper`."""

__all__ = [
    "Test_PotentialWrapperMeta",
    "Test_PotentialWrapper",
]


##############################################################################
# IMPORTS

# BUILT-IN
import abc

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import pytest

# PROJECT-SPECIFIC
from discO.core import wrapper
from discO.tests.helper import ObjectTest
from discO.utils.coordinates import UnFrame

##############################################################################
# TESTS
##############################################################################


class Test_PotentialWrapperMeta(ObjectTest, obj=wrapper.PotentialWrapperMeta):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        # test class
        class TestClass(metaclass=cls.obj):
            _frame = coord.Galactocentric()

        cls.subclass = TestClass

        cls.frame = coord.ICRS
        cls.rep = coord.SphericalRepresentation(
            lon=[0, 1, 2] * u.deg,
            lat=[3, 4, 5] * u.deg,
            distance=[6, 7, 8] * u.kpc,
        )
        cls.points = coord.SkyCoord(cls.frame(cls.rep), copy=False)

        # the potential
        cls.potential = object()

    # /def

    #################################################################
    # Method Tests

    def test__convert_to_frame(self):
        """Test method ``_convert_to_frame``."""
        # ---------------
        # points is not SkyCoord, BaseCoordinateFrame or BaseRepresentation

        with pytest.raises(TypeError) as e:
            self.subclass._convert_to_frame(self.points.data._values, None)

        assert "<SkyCoord, CoordinateFrame, or Representation>" in str(e.value)

        # ---------------
        # frame is None and points is SkyCoord or CoordinateFrame

        with pytest.raises(TypeError) as e:
            self.subclass._convert_to_frame(self.points, None)

        assert "the potential must have a frame." in str(e.value)

        with pytest.raises(TypeError) as e:
            self.subclass._convert_to_frame(self.points.frame, None)

        assert "the potential must have a frame." in str(e.value)

        # ---------------
        # representation_type is wrong

        with pytest.raises(TypeError) as e:
            self.subclass._convert_to_frame(
                self.points.data,
                None,
                representation_type=TypeError,
            )

        assert "Input representation must be" in str(e.value)

        # ---------------
        # frame is None and points is Representation
        # passes through unchanged

        points, from_frame = self.subclass._convert_to_frame(
            self.points.data,
            None,
        )
        assert points is self.points.data
        assert from_frame is None

        # ------------------------------

        for p in (self.points, self.points.frame, self.points.data):

            # ---------------
            # frame is CoordinateFrame / SkyCoord
            for frame in (
                coord.Galactocentric(),
                coord.Galactocentric,
                "galactocentric",
            ):
                points, from_frame = self.subclass._convert_to_frame(p, frame)

                # the points
                if isinstance(points, coord.SkyCoord):
                    assert isinstance(points.frame, coord.Galactocentric)
                elif isinstance(points, coord.BaseCoordinateFrame):
                    assert isinstance(points, coord.Galactocentric)
                else:
                    assert isinstance(points, coord.BaseRepresentation)
                    assert from_frame is None

                # the frame
                if hasattr(p, "frame"):
                    assert isinstance(from_frame, p.frame.__class__)
                elif isinstance(p, coord.BaseCoordinateFrame):
                    assert isinstance(from_frame, p.__class__)
                else:
                    assert from_frame is None

            # ---------------
            # representation

            for rep_type in (
                None,
                coord.PhysicsSphericalRepresentation,
                "physicsspherical",
            ):

                points, from_frame = self.subclass._convert_to_frame(
                    p,
                    "galactocentric",
                    representation_type=rep_type,
                )
                if rep_type is None:
                    expected = self.points.representation_type
                else:
                    expected = coord.PhysicsSphericalRepresentation

                assert isinstance(points.data, expected)

        # ---------------
        # same frame

        points, from_frame = self.subclass._convert_to_frame(
            self.points,
            frame=self.points.frame.replicate_without_data(),
        )

        # ---------------
        # TypeError

        with pytest.raises(TypeError) as e:
            self.subclass._convert_to_frame(
                TypeError,
                coord.Galactocentric(),
            )

        assert "<SkyCoord, CoordinateFrame, or Representation>" in str(e.value)

    # /def

    def test_total_mass(self):
        """Test method ``total_mass``."""
        with pytest.raises(NotImplementedError) as e:
            self.subclass.total_mass(self.potential)

        assert "Please use the appropriate subpackage." in str(e.value)

    # /def

    def test_potential(self):
        """Test method ``potential``."""
        with pytest.raises(NotImplementedError) as e:
            self.subclass.potential(self.potential, self.points)

        assert "Please use the appropriate subpackage." in str(e.value)

    # /def

    def test_specific_force(self):
        """Test method ``specific_force``."""
        with pytest.raises(NotImplementedError) as e:
            self.subclass.specific_force(self.potential, self.points)

        assert "Please use the appropriate subpackage." in str(e.value)

    # /def

    def test_acceleration(self):
        """Test method ``acceleration``."""
        with pytest.raises(NotImplementedError) as e:
            self.subclass.acceleration(self.potential, self.points)

        assert "Please use the appropriate subpackage." in str(e.value)

    # /def

    def test_coefficients(self):
        """Test method ``coefficients``."""
        with pytest.raises(NotImplementedError) as e:
            self.subclass.coefficients(self.potential)

        assert "Please use the appropriate subpackage." in str(e.value)

    # /def

    #################################################################
    # Usage Tests


# /class


#####################################################################


class Test_PotentialWrapper(ObjectTest, obj=wrapper.PotentialWrapper):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        # subclasses can define an attribute "potential" that is used by the
        # wrapper. Else, let's just make an object
        if not hasattr(cls, "potential"):
            cls.potential = object()

        cls.inst = cls.obj(cls.potential, frame="galactocentric")

        cls.frame = coord.ICRS
        cls.rep = coord.SphericalRepresentation(
            lon=[0, 1, 2] * u.deg,
            lat=[3, 4, 5] * u.deg,
            distance=[6, 7, 8] * u.kpc,
        )
        cls.points = coord.SkyCoord(cls.frame(cls.rep), copy=False)

        class SubClass(cls.obj, key=cls.obj.__name__):
            pass

        cls.subclass = SubClass

    # /def

    @classmethod
    def teardown_class(cls):
        """Teardown fixtures for testing."""
        wrapper.WRAPPER_REGISTRY.pop(cls.subclass._key)

    # /def

    #################################################################
    # Method Tests

    def test___init_subclass__(self):
        """Test method ``__init_subclass__``."""
        # ---------------
        # None

        class SubClass1(self.obj, key=None):
            pass

        assert None not in wrapper.WRAPPER_REGISTRY.keys()

        # ---------------
        # module

        class SubClass2(self.obj, key=pytest):
            pass

        assert "pytest" in wrapper.WRAPPER_REGISTRY.keys()

        wrapper.WRAPPER_REGISTRY.pop("pytest")  # cleanup

        # ---------------
        # string

        class SubClass3(self.obj, key="pytest"):
            pass

        assert "pytest" in wrapper.WRAPPER_REGISTRY.keys()

        wrapper.WRAPPER_REGISTRY.pop("pytest")  # cleanup

    # /def

    def test___class_getitem__(self):
        """Test method ``__class_getitem__``."""
        assert self.obj[self.subclass._key] is self.subclass

    # /def

    def test___new__(self):
        """Test method ``__new__``."""
        # when potential is right type if returns self
        assert isinstance(self.obj(self.potential), self.obj)

        # for tests run in the subclass of PotentialWrapper,
        # let's see if PotentialWrapper can correctly infer
        # the package.
        if self.obj is not wrapper.PotentialWrapper:
            wrapped = wrapper.PotentialWrapper(self.potential)

            assert isinstance(wrapped, wrapper.PotentialWrapper)
            assert isinstance(wrapped, self.obj)  # it's the subclass

        if wrapper.WRAPPER_REGISTRY:  # not empty
            potential = tuple(wrapper.WRAPPER_REGISTRY.values())[0]
            wrapped = wrapper.PotentialWrapper(potential)

            assert isinstance(wrapped, wrapper.PotentialWrapper)

    # /def

    def test___init__(self):
        """Test method ``__init__``."""
        # ---------------
        # basic

        obj = self.obj(2)

        assert obj.__wrapped__ == 2
        assert obj.frame == UnFrame()

        # ---------------
        # specify frame

        obj = self.obj(2, frame="galactocentric")

        assert obj.__wrapped__ == 2
        assert isinstance(obj._frame, coord.Galactocentric)

        # ---------------
        # on a wrapper

        obj = self.obj(obj, frame="galactocentric")

        assert obj.__wrapped__ == 2
        assert isinstance(obj._frame, coord.Galactocentric)

    # /def

    def test_frame(self):
        """Test method ``frame``."""
        obj = self.obj(self.potential, frame="galactocentric")

        assert obj.frame is obj._frame
        assert isinstance(obj.frame, coord.Galactocentric)

    # /def
    
    def test_default_representation(self):
        """Test method ``default_representation``."""
        obj = self.obj(self.potential, frame="galactocentric")

        assert obj.default_representation is obj._default_representation

    # /def

    def test___call__(self):
        """Test method ``__call__``."""
        with pytest.raises(NotImplementedError):
            self.inst(self.points)

    # /def

    @abc.abstractmethod
    def test_potential(self):
        """Test method ``potential``."""
        with pytest.raises(NotImplementedError):
            self.inst.potential(self.points)

    # /def

    @abc.abstractmethod
    def test_specific_force(self):
        """Test method ``specific_force``."""
        with pytest.raises(NotImplementedError):
            self.inst.specific_force(self.points)

    # /def

    @abc.abstractmethod
    def test_acceleration(self):
        """Test method ``acceleration``."""
        with pytest.raises(NotImplementedError):
            self.inst.acceleration(self.points)

    # /def

    def test__infer_key(self):
        """Test method ``_infer_key``."""
        assert self.inst._infer_key(self.points, None) == "astropy"
        assert self.inst._infer_key(None, pytest) == "pytest"
        assert self.inst._infer_key(None, "pytest") == "pytest"

        with pytest.raises(TypeError) as e:
            self.inst._infer_key(None, TypeError)

        assert "package must be <module, str, or None>" in str(e.value)

    # /def

    def test___repr__(self):
        """Test method ``__repr__``."""
        s = repr(self.inst)

        assert isinstance(s, str)
        assert self.inst.__class__.__name__ in s
        assert "potential :" in s
        assert "frame     :" in s

    # /def


# /class


##############################################################################
# END
