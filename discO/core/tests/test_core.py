# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.core.core`."""

__all__ = [
    "Test_CommonBase",
    "Test_PotentialWrapperMeta",
    "Test_PotentialWrapper",
]


##############################################################################
# IMPORTS

# BUILT-IN
from abc import abstractmethod
from collections.abc import Mapping
from types import MappingProxyType

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import pytest
from astropy.utils.introspection import resolve_name

# PROJECT-SPECIFIC
import discO
from discO.core import core
from discO.tests.helper import ObjectTest

##############################################################################
# TESTS
##############################################################################


class Test_CommonBase(ObjectTest, obj=core.CommonBase):

    #################################################################
    # Method Tests

    def test___init_subclass__(self):
        """Test subclassing."""
        # --------------------
        # When key is None
        class SubClasss1(self.obj):
            _registry = {}

        assert not hasattr(SubClasss1, "_key")

        # --------------------
        # When key is str

        class SubClasss2(self.obj, key="pytest"):
            _registry = {}

        assert SubClasss2._key == "pytest"

        # --------------------
        # test error

        with pytest.raises(TypeError):

            class SubClasss3(self.obj, key=Exception):
                _registry = {}

    # /def

    # -------------------------------

    @abstractmethod
    def test__registry(self):
        """Test method ``_registry``."""
        pass

    # /def

    def test_registry(self):
        # This doesn't run on `Test_CommonBase`, but should
        # run on all registry subclasses.
        if isinstance(self.obj._registry, Mapping):
            assert isinstance(self.obj.registry, MappingProxyType)

    # -------------------------------

    @abstractmethod
    def test___class_getitem__(self):
        """Test method ``__class_getitem__``."""
        # _registry can either be a property (for abstract base-classes)
        # or a Mapping, for normal classes.
        assert isinstance(self.obj._registry, (property, Mapping))

        # ---------

        # This doesn't run on `Test_CommonBase`, but should
        # run on all registry subclasses.
        if isinstance(self.obj._registry, Mapping):
            # a very basic equality test
            for k in self.obj._registry:
                # str
                assert self.obj[k] is self.obj._registry[k]

                # iterable of len = 1
                assert self.obj[[k]] is self.obj._registry[k]

                # multi-length iterable that fails
                with pytest.raises(KeyError):
                    self.obj[[k, KeyError]]

    # /def

    # -------------------------------

    @abstractmethod
    def test___init__(self):
        """Test method ``__init__``."""
        if self.obj is core.CommonBase:

            with pytest.raises(TypeError) as e:
                self.obj()

            assert "Can't instantiate abstract class" in str(e.value)

        # else: subclasses have to do their own.

    # /def

    # -------------------------------

    @abstractmethod
    def test___call__(self):
        """Test method ``__call__``."""
        pass

    # /def

    # -------------------------------

    def test__infer_package(self):
        """Test method ``_infer_package``."""
        # when key is None
        assert self.obj._infer_package(self.obj) == discO

        # when pass package
        # this overrides the package consideration
        assert self.obj._infer_package(None, package=pytest) == pytest

        # when pass package and it's a string
        # this overrides the package consideration
        assert self.obj._infer_package(None, package="pytest") == pytest

        # when package is None and object is c-compiled
        assert self.obj._infer_package(object(), package=None) == resolve_name(
            "builtins",
        )

        # when fails
        with pytest.raises(TypeError):
            self.obj._infer_package(object(), package=TypeError)

    # /def

    # -------------------------------

    def test__parse_registry_path(self):
        """Test method ``_parse_registry_path``."""
        # str -> str
        assert self.obj._parse_registry_path("pytest") == "pytest"

        # module -> str
        assert self.obj._parse_registry_path(pytest) == "pytest"

        # Sequence
        assert self.obj._parse_registry_path(("pytest", discO)) == [
            "pytest",
            "discO",
        ]

        # failure in Sequence
        with pytest.raises(TypeError):
            self.obj._parse_registry_path((None,))

        # failure in normal call
        with pytest.raises(TypeError):
            self.obj._parse_registry_path(None)

    #################################################################
    # Usage Tests

    # N/A b/c abstract base-class


# /class


#####################################################################


class Test_PotentialWrapperMeta(ObjectTest, obj=core.PotentialWrapperMeta):
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

        assert "<Representation, str, or None>" in str(e.value)

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
        # TypeError

        with pytest.raises(TypeError) as e:
            self.subclass._convert_to_frame(
                TypeError,
                coord.Galactocentric(),
            )

        assert "<SkyCoord, CoordinateFrame, or Representation>" in str(e.value)

    # /def

    def test_specific_potential(self):
        """Test method ``specific_force``."""
        with pytest.raises(NotImplementedError) as e:
            self.subclass.specific_potential(self.potential, self.points)

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

    #################################################################
    # Usage Tests


# /class


#####################################################################


class Test_PotentialWrapper(ObjectTest, obj=core.PotentialWrapper):
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
        core.WRAPPER_REGISTRY.pop(cls.subclass._key)

    # /def

    #################################################################
    # Method Tests

    def test___init_subclass__(self):
        """Test method ``__init_subclass__``."""
        # ---------------
        # None

        class SubClass1(self.obj, key=None):
            pass

        assert None not in core.WRAPPER_REGISTRY.keys()

        # ---------------
        # module

        class SubClass2(self.obj, key=pytest):
            pass

        assert "pytest" in core.WRAPPER_REGISTRY.keys()

        core.WRAPPER_REGISTRY.pop("pytest")  # cleanup

        # ---------------
        # string

        class SubClass3(self.obj, key="pytest"):
            pass

        assert "pytest" in core.WRAPPER_REGISTRY.keys()

        core.WRAPPER_REGISTRY.pop("pytest")  # cleanup

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
        if self.obj is not core.PotentialWrapper:
            wrapped = core.PotentialWrapper(self.potential)

            assert isinstance(wrapped, core.PotentialWrapper)
            assert isinstance(wrapped, self.obj)  # it's the subclass

    # /def

    def test___init__(self):
        """Test method ``__init__``."""
        # ---------------
        # basic

        obj = self.obj(2)

        assert obj.__wrapped__ == 2
        assert obj._frame is None

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
        assert self.inst.frame is self.inst._frame
        assert isinstance(self.inst.frame, coord.Galactocentric)

    # /def

    def test___call__(self):
        """Test method ``__call__``."""
        with pytest.raises(NotImplementedError):
            self.inst(self.points)

    # /def

    @abstractmethod
    def test_specific_potential(self):
        """Test method ``specific_potential``."""
        with pytest.raises(NotImplementedError):
            self.inst.specific_potential(self.points)

    # /def

    @abstractmethod
    def test_specific_force(self):
        """Test method ``specific_force``."""
        with pytest.raises(NotImplementedError):
            self.inst.specific_force(self.points)

    # /def

    @abstractmethod
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
