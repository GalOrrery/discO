# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.core.common`."""

__all__ = [
    "Test_CommonBase",
]


##############################################################################
# IMPORTS

# BUILT-IN
from abc import abstractmethod
from collections.abc import Mapping
from types import MappingProxyType

# THIRD PARTY
import pytest
from astropy.utils.introspection import resolve_name

# PROJECT-SPECIFIC
import discO
from discO.core import common
from discO.tests.helper import ObjectTest

##############################################################################
# TESTS
##############################################################################


class Test_CommonBase(ObjectTest, obj=common.CommonBase):
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

    def test__in_registry(self):
        """Test method ``_in_registry``."""
        assert self.obj._in_registry("not in registry") is False
        assert self.obj._in_registry(["not", "in", "registry"]) is False

        if self.obj.registry is not None:
            for key in self.obj.registry.keys():
                assert self.obj._in_registry(key) is True

    # /def

    @abstractmethod
    def test__registry(self):
        """Test method ``_registry``."""
        if self.obj is common.CommonBase:
            assert isinstance(self.obj._registry, property)

        else:
            assert isinstance(self.obj._registry, Mapping)

    # /def

    def test_registry(self):
        # This doesn't run on `Test_CommonBase`, but should
        # run on all registry subclasses.
        if not isinstance(self.obj._registry, property):
            assert isinstance(self.obj.registry, MappingProxyType)

            for key, klass in self.obj.registry.items():
                assert issubclass(klass, self.obj), key

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
        if isinstance(self.obj.registry, Mapping):
            # a very basic equality test
            for k in self.obj.registry:
                # str
                assert self.obj[k] is self.obj.registry[k]

                # iterable of len = 1
                assert self.obj[[k]] is self.obj.registry[k]

                # multi-length iterable that fails
                with pytest.raises(KeyError):
                    self.obj[[k, KeyError]]

    # /def

    # -------------------------------

    @abstractmethod
    def test___init__(self):
        """Test method ``__init__``."""
        if self.obj is common.CommonBase:
            with pytest.raises(TypeError, match="instantiate abstract class"):
                self.obj()

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

##############################################################################
# END
