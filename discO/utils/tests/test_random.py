# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.utils.random`."""

__all__ = [
    "Test_NumpyRNGContext",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import numpy as np
import pytest

# PROJECT-SPECIFIC
from discO.tests.helper import ObjectTest
from discO.utils import random

##############################################################################
# PARAMETERS


##############################################################################
# TESTS
##############################################################################


class Test_NumpyRNGContext(ObjectTest, obj=random.NumpyRNGContext):

    #######################################################
    # Method tests

    def test___init__(self):
        """Test method ``__init__``."""
        # seed None
        obj = self.obj(None)
        assert obj.seed is None

        # int
        obj = self.obj(2)
        assert obj.seed == 2

        # RandomState
        obj = self.obj(np.random.RandomState(3))
        name1, state1, *rest1 = obj.seed.get_state()
        name2, state2, *rest2 = np.random.RandomState(3).get_state()

        assert name1 == name2
        assert all(np.equal(state1, state2))
        assert rest1 == rest2

        # Generator
        obj = self.obj(np.random.default_rng(3))
        assert (
            obj.seed.__getstate__() == np.random.default_rng(3).__getstate__()
        )

    # /def

    def test___enter__(self):
        """Test method ``__enter__``."""
        # seed None
        with self.obj(None):
            ns = np.random.rand(5)
            assert len(ns) == 5

        # int
        with self.obj(2):
            ns = np.random.rand(5)
            assert np.allclose(
                ns,
                np.array(
                    [0.4359949, 0.02592623, 0.54966248, 0.43532239, 0.4203678]
                ),
            )

        # RandomState
        with self.obj(np.random.RandomState(3)):
            ns = np.random.rand(5)
            assert np.allclose(
                ns,
                np.array(
                    [0.5507979, 0.70814782, 0.29090474, 0.51082761, 0.89294695]
                ),
            )

        # Generator
        with pytest.warns(UserWarning):
            with self.obj(np.random.default_rng(3)):
                ns = np.random.rand(5)
                assert len(ns) == 5

    # /def

    def test___exit__(self):
        """Test method ``__exit__``."""
        # tested in __enter__
        pass

    # /def

    #######################################################
    # Usage tests


# /class


# -------------------------------------------------------------------


##############################################################################
# END
