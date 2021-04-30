# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.core.measurement`."""

__all__ = [
    "Test_MeasurementErrorSampler",
    "Test_RVS_ContinuousMeasurementErrorSampler",
    "Test_GaussianMeasurementError",
]


##############################################################################
# IMPORTS

# BUILT-IN
from abc import abstractmethod
from collections.abc import Generator
from types import MappingProxyType

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest
import scipy.stats
from astropy.coordinates import concatenate

# PROJECT-SPECIFIC
from discO.core import measurement
from discO.core.tests.test_common import Test_CommonBase as CommonBase_Test
from discO.utils.coordinates import UnFrame

##############################################################################
# TESTS
##############################################################################


def test__MEASURE_REGISTRY():
    """Test the registry ``_MEASURE_REGISTRY``."""
    assert isinstance(measurement._MEASURE_REGISTRY, dict)
    assert all(
        [isinstance(key, str) for key in measurement._MEASURE_REGISTRY.keys()],
    )
    assert all(
        [
            issubclass(val, measurement.MeasurementErrorSampler)
            for val in measurement._MEASURE_REGISTRY.values()
        ],
    )


# /def


##############################################################################


class Test_MeasurementErrorSampler(
    CommonBase_Test,
    obj=measurement.MeasurementErrorSampler,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.c = coord.SkyCoord(
            coord.ICRS(ra=[1, 2] * u.deg, dec=[2, 3] * u.deg),
        )
        cls.c_err = coord.SkyCoord(
            coord.ICRS(ra=[0.1, 0.2] * u.deg, dec=[0.2, 0.3] * u.deg),
        )

        if cls.obj is measurement.MeasurementErrorSampler:

            class SubClass(cls.obj):
                def __call__(self, *args, **kwargs):
                    super().__call__(*args, **kwargs)

            # have to go the long way around
            cls.inst = SubClass(
                c_err=cls.c_err,
                frame=coord.ICRS(),
                representation_type=coord.SphericalRepresentation,
            )
        else:
            pass  # need to do in subclass

    # /def

    #################################################################
    # Method Tests

    def test___init_subclass__(self):
        """Test subclassing."""
        # can't run tests on super b/c doesn't accept "package"
        # super().test___init_subclass__()

        # -------------------------------
        try:
            # registered by name
            class SubClass1(self.obj):
                pass

            assert not hasattr(SubClass1, "_key")
            assert "subclass1" in measurement._MEASURE_REGISTRY  # lower case!
        except Exception:
            pass
        finally:  # cleanup
            measurement._MEASURE_REGISTRY.pop("subclass1", None)

        # -------------------------------
        # registered by key

        try:
            # registered by name
            class SubClass2(self.obj, key="SubClass2"):
                pass

            assert not hasattr(SubClass2, "_key")
            assert "SubClass2" in measurement._MEASURE_REGISTRY
        except Exception:
            pass
        finally:  # cleanup
            measurement._MEASURE_REGISTRY.pop("SubClass2", None)

        # -------------------------------
        # error when already in registry

        try:
            # registered by name
            class SubClass3(self.obj):
                pass

            assert issubclass(SubClass3, self.obj)
            # it's stupid, but need to use the class to avoid flake8 error

            # doing it again raises error
            with pytest.raises(KeyError):

                class SubClass3(self.obj):
                    pass

        except Exception:
            pass
        finally:  # cleanup
            measurement._MEASURE_REGISTRY.pop("subclass3", None)

    # /def

    # -------------------------------

    def test__registry(self):
        """Test method ``_registry``.

        As ``_registry`` is never overwritten in the subclasses, this test
        should carry though.

        """
        # run tests on super
        super().test__registry()

        # -------------------------------
        assert isinstance(self.obj._registry, MappingProxyType)

        # The GaussianMeasurementErrorSampler is already registered, so can
        # test for that.
        assert "Gaussian" in self.obj._registry
        assert (
            self.obj._registry["Gaussian"]
            is measurement.GaussianMeasurementError
        )

    # /def

    # -------------------------------

    def test___class_getitem__(self):
        """Test method ``__class_getitem__``."""
        # run tests on super
        super().test___class_getitem__()

        # -------------------------------
        # test a specific item in the registry
        if self.obj is measurement.MeasurementErrorSampler:

            assert self.obj["Gaussian"] is measurement.GaussianMeasurementError
            assert (
                self.obj["rvs", "Gaussian"]
                is measurement.GaussianMeasurementError
            )

            # not in own registry
            with pytest.raises(TypeError):
                self.obj[self.obj._key]

        else:
            assert False

    # /def

    # -------------------------------

    def test___new__(self):
        """Test method ``__new__``."""
        # ---------------
        # Need the "method" argument

        with pytest.raises(ValueError) as e:
            self.obj()

        assert (
            "MeasurementErrorSampler has no "
            "registered measurement resampler"
        ) in str(e.value)

        # ---------------
        # with method specified

        method, klass = tuple(self.obj._registry.items())[-1]

        msamp = self.obj(
            c_err=self.c_err,
            method=method,
            representation_type="cartesian",
        )

        # test class type
        assert isinstance(msamp, klass)
        assert isinstance(msamp, self.obj)

        # test inputs
        assert all(msamp.c_err == self.c_err)

    # /def

    # -------------------------------

    @abstractmethod
    def test___init__(self):
        """Test method ``__init__``."""
        # run tests on super
        super().test___init__()

        # --------------------------
        # default

        obj = self.obj(
            c_err=self.c_err,
            method="Gaussian",
            representation_type="cartesian",
        )
        obj.c_err == self.c_err
        assert isinstance(obj.frame, UnFrame)
        assert obj.representation_type is coord.CartesianRepresentation
        assert "method" not in obj.params

        # --------------------------
        # iter

        for frame, rep_type in (
            # instance
            (
                coord.Galactocentric(),
                coord.CartesianRepresentation((1, 2, 3)),
            ),
            # class
            (coord.Galactocentric, coord.CartesianRepresentation),
            # str
            ("galactocentric", "cartesian"),
        ):

            obj = self.obj(
                c_err=self.c_err,
                method="Gaussian",
                frame=frame,
                representation_type=rep_type,
            )
            assert obj.frame == coord.Galactocentric(), frame
            assert (
                obj.representation_type == coord.CartesianRepresentation
            ), rep_type
            assert "method" not in obj.params

    # /def

    # -------------------------------

    def test__fix_branch_cuts(self):
        """Test method ``_fix_branch_cuts``.

        .. todo::

            graphical proof via mpl_test that the point hasn't moved.

        """
        # -------------------------------
        # no angular units

        rep = coord.CartesianRepresentation(
            x=[1, 2] * u.kpc,
            y=[3, 4] * u.kpc,
            z=[5, 6] * u.kpc,
        )
        array = rep._values.view(dtype=np.float64).reshape(rep.shape[0], -1).T
        got = self.inst._fix_branch_cuts(array, rep.__class__, rep._units)

        assert got is array

        # -------------------------------
        # UnitSphericalRepresentation

        # 1) all good
        rep = coord.UnitSphericalRepresentation(
            lon=[1, 2] * u.deg,
            lat=[3, 4] * u.deg,
        )
        array = rep._values.view(dtype=np.float64).reshape(rep.shape[0], -1).T
        got = self.inst._fix_branch_cuts(
            array.copy(),
            rep.__class__,
            rep._units,
        )
        assert np.allclose(got, array)

        # 2) needs correction
        array = np.array([[-360, 0, 360], [-91, 0, 91]])
        got = self.inst._fix_branch_cuts(
            array.copy(),
            rep.__class__,
            rep._units,
        )
        assert np.allclose(got, np.array([[-180, 0, 540], [-89, 0, 89]]))

        # -------------------------------
        # SphericalRepresentation

        # 1) all good
        rep = coord.SphericalRepresentation(
            lon=[1, 2] * u.deg,
            lat=[3, 4] * u.deg,
            distance=[5, 6] * u.kpc,
        )
        array = rep._values.view(dtype=np.float64).reshape(rep.shape[0], -1).T
        got = self.inst._fix_branch_cuts(
            array.copy(),
            rep.__class__,
            rep._units,
        )
        assert np.allclose(got, array)

        # 2) needs correction
        array = np.array([[-360, 0, 360], [-91, 0, 91], [5, 6, 7]])
        got = self.inst._fix_branch_cuts(
            array.copy(),
            rep.__class__,
            rep._units,
        )
        assert np.allclose(
            got,
            np.array([[-180, 0, 540], [-89, 0, 89], [5, 6, 7]]),
        )

        # 3) needs correction
        array = np.array([[-360, 0, 360], [-91, 0, 91], [-5, 6, -7]])
        got = self.inst._fix_branch_cuts(
            array.copy(),
            rep.__class__,
            rep._units,
        )
        assert np.allclose(
            got,
            np.array([[0, 0, 720], [89, 0, -89], [5, 6, 7]]),
        )

        # -------------------------------
        # CylindricalRepresentation

        # 1) all good
        rep = coord.CylindricalRepresentation(
            rho=[5, 6] * u.kpc,
            phi=[1, 2] * u.deg,
            z=[3, 4] * u.parsec,
        )
        array = rep._values.view(dtype=np.float64).reshape(rep.shape[0], -1).T
        got = self.inst._fix_branch_cuts(
            array.copy(),
            rep.__class__,
            rep._units,
        )
        assert np.allclose(got, array)

        # 2) needs correction
        array = np.array([[-5, 6, -7], [-180, 0, 180], [-4, 4, 4]])
        got = self.inst._fix_branch_cuts(
            array.copy(),
            rep.__class__,
            rep._units,
        )
        assert np.allclose(got, np.array([[5, 6, 7], [0, 0, 360], [-4, 4, 4]]))

        # -------------------------------
        # NotImplementedError

        with pytest.raises(NotImplementedError):

            rep = coord.PhysicsSphericalRepresentation(
                phi=[1, 2] * u.deg,
                theta=[3, 4] * u.deg,
                r=[5, 6] * u.kpc,
            )
            array = (
                rep._values.view(dtype=np.float64).reshape(rep.shape[0], -1).T
            )
            self.inst._fix_branch_cuts(
                array.copy(),
                coord.PhysicsSphericalRepresentation,
                rep._units,
            )

    # /def

    # -------------------------------

    @abstractmethod
    def test___call__(self):
        """Test method ``__call__``. """
        # run tests on super
        super().test___call__()

        with pytest.raises(NotImplementedError):
            self.obj.__call__(self.obj, None)

    # /def

    @abstractmethod
    def test_run(self):
        """Test method ``run``.

        When Test_MeasurementErrorSampler this calls on the wrapped instance,
        which is GaussianMeasurementErrorSampler.

        Subclasses should do real tests on the output. This only tests
        that we can even call the method.

        .. todo::

            Tests in the subclasses that the results make sense.
            ie, follow the expected distribution

        """
        if self.obj is not measurement.MeasurementErrorSampler:
            assert False  # should never be here b/c subclasses override

        # ---------------
        # c_err = None

        with pytest.raises(NotImplementedError):
            tuple(self.inst.run(self.c, random=0))
            # the tuple is for ``_run_iter`` version

        # ---------------
        # random

        with pytest.raises(NotImplementedError):
            tuple(self.inst.run(self.c, random=1))
            # the tuple is for ``_run_iter`` version

        # ---------------
        # len(c.shape) == 1

        assert len(self.c.shape) == 1

        with pytest.raises(NotImplementedError):
            tuple(self.inst.run(self.c, self.c_err, random=0))
            # the tuple is for ``_run_iter`` version

        # ---------------
        # 2D array, SkyCoord, nerriter = 1

        c = concatenate([self.c, self.c]).reshape(len(self.c), -1)

        with pytest.raises(NotImplementedError):
            tuple(self.inst.run(c, c_err=self.c_err, random=0))
            # the tuple is for ``_run_iter`` version

        # ---------------
        # 2D array, SkyCoord, nerriter != niter

        c_err = concatenate(
            [self.c_err, self.c_err, self.c_err],
        ).reshape(len(self.c), -1)

        with pytest.raises(ValueError) as e:
            tuple(self.inst.run(c, c_err))
            # the tuple is for ``_run_iter`` version

        assert "c & c_err shape mismatch" in str(e.value)

        # ---------------
        # 2D array, SkyCoord, nerriter = niter

        c_err = concatenate([self.c_err, self.c_err]).reshape(len(self.c), -1)
        with pytest.raises(NotImplementedError):
            tuple(self.inst.run(c, c_err=c_err, random=0))
            # the tuple is for ``_run_iter`` version

        # ---------------
        # 2D array, (Mapping, scalar, callable, %-unit)

        with pytest.raises(NotImplementedError):
            tuple(self.inst.run(c, c_err=1 * u.percent, random=0))
            # the tuple is for ``_run_iter`` version

        # ---------------
        # 2D array, other

        with pytest.raises(NotImplementedError):
            tuple(self.inst.run(self.c, NotImplementedError()))
            # the tuple is for ``_run_iter`` version

        # ---------------
        # in different frame

        assert self.inst.frame != coord.Galactic()

        with pytest.raises(NotImplementedError):
            tuple(
                self.inst.run(
                    c,
                    c_err=10 * u.percent,
                    random=0,
                    frame=coord.Galactic(),
                ),
            )
            # the tuple is for ``_run_iter`` version

        # ---------------
        # & in different representation

        assert self.inst.frame != coord.Galactic()
        assert self.inst.representation_type != coord.CylindricalRepresentation

        with pytest.raises(NotImplementedError):
            tuple(
                self.inst.run(
                    c,
                    c_err=10 * u.percent,
                    random=0,
                    frame=coord.Galactic(),
                    representation_type=coord.CylindricalRepresentation,
                ),
            )
            # the tuple is for ``_run_iter`` version

    # /def

    def test__parse_c_err(self):
        """Test method ``_parse_c_err```."""
        expected_dpos = np.array([[0.1, 0.2, 1.0], [0.2, 0.3, 1.0]])

        # --------------------------
        # with c_err = None
        # c_err -> <SkyCoord (ICRS): (ra, dec) in deg
        #               [(0.1, 0.2), (0.2, 0.3)]>
        d_pos = self.inst._parse_c_err(None, self.c)

        assert np.allclose(d_pos, expected_dpos)

        # --------------------------
        # BaseCoordinateFrame, SkyCoord

        r_err = coord.SphericalRepresentation(
            (0.1, 0.2) * u.deg,
            (0.2, 0.3) * u.deg,
            1,
        )
        c_err = coord.ICRS(r_err)

        d_pos = self.inst._parse_c_err(c_err, self.c)
        assert np.allclose(d_pos, expected_dpos)

        # Now with the wrong representation type
        with pytest.raises(TypeError) as e:
            self.inst._parse_c_err(
                coord.ICRS(
                    r_err.to_cartesian(),
                    representation_type=coord.CartesianRepresentation,
                ),
                self.c,
            )

        assert (
            "`c` & `c_err` must have matching `representation_type`."
            in str(e.value)
        )

        # --------------------------
        # BaseRepresentation

        d_pos = self.inst._parse_c_err(r_err, self.c)
        assert np.allclose(d_pos, expected_dpos)

        # Now with the wrong representation type
        with pytest.raises(TypeError) as e:
            self.inst._parse_c_err(r_err.to_cartesian(), self.c)

        assert (
            "`c_err` must be the same Representation type as in `c`."
            in str(e.value)
        )

        # --------------------------
        # Mapping

        with pytest.raises(NotImplementedError):
            self.inst._parse_c_err({}, self.c)

        # --------------------------
        # percent error

        d_pos = self.inst._parse_c_err(10 * u.percent, self.c)
        assert np.allclose(d_pos, expected_dpos[:, :-1])

        # --------------------------
        # number

        d_pos = self.inst._parse_c_err(0.1, self.c)
        assert d_pos == 0.1

        # --------------------------
        # callable

        d_pos = self.inst._parse_c_err(lambda c: 0.1, self.c)
        assert d_pos == 0.1

        # --------------------------
        # unrecognized

        with pytest.raises(NotImplementedError) as e:
            self.inst._parse_c_err(NotImplementedError(), self.c)

        assert "is not yet supported." in str(e.value)

    # /def

    #################################################################
    # Pipeline Tests

    # N/A b/c abstract base-class


# /class

#####################################################################


class Test_RVS_ContinuousMeasurementErrorSampler(
    Test_MeasurementErrorSampler,
    obj=measurement.RVS_Continuous,
):
    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        super().setup_class()

        cls.inst = cls.obj(
            rvs=scipy.stats.norm,
            c_err=cls.c_err,
            frame=coord.ICRS(),
            representation_type=coord.SphericalRepresentation,
        )
        cls.rvs = scipy.stats.norm

    # /def

    #################################################################
    # Method Tests

    def test___class_getitem__(self):
        """Test method ``__class_getitem__``."""
        # -------------------------------
        # test a specific item in the registry
        if self.obj is measurement.RVS_Continuous:

            assert self.obj["Gaussian"] is measurement.GaussianMeasurementError

            with pytest.raises(KeyError):
                self.obj[self.obj._key]

        else:
            # work through each in a registry
            for key, val in self.obj.registry.items():
                assert self.obj[key] is val

    # /def

    # --------------------------------------------------------------

    def test___new__(self):
        """Test method ``__new__``.

        This is a wrapper class that acts differently when instantiating
        a RVS_Continuous than one of it's subclasses.

        """
        # ---------------
        # with method specified

        method, klass = tuple(self.obj.registry.items())[-1]
        msamp = self.obj(
            rvs=self.rvs,
            c_err=self.c_err,
            method=method,
            representation_type="cartesian",
        )

        # test class type
        assert isinstance(msamp, klass)
        assert isinstance(msamp, self.obj)

        # test inputs
        assert all(msamp.c_err == self.c_err)

        # ---------------
        # bad "method" argument

        with pytest.raises(ValueError) as e:
            self.obj(rvs=scipy.stats.norm, method="not in registry")

        assert (
            "RVS_Continuous has no registered measurement resampler"
        ) in str(e.value)

    # /def

    # --------------------------------------------------------------

    def test___init__(self):
        """Test method ``__init__``."""
        # --------------------------
        # default

        self.obj(
            rvs=self.rvs,
            c_err=self.c_err,
            method="Gaussian",
            representation_type="cartesian",
        )

        # --------------------------
        # explicitly None

        obj = self.obj(
            rvs=self.rvs,
            c_err=self.c_err,
            method="Gaussian",
            frame=None,
            representation_type="cartesian",
        )
        assert isinstance(obj.frame, UnFrame)
        assert obj.representation_type is coord.CartesianRepresentation
        assert "method" not in obj.params

        # --------------------------
        # iter

        for frame, rep_type in (
            # instance
            (
                coord.Galactocentric(),
                coord.CartesianRepresentation((1, 2, 3)),
            ),
            # class
            (coord.Galactocentric, coord.CartesianRepresentation),
            # str
            ("galactocentric", "cartesian"),
        ):

            obj = self.obj(
                rvs=self.rvs,
                c_err=self.c_err,
                method="Gaussian",
                frame=frame,
                representation_type=rep_type,
            )
            assert obj.frame == coord.Galactocentric(), frame
            assert (
                obj.representation_type == coord.CartesianRepresentation
            ), rep_type
            assert "method" not in obj.params

    # /def

    # --------------------------------------------------------------

    def test_rvs(self):
        """Test attribute ``rvs``."""
        assert self.inst.rvs == self.inst._rvs

    # /def

    # --------------------------------------------------------------

    @pytest.mark.parametrize("random", [0, "RandomState(0)"])
    @pytest.mark.parametrize(
        "c_err",
        [
            "self.c_err",
            "self.c_err.frame",  # BaseCoordinateFrame, not SkyCoord
            "self.c_err.represent_as(coord.SphericalRepresentation)",
        ],
    )
    def test___call__set_random(self, c_err, random):
        """Test method ``__call__``."""
        # test "random" by setting the seed
        # doing this here b/c want to control random for all the rest.

        expected_ra = [1.17640523, 2.44817864]
        expected_dec = [2.08003144, 3.5602674]
        expected_dist = [1.97873798, 0.02272212]

        random = (
            np.random.RandomState(0) if random == "RandomState(0)" else random
        )

        res = self.inst(self.c, c_err=eval(c_err), random=random)  # TODO!
        assert np.allclose(res.ra.deg, expected_ra)
        assert np.allclose(res.dec.deg, expected_dec)
        assert np.allclose(res.distance, expected_dist)

    def test___call__(self):
        """Test method ``__call__``."""
        # --------------------------
        # just "c", cannot predict "random"

        res = self.inst(self.c)
        assert res.frame.__class__ == self.c.frame.__class__  # same class

        # --------------------------
        # "c" and c_err, c_err is scalar | random

        expected_ra2 = [1.17640523, 2.22408932]
        expected_dec2 = [2.04001572, 3.1867558]
        expected_dist2 = [1.0978738, 0.90227221]

        res = self.inst(self.c, 0.1, random=0)
        assert np.allclose(res.ra.deg, expected_ra2)
        assert np.allclose(res.dec.deg, expected_dec2)
        assert np.allclose(res.distance, expected_dist2)

        # --------------------------
        # "c" and c_err, c_err is callable | random

        res = self.inst(self.c, lambda c: 0.1, random=0)
        assert np.allclose(res.ra.deg, expected_ra2)
        assert np.allclose(res.dec.deg, expected_dec2)
        assert np.allclose(res.distance, expected_dist2)

        # ------------------
        # :fun:`~discO.core.measurement.xpercenterror_factory`

        xpercenterror = measurement.xpercenterror_factory(10 * u.percent)
        expected_ra = [1.17640523, 2.44817864]
        expected_dec = [2.08003144, 3.5602674]
        expected_dist = [1.0978738, 0.90227221]

        res = self.inst(self.c, xpercenterror, random=0)
        assert np.allclose(res.ra.deg, expected_ra)
        assert np.allclose(res.dec.deg, expected_dec)
        assert np.allclose(res.distance, expected_dist)

        # ------------------
        # a raw percent scalar has the same effect

        res = self.inst(self.c, 10 * u.percent, random=0)
        assert np.allclose(res.ra.deg, expected_ra)
        assert np.allclose(res.dec.deg, expected_dec)
        assert np.allclose(res.distance, expected_dist)

        # --------------------------
        # "c" and c_err, c_err is Mapping | random

        with pytest.raises(NotImplementedError):
            self.inst(self.c, {}, random=0)

        # --------------------------
        # "c" and c_err, c_err is none of the above | random

        with pytest.raises(NotImplementedError):

            self.inst(self.c, c_err=Exception(), random=0)

    # /def

    # --------------------------------------------------------------

    @abstractmethod
    def test_run(self):
        """Test method ``run``.

        When Test_MeasurementErrorSampler this calls on the wrapped instance,
        which is GaussianMeasurementErrorSampler.

        Subclasses should do real tests on the output. This only tests
        that we can even call the method.

        .. todo::

            Tests in the subclasses that the results make sense.
            ie, follow the expected distribution

        """
        if self.obj is not measurement.RVS_Continuous:
            assert False  # should never be here b/c subclasses override

        # ---------------
        # c_err = None

        res = self.inst.run(self.c, random=0, batch=True)
        res = np.array(res) if isinstance(res, Generator) else res
        assert res.shape == self.c.shape
        assert np.allclose(res.ra.value, np.array([1.17640523, 2.44817864]))
        assert np.allclose(res.dec.value, np.array([2.08003144, 3.5602674]))
        # TODO! more tests

        # ---------------
        # random

        res2 = self.inst.run(self.c, random=1, batch=True)
        for c in res2.representation_component_names.keys():
            assert not np.allclose(getattr(res, c), getattr(res2, c))
        assert np.allclose(res2.ra.value, np.array([1.16243454, 181.78540628]))
        assert np.allclose(res2.dec.value, np.array([1.87764872, -3.25962229]))
        # TODO! more tests

        # ---------------
        # len(c.shape) == 1

        assert len(self.c.shape) == 1

        res = self.inst.run(self.c, self.c_err, random=0, batch=True)
        assert res.shape == self.c.shape
        assert np.allclose(res.ra.value, np.array([1.17640523, 2.44817864]))
        assert np.allclose(res.dec.value, np.array([2.08003144, 3.5602674]))
        # TODO! more tests

        # ---------------
        # 2D array, SkyCoord, nerriter = 1

        c = concatenate([self.c, self.c]).reshape(len(self.c), -1)

        res = self.inst.run(c, c_err=self.c_err, random=0, batch=True)
        assert res.shape == c.shape
        assert np.allclose(
            res.ra.value,
            np.array([[1.17640523, 1.44817864], [2.09500884, 2.0821197]]),
        )
        assert np.allclose(
            res.dec.value,
            np.array([[2.08003144, 2.5602674], [2.96972856, 3.04321307]]),
        )

        # ---------------
        # 2D array, SkyCoord, nerriter != niter

        c_err = concatenate(
            [self.c_err, self.c_err, self.c_err],
        ).reshape(len(self.c), -1)

        with pytest.raises(ValueError) as e:
            self.inst.run(c, c_err, batch=True)

        assert "c & c_err shape mismatch" in str(e.value)

        # ---------------
        # 2D array, SkyCoord, nerriter = niter

        c_err = concatenate([self.c_err, self.c_err]).reshape(
            len(self.c),
            -1,
        )
        res = self.inst.run(c, c_err=c_err, random=0, batch=True)
        assert res.shape == c.shape
        assert np.allclose(
            res.ra.value,
            np.array([[1.17640523, 1.22408932], [2.19001768, 2.0821197]]),
        )
        assert np.allclose(
            res.dec.value,
            np.array([[2.08003144, 2.3735116], [2.95459284, 3.04321307]]),
        )

        # ---------------
        # 2D array, (Mapping, scalar, callable, %-unit)

        res = self.inst.run(c, c_err=1 * u.percent, random=0, batch=True)
        assert res.shape == c.shape
        assert np.allclose(
            res.ra.value,
            np.array([[1.01764052, 1.02240893], [2.01900177, 2.00821197]]),
        )
        assert np.allclose(
            res.dec.value,
            np.array([[2.00800314, 2.03735116], [2.99545928, 3.00432131]]),
        )

        # ---------------
        # 2D array, other

        with pytest.raises(NotImplementedError) as e:
            self.inst.run(self.c, NotImplementedError(), batch=True)

        assert "not yet supported." in str(e.value)

    # /def


# /class

#####################################################################


class Test_GaussianMeasurementError(
    Test_RVS_ContinuousMeasurementErrorSampler,
    obj=measurement.GaussianMeasurementError,
):
    """docstring for Test_GaussianMeasurementError"""

    def test___new__(self):
        """Test method ``__new__``.

        This is a wrapper class that acts differently when instantiating
        a RVS_Continuous than one of it's subclasses.

        """
        # ---------------
        # Can't have the "method" argument

        with pytest.raises(ValueError) as e:
            self.obj(rvs=self.rvs, method="not None")

        assert "Can't specify 'method'" in str(e.value)

        # ---------------
        # AOK

        msamp = self.obj(
            rvs=self.rvs,
            c_err=self.c_err,
            method=None,
            representation_type="cartesian",
        )

        assert self.obj is not measurement.MeasurementErrorSampler
        assert isinstance(msamp, self.obj)
        assert isinstance(msamp, measurement.MeasurementErrorSampler)
        assert all(msamp.c_err == self.c_err)

    # /def

    # --------------------------------------------------------------

    def test___init__(self):
        """Test method ``__init__``."""
        # --------------------------
        # default

        self.obj(
            rvs=self.rvs, c_err=self.c_err, representation_type="cartesian"
        )

        # --------------------------
        # explicitly None

        obj = self.obj(
            rvs=self.rvs,
            c_err=self.c_err,
            frame=None,
            representation_type="cartesian",
        )
        assert isinstance(obj.frame, UnFrame)
        assert obj.representation_type is coord.CartesianRepresentation
        assert "method" not in obj.params

        # --------------------------
        # iter

        for frame, rep_type in (
            # instance
            (
                coord.Galactocentric(),
                coord.CartesianRepresentation((1, 2, 3)),
            ),
            # class
            (coord.Galactocentric, coord.CartesianRepresentation),
            # str
            ("galactocentric", "cartesian"),
        ):
            obj = self.obj(
                rvs=self.rvs,
                c_err=self.c_err,
                frame=frame,
                representation_type=rep_type,
            )
            assert obj.frame == coord.Galactocentric(), frame
            assert (
                obj.representation_type == coord.CartesianRepresentation
            ), rep_type
            assert "method" not in obj.params

    # /def

    # --------------------------------------------------------------

    @abstractmethod
    def test_run(self):
        """Test method ``run``.

        When Test_MeasurementErrorSampler this calls on the wrapped instance,
        which is GaussianMeasurementErrorSampler.

        Subclasses should do real tests on the output. This only tests
        that we can even call the method.

        .. todo::

            Tests in the subclasses that the results make sense.
            ie, follow the expected distribution

        """
        # ---------------
        # c_err = None

        res = self.inst.run(self.c, random=0, batch=True)
        assert res.shape == self.c.shape
        assert np.allclose(res.ra.value, np.array([1.17640523, 2.44817864]))
        assert np.allclose(res.dec.value, np.array([2.08003144, 3.5602674]))
        # TODO! more tests

        # ---------------
        # random

        res2 = self.inst.run(self.c, random=1, batch=True)
        for c in res2.representation_component_names.keys():
            assert not np.allclose(getattr(res, c), getattr(res2, c))
        assert np.allclose(res2.ra.value, np.array([1.16243454, 181.78540628]))
        assert np.allclose(res2.dec.value, np.array([1.87764872, -3.25962229]))
        # TODO! more tests

        # ---------------
        # len(c.shape) == 1

        assert len(self.c.shape) == 1

        res = self.inst.run(self.c, self.c_err, random=0, batch=True)
        assert res.shape == self.c.shape
        assert np.allclose(res.ra.value, np.array([1.17640523, 2.44817864]))
        assert np.allclose(res.dec.value, np.array([2.08003144, 3.5602674]))
        # TODO! more tests

        # ---------------
        # 2D array, SkyCoord, nerriter = 1

        c = concatenate([self.c, self.c]).reshape(len(self.c), -1)

        res = self.inst.run(c, c_err=self.c_err, random=0, batch=True)
        assert res.shape == c.shape
        assert np.allclose(
            res.ra.value,
            np.array([[1.17640523, 1.44817864], [2.09500884, 2.0821197]]),
        )
        assert np.allclose(
            res.dec.value,
            np.array([[2.08003144, 2.5602674], [2.96972856, 3.04321307]]),
        )

        # ---------------
        # 2D array, SkyCoord, nerriter != niter

        c_err = concatenate(
            [self.c_err, self.c_err, self.c_err],
        ).reshape(len(self.c), -1)

        with pytest.raises(ValueError) as e:
            self.inst.run(c, c_err, batch=True)

        assert "c & c_err shape mismatch" in str(e.value)

        # ---------------
        # 2D array, SkyCoord, nerriter = niter

        c_err = concatenate([self.c_err, self.c_err]).reshape(
            len(self.c),
            -1,
        )
        res = self.inst.run(c, c_err=c_err, random=0, batch=True)
        assert res.shape == c.shape
        assert np.allclose(
            res.ra.value,
            np.array([[1.17640523, 1.22408932], [2.19001768, 2.0821197]]),
        )
        assert np.allclose(
            res.dec.value,
            np.array([[2.08003144, 2.3735116], [2.95459284, 3.04321307]]),
        )

        # ---------------
        # 2D array, (Mapping, scalar, callable, %-unit)

        res = self.inst.run(c, c_err=1 * u.percent, random=0, batch=True)
        assert res.shape == c.shape
        assert np.allclose(
            res.ra.value,
            np.array([[1.01764052, 1.02240893], [2.01900177, 2.00821197]]),
        )
        assert np.allclose(
            res.dec.value,
            np.array([[2.00800314, 2.03735116], [2.99545928, 3.00432131]]),
        )

        # ---------------
        # 2D array, other

        with pytest.raises(NotImplementedError) as e:
            self.inst.run(self.c, NotImplementedError(), batch=True)

        assert "not yet supported." in str(e.value)

    # /def


# /class

##############################################################################
# Test Utils


def test_xpercenterror_factory():
    """Test :fun:`~discO.core.measurement.xpercenterror_factory`."""
    rep = coord.SphericalRepresentation(1 * u.rad, 1 * u.rad, 1 * u.kpc)
    crd = coord.ICRS(rep.reshape(-1, 1))

    # --------------------------
    # percent input

    func = measurement.xpercenterror_factory(10 * u.percent)
    res = func(crd)

    assert callable(func)
    assert np.allclose(res, [0.1, 0.1, 0.1])

    # --------------------------
    # fractional error input

    func2 = measurement.xpercenterror_factory(0.2)
    res2 = func2(crd)

    assert callable(func2)
    assert np.allclose(res2, [0.2, 0.2, 0.2])

    # --------------------------
    # caching

    assert measurement.xpercenterror_factory(10 * u.percent) is func
    assert measurement.xpercenterror_factory(0.2) is func2

    # --------------------------
    # docstring editing

    assert "10.0%" in func.__doc__
    assert "20.0%" in func2.__doc__


# /def


##############################################################################
# END
