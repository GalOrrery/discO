# -*- coding: utf-8 -*-

"""Vector Field.

This is some documentation.


An example construction.

.. todo::

    store VECTORFIELD_REPRESENTATIONS keys as str, not the BaseRepresentation
    class.


"""

__all__ = [
    "BaseVectorField",
    "CartesianVectorField",
    "CylindricalVectorField",
    "SphericalVectorField",
    "PhysicsSphericalVectorField",
]


##############################################################################
# IMPORTS

# BUILT-IN
import functools
import inspect
import operator
import typing as T

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.coordinates.representation import (
    REPRESENTATION_CLASSES as _REP_CLSs,
)
from astropy.coordinates.representation import (
    BaseRepresentationOrDifferential,
    _array2string,
    _make_getter,
)
from erfa import ufunc as erfa_ufunc

# PROJECT-SPECIFIC
from .coordinates import resolve_framelike
from discO.type_hints import (
    FrameLikeType,
    FrameType,
    QuantityType,
    RepresentationType,
)

##############################################################################
# PARAMETERS

_VECTORFIELD_CLASSES: T.Dict[str, object] = {}
VECTORFIELD_REPRESENTATIONS: T.Dict[coord.BaseRepresentation, object] = {}


def _invalidate_psp_cls_hash():
    global _PSP_HASH
    _PSP_HASH = None


##############################################################################
# CODE
##############################################################################


class BaseVectorField(BaseRepresentationOrDifferential):
    """Base Vector-Field.

    Parameters
    ----------
    points : |Representation|
    *args
        The components
    frame : frame-like or None (optional, keyword only)
        The frame of the vector-field. None (default), does not attach a frame.
    **kwargs
        passed along

    """

    def __init_subclass__(cls, **kwargs) -> None:
        """Set default ``attr_classes`` and component getters on a VectorField.
        class BaseVectorField(BaseRepresentationOrDifferential):
        For these, the components are those of the base representation prefixed
        by 'd_', and the class is `~astropy.units.Quantity`.

        """
        if not hasattr(cls, "base_representation"):
            raise NotImplementedError(
                "VectorField representations must have a"
                '"base_representation" class attribute.',
            )

        # If not defined explicitly, create attr_classes.
        if not hasattr(cls, "attr_classes"):
            base_attr_classes = cls.base_representation.attr_classes
            cls.attr_classes = {
                "vf_" + c: u.Quantity for c in base_attr_classes
            }

        # Now check caches!
        repr_name = cls.get_name()
        if repr_name in _VECTORFIELD_CLASSES:
            raise ValueError(
                f"VectorField class '{repr_name}' already exists.",
            )
        elif cls.base_representation in VECTORFIELD_REPRESENTATIONS:
            raise ValueError(
                "VectorField with representation "
                f"'{cls.base_representation}' already exists.",
            )

        _VECTORFIELD_CLASSES[repr_name] = cls
        _invalidate_psp_cls_hash()

        # add to representations dict
        VECTORFIELD_REPRESENTATIONS[cls.base_representation] = cls

        # If not defined explicitly, create properties for the components.
        for component in cls.attr_classes:
            if not hasattr(cls, component):
                setattr(
                    cls,
                    component,
                    property(
                        _make_getter(component),
                        doc=f"Component '{component}' of the VectorField.",
                    ),
                )

        super().__init_subclass__(**kwargs)

    # /def

    def __init__(
        self,
        points: RepresentationType,
        *args,
        frame: T.Optional[FrameLikeType] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._frame = None if frame is None else resolve_framelike(frame)

        vf_q1 = getattr(self, "_" + self.components[0])
        vf_qs = [getattr(self, "_" + c) for c in self.components[1:]]

        if not all(vf_q1.unit.is_equivalent(vf_q.unit) for vf_q in vf_qs):
            raise u.UnitsError("components should have equivalent units.")

        if not isinstance(points, coord.BaseRepresentation):
            raise TypeError("points is not <BaseRepresentation>.")

        # TODO store in CoordinateFrame. If representation, use GenericFrame
        # "points" property and stuff links to the _points.data
        self._points = points.represent_as(self.base_representation)

    # /def

    @property
    def points(self):
        return self._points

    @property
    def frame(self) -> FrameType:
        return self._frame

    # /def

    #######################################################
    # Representation

    def to_cartesian(self):
        """Convert the field to 3D rectangular cartesian coordinates.

        Returns
        -------
        `CartesianVectorField`
            This object, converted

        """
        base_e = self.points.unit_vectors()
        c = functools.reduce(
            operator.add,
            (
                getattr(self, d_c) * base_e[c]
                for d_c, c in zip(self.components, self.points.components)
            ),
        )

        return CartesianVectorField(
            self.points.to_cartesian(),
            vf_x=c.x,
            vf_y=c.y,
            vf_z=c.z,
            frame=self.frame,
        )

    # /def

    @classmethod
    def from_cartesian(cls, other):
        """Convert field from 3D Cartesian coordinates to the desired class.

        Parameters
        ----------
        other : `CartesianVectorField`
            The object to convert into this vector field.

        Returns
        -------
        BaseVectorField
            A new Vector Field object that is this class' type.

        """
        points = cls.base_representation.from_cartesian(other.points)
        base_e = points.unit_vectors()

        return cls(
            points,
            *(other.dot(e) for e in base_e.values()),
            copy=False,
            frame=other.frame,
        )

    # /def

    def represent_as(self, other_class):
        """Convert coordinates to another representation.

        If the instance is of the requested class, it is returned unmodified.
        By default, conversion is done via cartesian coordinates.

        Parameters
        ----------
        other_class : `~BaseVectorField` subclass
            The type of representation to turn the coordinates into.

        """
        if other_class is self.__class__:
            return self

        # The default is to convert via cartesian coordinates.
        self_cartesian = self.to_cartesian()

        if inspect.isclass(other_class) and issubclass(
            other_class,
            BaseVectorField,
        ):
            pass
        elif inspect.isclass(other_class) and issubclass(
            other_class,
            coord.BaseRepresentation,
        ):
            # convert other_class to the corresponding VectorField
            other_class = VECTORFIELD_REPRESENTATIONS[other_class]
        elif isinstance(other_class, str):
            rep_cls = _REP_CLSs[other_class]
            # convert rep_cls to the corresponding VectorField
            other_class = VECTORFIELD_REPRESENTATIONS[rep_cls]
        else:
            raise TypeError

        return other_class.from_cartesian(self_cartesian)

    # /def

    @classmethod
    def from_field(cls, vectorfield):
        """Create a new instance of this vectorfield from another one.

        Parameters
        ----------
        vectorfield : `~BaseVectorField` instance
            The presentation that should be converted to this class.

        """
        if isinstance(vectorfield, BaseVectorField):
            cartesian = vectorfield.to_cartesian(
                # base.represent_as(vectorfield.base_representation)
            )
        else:
            raise TypeError

        return cls.from_cartesian(cartesian)

    # /def

    #######################################################
    # math

    def _scale_operation(self, op: T.Callable, *args):
        """Scale all components.

        Parameters
        ----------
        op : `~operator` callable
            Operator to apply (e.g., `~operator.mul`, `~operator.neg`, etc.
        *args
            Any arguments required for the operator (typically, what is to
            be multiplied with, divided by).

        """
        scaled_attrs = [op(getattr(self, c), *args) for c in self.components]
        scaled_points = self.points._scale_operation(op, *args)
        return self.__class__(
            scaled_points,
            *scaled_attrs,
            copy=False,
            frame=self.frame,
        )

    # /def

    def _combine_operation(self, op: T.Callable, other, reverse: bool = False):
        """Combine two vector fields.

        If ``other`` is of the same phase space position type as ``self``, the
        components will simply be combined.  If ``other`` is a representation,
        it will be used as a base for which to evaluate the phase space
        position, and the result is a new representation.

        Parameters
        ----------
        op : `~operator` callable
            Operator to apply (e.g., `~operator.add`, `~operator.sub`, etc.
        other : `BaseVectorField` instance
            The other phase space position or representation.
        reverse : bool
            Whether the operands should be reversed (e.g., as we got here via
            ``self.__rsub__`` because ``self`` is a subclass of ``other``).

        """
        # ----------
        # make sure points are the same

        diff = (
            self.points.represent_as(
                coord.CartesianRepresentation,
            )
            - other.points.represent_as(coord.CartesianRepresentation)
        )

        if not np.allclose(diff.norm().value, 0):
            raise Exception("can't combine mismatching points.")

        # ----------

        if isinstance(self, type(other)):

            first, second = (self, other) if not reverse else (other, self)
            return self.__class__(
                self.points,
                *[
                    op(getattr(first, c), getattr(second, c))
                    for c in self.components
                ],
                frame=self.frame,
            )
        else:
            try:
                self_cartesian = self.to_cartesian()
            except TypeError:
                return NotImplemented

            return other._combine_operation(op, self_cartesian, not reverse)

    # /def

    def norm(self) -> QuantityType:
        """Vector norm.

        The norm is the standard Frobenius norm, i.e., the square root of the
        sum of the squares of all components with non-angular units.

        Note that any associated differentials will be dropped during this
        operation.

        Returns
        -------
        norm : `astropy.units.Quantity`
            Vector norm, with the same shape as the representation.

        """
        return np.sqrt(
            functools.reduce(
                operator.add,
                (
                    getattr(self, component) ** 2
                    for component, cls in self.attr_classes.items()
                ),
            ),
        )

    # /def

    #######################################################
    # utils

    def unit_vectors(self) -> T.Dict[str, RepresentationType]:
        r"""Cartesian unit vectors in the direction of each component.

        Given unit vectors :math:`\hat{e}_c` and scale factors :math:`f_c`,
        a change in one component of :math:`\delta c` corresponds to a change
        in representation of :math:`\delta c \times f_c \times \hat{e}_c`.

        Returns
        -------
        unit_vectors : dict of `CartesianRepresentation`
            The keys are the component names.

        """
        return self.points.unit_vectors()

    # /def

    def scale_factors(self) -> T.Dict[str, QuantityType]:
        r"""Scale factors for each component's direction.

        Given unit vectors :math:`\hat{e}_c` and scale factors :math:`f_c`,
        a change in one component of :math:`\delta c` corresponds to a change
        in representation of :math:`\delta c \times f_c \times \hat{e}_c`.

        Returns
        -------
        scale_factors : dict of `~astropy.units.Quantity`
            The keys are the component names.

        """
        return self.points.scale_factors()

    # /def

    def __repr__(self) -> str:
        prefixstr = "    "
        # TODO combine with points
        arrstr = _array2string(
            np.lib.recfunctions.merge_arrays(
                (self.points._values, self._values),
            ),
            prefix=prefixstr,
        )

        pointsunitstr = (
            ("in " + self.points._unitstr)
            if self.points._unitstr
            else "[dimensionless]"
        )
        unitstr = (
            ("in " + self._unitstr) if self._unitstr else "[dimensionless]"
        )
        return "<{} ({}) {:s} | ({}) {:s}\n{}{}>".format(
            self.__class__.__name__,
            ", ".join(self.points.components),
            pointsunitstr,
            ", ".join(self.components),
            unitstr,
            prefixstr,
            arrstr,
        )

    # /def

    def _apply(self, method: T.Union[str, T.Callable], *args, **kwargs):
        """Create a new representation or differential with ``method`` applied
        to the component data.

        In typical usage, the method is any of the shape-changing methods for
        `~numpy.ndarray` (``reshape``, ``swapaxes``, etc.), as well as those
        picking particular elements (``__getitem__``, ``take``, etc.), which
        are all defined in `~astropy.utils.shapes.ShapedLikeNDArray`. It will be
        applied to the underlying arrays (e.g., ``x``, ``y``, and ``z`` for
        `~astropy.coordinates.CartesianRepresentation`), with the results used
        to create a new instance.

        Internally, it is also used to apply functions to the components
        (in particular, `~numpy.broadcast_to`).

        Parameters
        ----------
        method : str or callable
            If str, it is the name of a method that is applied to the internal
            ``components``. If callable, the function is applied.
        args : tuple
            Any positional arguments for ``method``.
        kwargs : dict
            Any keyword arguments for ``method``.

        """
        if callable(method):

            def apply_method(array):
                return method(array, *args, **kwargs)

        else:
            apply_method = operator.methodcaller(method, *args, **kwargs)

        new = super().__new__(self.__class__)
        new._points = self.points._apply(method, *args, **kwargs)
        for component in self.components:
            setattr(
                new,
                "_" + component,
                apply_method(getattr(self, component)),
            )

        # Copy other 'info' attr only if it has actually been defined.
        # See PR #3898 for further explanation and justification, along
        # with Quantity.__array_finalize__
        if "info" in self.__dict__:
            new.info = self.info

        return new

    # /def


# /class


# -------------------------------------------------------------------


class CartesianVectorField(BaseVectorField):
    """Cartesian Vector Field."""

    _xyz = None
    _vf_xyz = None

    base_representation = coord.CartesianRepresentation

    @property
    def x(self) -> QuantityType:
        return self.points.x

    @property
    def y(self) -> QuantityType:
        return self.points.y

    @property
    def z(self) -> QuantityType:
        return self.points.z

    def __init__(
        self,
        points: RepresentationType,
        vf_x,
        vf_y=None,
        vf_z=None,
        frame: T.Optional[FrameLikeType] = None,
        copy: bool = False,
    ) -> None:
        super().__init__(points, vf_x, vf_y, vf_z, frame=frame, copy=copy)

    # /def

    def get_xyz(self, xyz_axis: int = 0) -> QuantityType:
        """Return a vector array of the x, y, and z coordinates.

        Parameters
        ----------
        xyz_axis : int, optional
            The axis in the final array along which the x, y, z components
            should be stored (default: 0).

        Returns
        -------
        xyz : `~astropy.units.Quantity`
            With dimension 3 along ``xyz_axis``.  Note that, if possible,
            this will be a view.

        """
        return self.points.get_xyz(xyz_axis=xyz_axis)

    xyz = property(get_xyz)
    # /def

    def get_vf_xyz(self, vf_xyz_axis: int = 0):
        """Return a vector array of the vf_x, vf_y, and vf_z coordinates.

        Parameters
        ----------
        vf_xyz_axis : int, optional
            The axis in the final array along which the vf_x, vf_y, vf_z
            components should be stored (default: 0).

        Returns
        -------
        vf_xyz : `~astropy.units.Quantity`
            With dimension 3 along ``vf_xyz_axis``.  Note that, if possible,
            this will be a view.

        """
        if self._vf_xyz is not None:
            if self._vf_xyz_axis == vf_xyz_axis:
                return self._vf_xyz
            else:
                return np.moveaxis(
                    self._vf_xyz,
                    self._vf_xyz_axis,
                    vf_xyz_axis,
                )

        # Create combined array.  TO DO: keep it in _xyz for repeated use?
        # But then in-place changes have to cancel it. Likely best to
        # also update components.
        return np.stack([self._vf_x, self._vf_y, self._vf_z], axis=vf_xyz_axis)

    vf_xyz = property(get_vf_xyz)
    # /def

    def dot(self, other):
        """Dot product of two vector fields.

        Note that any associated differentials will be dropped during this
        operation.

        Parameters
        ----------
        other : `BaseVectorField` or |Representation|
            If not already cartesian, it is converted.

        Returns
        -------
        dot_product : `~astropy.units.Quantity`
            The sum of the product of the x, y, and z components of ``self``
            and ``other``.

        """
        try:
            other_c = other.to_cartesian()
        except Exception:
            raise TypeError(
                "cannot only take dot product with another "
                "vector field, not a {} instance.".format(type(other)),
            )

        if isinstance(other_c, BaseVectorField):
            other_vf_xyz = other_c.get_vf_xyz(vf_xyz_axis=-1)
        else:
            other_vf_xyz = other_c.get_xyz(xyz_axis=-1)

        # erfa pdp: p-vector inner (=scalar=dot) product.
        return erfa_ufunc.pdp(self.get_vf_xyz(vf_xyz_axis=-1), other_vf_xyz)

    # /def


# /class

# -------------------------------------------------------------------


class CylindricalVectorField(BaseVectorField):
    """Cylindrical Vector Field."""

    base_representation = coord.CylindricalRepresentation

    def __init__(
        self,
        points: RepresentationType,
        vf_rho,
        vf_phi=None,
        vf_z=None,
        frame: T.Optional[FrameLikeType] = None,
        copy: bool = False,
    ) -> None:
        super().__init__(points, vf_rho, vf_phi, vf_z, frame=frame, copy=copy)

    # /def

    @property
    def rho(self) -> QuantityType:
        return self.points.rho

    @property
    def phi(self) -> QuantityType:
        return self.points.phi

    @property
    def z(self) -> QuantityType:
        return self.points.z

    # /def


# /class

# -------------------------------------------------------------------


class SphericalVectorField(BaseVectorField):
    """Spherical Vector Field."""

    base_representation = coord.SphericalRepresentation

    def __init__(
        self,
        points: RepresentationType,
        vf_lon,
        vf_lat=None,
        vf_distance=None,
        frame: T.Optional[FrameLikeType] = None,
        copy: bool = False,
    ) -> None:
        super().__init__(
            points,
            vf_lon,
            vf_lat,
            vf_distance,
            frame=frame,
            copy=copy,
        )

    # /def

    @property
    def lon(self) -> QuantityType:
        return self.points.lon

    @property
    def lat(self) -> QuantityType:
        return self.points.lat

    @property
    def distance(self) -> QuantityType:
        return self.points.distance

    # /def


# /class

# -------------------------------------------------------------------


class PhysicsSphericalVectorField(BaseVectorField):
    """PhysicsSpherical Vector Field."""

    base_representation = coord.PhysicsSphericalRepresentation

    def __init__(
        self,
        points: RepresentationType,
        vf_phi,
        vf_theta=None,
        vf_r=None,
        frame: T.Optional[FrameLikeType] = None,
        copy: bool = False,
    ) -> None:
        super().__init__(
            points,
            vf_phi,
            vf_theta,
            vf_r,
            frame=frame,
            copy=copy,
        )

    # /def

    @property
    def phi(self) -> QuantityType:
        return self.points.phi

    @property
    def theta(self) -> QuantityType:
        return self.points.theta

    @property
    def r(self) -> QuantityType:
        return self.points.r

    # /def


# /class

##############################################################################
# END
