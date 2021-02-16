# -*- coding: utf-8 -*-

"""Residuals."""


__all__ = [
    "ResidualMethod",
    "GridResidual",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T
from types import ModuleType

# THIRD PARTY
import astropy.coordinates as coord

# PROJECT-SPECIFIC
import discO.type_hints as TH
from .core import CommonBase
from .wrapper import PotentialWrapper
from discO.utils.coordinates import resolve_representationlike

##############################################################################
# PARAMETERS

RESIDUAL_REGISTRY: T.Dict[str, object] = dict()  # key : sampler

##############################################################################
# CODE
##############################################################################


class ResidualMethod(CommonBase):
    """Calculate Residual."""

    #################################################################
    # On the class

    _registry = RESIDUAL_REGISTRY

    def __init_subclass__(cls, key: T.Union[str, ModuleType, None] = None):
        """Initialize subclass, adding to registry by `package`.

        This method applies to all subclasses, not matter the
        inheritance depth, unless the MRO overrides.

        """
        super().__init_subclass__(key=key)

        if key is not None:  # same trigger as CommonBase
            # cls._package defined in super()
            cls.__bases__[0]._registry[cls._key] = cls

        # TODO? insist that subclasses define a evaluate method
        # this "abstractifies" the base-class even though it can be used
        # as a wrapper class.

    # /def

    def __call__(
        self,
        fit_pot: T.Any,
        original_pot: T.Any = None,
        observable: T.Optional[str] = None,
        *,
        representation_type: TH.OptRepresentationLikeType = None,
        **kwargs
    ):
        original_pot = original_pot or self.original_pot
        if original_pot is None:  # both passed and init are None
            raise ValueError("`original_pot` not set. Need to pass.")

        observable = observable or self.observable
        if observable is None:  # TODO get from config
            raise ValueError("`observable` not set. Need to pass.")

        origval = self.evaluate(
            original_pot,
            observable=observable,
            points=self.points,
            representation_type=coord.CartesianRepresentation,
            **kwargs
        )
        fitval = self.evaluate(
            fit_pot,
            observable=observable,
            points=self.points,
            representation_type=coord.CartesianRepresentation,
            **kwargs
        )

        # get difference
        # TODO! weighting by errors
        residual = fitval - origval

        # output representation type
        if representation_type is None:
            representation_type = residual.base_representation
        representation_type = resolve_representationlike(representation_type)

        return residual.represent_as(representation_type)

    # /def


# /class

##############################################################################


class GridResidual(ResidualMethod, key="grid"):
    """Residual on a grid.

    .. todo::

        - This is a bad way of getting galpy and agama
        - want to allow grid to be in a Frame and awesomely transform

    """

    def __init__(
        self,
        grid: TH.RepresentationType,
        original_pot: T.Optional[T.Any] = None,
        observable: str = "acceleration",  # TODO make None and have config
        representation_type: TH.OptRepresentationLikeType = None,
    ):
        self.points = grid
        self.original_pot = original_pot
        self.observable = observable
        self.representation_type = representation_type

    # /def

    def evaluate(
        self,
        potential: T.Any,
        observable: T.Optional[str] = None,
        points: T.Optional[TH.RepresentationType] = None,
        *,
        representation_type: TH.OptRepresentationLikeType = None,
        **kwargs
    ):
        """Evaluate residual.

        Parameters
        ----------
        potential : object
        observable : str
            method in `~PotentialWrapper`
        points : `~astropy.coordinates.BaseRepresentation` or None (optional)
        **kwargs

        Returns
        -------
        object

        """
        observable = observable or self.observable  # None -> stored
        if observable is None:  # still None
            raise ValueError("Need to pass observable.")

        if points is None:
            points = self.points

        # get class to evaluate
        key = self._infer_package(potential).__name__
        evaluator_cls = PotentialWrapper[key](potential)

        # get method from evaluator class
        evaluator = getattr(evaluator_cls, observable)

        # evaluate
        value = evaluator(
            points, representation_type=representation_type, **kwargs
        )

        # output representation type
        if representation_type is None:
            representation_type = value.base_representation
        representation_type = resolve_representationlike(representation_type)

        return value.represent_as(representation_type)

    # /def


# /class


##############################################################################
# END
