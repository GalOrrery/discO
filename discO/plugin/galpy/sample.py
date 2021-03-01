# -*- coding: utf-8 -*-

""":mod:`~galpy` Potential Sampler."""

__all__ = [
    "GalpyPotentialSampler",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T

# THIRD PARTY
import astropy.coordinates as coord
import galpy.df as gdf
import numpy as np

# PROJECT-SPECIFIC
import discO.type_hints as TH
from .type_hints import PotentialType
from discO.core.sample import PotentialSampler
from discO.core.wrapper import PotentialWrapper
from discO.utils.random import RandomLike

##############################################################################
# CODE
##############################################################################

DF_REGISTRY = {
    "HernquistPotential": gdf.isotropicHernquistdf,
}

##############################################################################


class GalpyPotentialSampler(PotentialSampler, key="galpy"):
    """Sample a :mod:`~galpy` Potential.

    Parameters
    ----------
    pot : `~galpy.df.df` or `~galpy.potential.Potential` or `PotentialWrapper`
        Distribution Function holding the potential.
        If potential, ties to match to distribution function registry.

    frame : frame-like or None (optional, keyword-only)
        The preferred frame in which to sample.

    **kwargs
        Other parameters. Also needed to absorb option from ``__new__``.

    Other Parameters
    ----------------
    df : :class:`~galpy.df.df.df.df` class
        The DF class to use if `pot` is not already a DF.

    """

    def __init__(
        self,
        pot: T.Union[PotentialType, gdf.df.df],
        *,
        frame: TH.OptFrameLikeType = None,
        representation_type: TH.OptRepresentationLikeType = None,
        **kwargs
    ):
        # three input options:
        # 1) wrapped potential
        #    need to pop from wrapper
        if isinstance(pot, PotentialWrapper):
            frame = pot.frame if frame is None else frame
            representation_type = (
                pot.representation_type
                if representation_type is None
                else representation_type
            )
            pot = pot.__wrapped__
            df = None
        # 2) DF. need to pop potential from DF
        elif isinstance(pot, gdf.df.df):
            df = pot
            pot = pot._pot
        # 3) A raw potential object
        else:
            df = None

        # There was no DF in the potential. Either get from kwargs or infer.
        if df is None:
            df_cls = (
                kwargs.pop("df")  # get from kwargs...
                if "df" in kwargs  # if exists...
                else DF_REGISTRY[pot.__class__.__name__]  # else infer
            )
            df = df_cls(pot)

        # initialize & store DF
        super().__init__(
            pot,
            frame=frame,
            representation_type=representation_type,
            **kwargs,
        )
        self._df: gdf.df.df = df

    # /def

    #################################################################
    # Sampling

    def __call__(
        self,
        n: int = 1,
        *,
        frame: TH.OptFrameLikeType = None,
        representation_type: TH.OptRepresentationLikeType = None,
        total_mass: TH.QuantityType = None,
        random: RandomLike = None,
        **kwargs
    ):
        """Sample.

        Parameters
        ----------
        n : int (optional)
            number of samples
        frame : frame-like or None (optional)
            output frame of samples

        total_mass : |Quantity| or None (optional)
            overload the mass. Necessary if the potential has infinite mass.

        **kwargs
            ignored

        Returns
        -------
        :class:`~astropy.coordinates.SkyCoord`

        """
        # Get preferred frames
        frame = self._infer_frame(frame)
        representation_type = self._infer_representation(representation_type)

        # make sure physical is on
        self._df._pot.turn_physical_on()

        # can't pass a random seed, set in context
        with self._random_context(random):
            orbits = self._df.sample(
                R=None,
                z=None,
                phi=None,
                n=n,
                return_orbit=True,
            )

        t = orbits.time()
        dif = coord.CartesianDifferential(
            d_x=orbits.vx(t, use_physical=True),
            d_y=orbits.vy(t, use_physical=True),
            d_z=orbits.vz(t, use_physical=True),
        )
        rep = coord.CartesianRepresentation(
            x=orbits.x(t, use_physical=True),
            y=orbits.y(t, use_physical=True),
            z=orbits.z(t, use_physical=True),
            differentials=dict(s=dif),
        )

        if representation_type is None:
            representation_type = rep.__class__
        samples = coord.SkyCoord(
            frame.realize_frame(rep, representation_type=representation_type),
            copy=False,
        )

        # TODO! better storage of these properties, so stay when transform.
        samples.potential = self.potential

        total_mass = (
            total_mass
            if total_mass is not None
            else self._kwargs.pop("total_mass", None)
        )
        total_mass = (
            total_mass
            if total_mass is not None
            else self.potential.total_mass()
        )
        samples.mass = np.ones(n) * total_mass / n  # AGAMA compatibility

        return samples

    # /def


# /class

##############################################################################
# END
