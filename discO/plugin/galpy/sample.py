# -*- coding: utf-8 -*-

""":mod:`~galpy` Potential Sampler."""

__all__ = [
    "GalpyPotentialSampler",
    "MeshGridPositionDF",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T

# THIRD PARTY
import astropy.coordinates as coord
import galpy.df as gdf
import numpy as np
from galpy.df.df import df as DF

# PROJECT-SPECIFIC
import discO.type_hints as TH
from .wrapper import GalpyPotentialWrapper
from discO.core.sample import MeshGridPotentialSampler, PotentialSampler
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
    potential : :class:`~discO.PotentialWrapper`
        Distribution Function holding the potential.
        If potential, ties to match to distribution function registry.

    df : :class:`~galpy.df.df.df.df` class
        The DF class to use.
    df_kwargs : Mapping or None
        kwargs into DF.

    representation_type: |Representation| or None (optional, keyword-only)
        The coordinate representation in which to return samples.
        If None (default) uses representation type from `potential`.

    **defaults
        default arguments for sampling parameters. In ``run``, parameters with
        default `None` will draw from these defaults.

    """

    def __init__(
        self,
        potential: PotentialWrapper,
        df: T.Optional[gdf.df.df] = None,
        df_kwargs: T.Optional[T.Mapping] = None,
        *,
        total_mass: T.Optional[TH.QuantityType] = None,
        representation_type: TH.OptRepresentationLikeType = None,
        **defaults
    ):
        # infer DF class if None
        if df is None:
            df = DF_REGISTRY[potential.wrapped.__class__.__name__]

        # create DF instance
        df = df(potential.wrapped, **(df_kwargs or {}))

        # initialize & store DF
        super().__init__(
            potential,
            representation_type=representation_type,
            total_mass=total_mass,
            **defaults
        )
        self._df: gdf.df.df = df

        # make sure physical is on  # TODO enfore more strictly
        getattr(self._potential, "turn_physical_on", object)()
        getattr(self._df._pot, "turn_physical_on", object)()

    # /def

    #################################################################
    # Sampling

    def __call__(
        self,
        n: int = 1,
        *,
        representation_type: TH.OptRepresentationLikeType = None,
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
        # can't pass a random seed, set in context
        with self._random_context(random):
            orbits = self._df.sample(
                R=None,
                z=None,
                phi=None,
                n=n,
                return_orbit=True,
            )

        if isinstance(orbits, (coord.BaseCoordinateFrame, coord.SkyCoord)):
            return coord.SkyCoord(orbits, copy=False)
        # else: need to turn an orbit into a SkyCoord
        # this uses the galpy GC info and then the astropy machinery,
        # to better control which galactocentric parameters are used.

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

        samples = coord.SkyCoord(
            self.frame.realize_frame(
                rep,
                representation_type=(
                    self._infer_representation(representation_type)
                    or rep.__class__
                ),
            ),
            copy=False,
        )

        # TODO! better storage of these properties, so stay when transform.
        samples.cache["potential"] = self.potential
        # from init if divergent mass, preloaded total_mass() otherwise.
        samples.cache["mass"] = np.ones(n) * self._total_mass / n
        # AGAMA compatibility

        return samples

    # /def


# /class


##############################################################################


class MeshGridPositionDF(DF):
    """Mesh-Grid Position Distribution.

    Parameters
    ----------
    pot : PotentialWrapper
    meshgrid : coord-like

    """

    def __init__(self, pot, meshgrid):
        self._sampler = MeshGridPotentialSampler(
            GalpyPotentialWrapper(pot),
            meshgrid,
        )

    # /def

    @property
    def _pot(self):
        return self._sampler._wrapper_potential

    # /def

    def sample(
        self, n: int, rng: T.Optional[np.random.Generator] = None, **kw
    ):
        """Sample.

        .. todo::

            handle uneven voxels

        Parameters
        ----------
        n : int
            number of sample points
        rng : `~numpy.random.Generator` or None

        """
        return self._sampler(n, rng=rng, **kw)

    # /def


# /class

##############################################################################
# END
