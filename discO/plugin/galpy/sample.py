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
import galpy.potential as gpot
import numpy as np

# PROJECT-SPECIFIC
import discO.type_hints as TH
from discO.core.sample import PotentialSampler
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
    df : :class:`~galpy.df.df.df.df` or :class:`~galpy.potential.Potential`
        Distribution Function holding the potential.
        If potential, ties to match to distribution function registry.

    frame : frame-like or None (optional, keyword-only)
        The preferred frame in which to sample.

    **kwargs
        Not used. Needed to absorb option from ``__new__``

    """

    def __init__(
        self,
        df,
        *,
        frame: T.Optional[TH.FrameLikeType] = None,
        representation_type: TH.OptRepresentationLikeType = None,
        **kwargs
    ):
        if isinstance(df, gpot.Potential):
            df = DF_REGISTRY[df.__class__.__name__](df)

        super().__init__(
            df._pot,
            frame=frame,
            representation_type=representation_type,
            **kwargs
        )
        self._df = df

    # /def

    #################################################################
    # Sampling

    def __call__(
        self,
        n: int = 1,
        *,
        frame: T.Optional[TH.FrameLikeType] = None,
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
        **kwargs
            ignored

        Returns
        -------
        :class:`~astropy.coordinates.SkyCoord`

        """
        # Get preferred frames
        frame = self._infer_frame(frame)
        representation_type = self._infer_representation(representation_type)

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
        samples.mass = (  # AGAMA compatibility
            np.ones(n) * self.potential.total_mass() / n
        )

        return samples

    # /def


# /class

##############################################################################
# END
