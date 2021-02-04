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
import numpy as np

# PROJECT-SPECIFIC
from discO.core.sample import PotentialSampler
from discO.type_hints import FrameLikeType

##############################################################################
# CODE
##############################################################################


class GalpyPotentialSampler(PotentialSampler, key="galpy"):
    """Sample a :mod:`~galpy` Potential.

    Parameters
    ----------
    df : `~galpy.df.df.df.df`
        Distribution Function.

    frame : frame-like or None (optional, keyword-only)
        The preferred frame in which to sample.

    **kwargs
        Not used. Needed to absorb option from ``__new__``

    """

    def __init__(
        self, df, *, frame: T.Optional[FrameLikeType] = None, **kwargs
    ):
        # TODO support potential -> df
        super().__init__(df, frame=frame, **kwargs)

    # /def

    def __call__(
        self, n: int = 1, frame: T.Optional[FrameLikeType] = None, **kwargs
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
        `SkyCoord`

        """
        # Get preferred frames
        frame = self._preferred_frame_resolve(frame)

        # can't pass a random seed. TODO set in with statement.s
        orbits = self._sampler.sample(
            R=None,
            z=None,
            phi=None,
            n=n,
            return_orbit=True,
        )

        # TODO make sure transformation is correct
        # and better storage of these properties, so stay when transform.
        samples = orbits.SkyCoord().transform_to(frame)
        samples.potential = self._sampler
        samples.mass = (  # AGAMA compatibility
            np.ones(n) * 2 * self._sampler._pot.mass(np.inf) / n
        )

        return samples

    # /def


# /class

##############################################################################
# END
