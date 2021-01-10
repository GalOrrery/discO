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
from discO.common import FrameLikeType
from discO.core.sample import PotentialSampler

##############################################################################
# CODE
##############################################################################


class GalpyPotentialSampler(PotentialSampler, package="galpy"):
    """Sample a :mod:`~galpy` Potential.

    Parameters
    ----------
    df : `~galpy.df.df.df.df`
        Distribution Function.

    frame : frame-like or None (optional, keyword only)
        The preferred frame in which to sample.

    """

    def __init__(self, df, *, frame: T.Optional[FrameLikeType] = None):
        # TODO support potential -> df
        super().__init__(df, frame=frame)

    # /def

    def __call__(
        self, n: int = 1, frame: T.Optional[FrameLikeType] = None, **kwargs
    ):
        """Sample.

        Parameters
        ----------
        n : int
            number of samples
        frame : frame-like or None
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
            R=None, z=None, phi=None, n=n, return_orbit=True
        )

        # TODO make sure transformation is correct
        samples = orbits.SkyCoord().transform_to(frame)
        samples.potential = self._sampler
        samples.mass = (  # AGAMA compatibility
            np.ones(n) * 2 * self._sampler._pot.mass(np.inf) / n
        )

        return samples

    # /def

    # def sample_at_c(self, c=None, n=1, frame=None, **kargs):
    #     if c is None:
    #         R, z, phi = None, None, None

    #     else:
    #         rep = c.represent_as(coord.CylindricalRepresentation)
    #         R, z, phi = rep.rho, rep.z, rep.phi

    #     orbits = self._sampler.sample(
    #         R=R, z=z, phi=phi, n=n, return_orbit=True,
    #     )
    #     samples = orbits.SkyCoord().transform_to(frame)

    #     return samples

    # # /def


# /class

##############################################################################
# END
