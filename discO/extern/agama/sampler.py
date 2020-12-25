# -*- coding: utf-8 -*-

"""**DOCSTRING**."""

__all__ = [
    "AGAMAPotentialSampler",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

# PROJECT-SPECIFIC
from discO.common import FrameLikeType, SkyCoordType
from discO.core.sampler import PotentialSampler

##############################################################################
# CODE
##############################################################################


class AGAMAPotentialSampler(PotentialSampler, package="agama"):
    """Sample a :mod:`~agama` Potential.

    Parameters
    ----------
    potential : `~agama.potential`
        The potential object.

    frame : frame-like or None (optional, keyword only)
        The preferred frame in which to sample.

    """

    def __call__(
        self, n: int = 1, frame: T.Optional[FrameLikeType] = None, **kwargs
    ) -> SkyCoordType:
        """Sample.

        Parameters
        ----------
        n : int
            number of samples
        frame : frame-like or None
            output frame of samples
        **kwargs:
            passed to underlying sampler.

        Returns
        -------
        SkyCoord

        """
        # Get preferred frame
        frame = self._preferred_frame_resolve(frame)

        # TODO accepts a potential parameter. what does this do?
        # TODO the random seed
        pos, masses = self._sampler.sample(n=n)  # potential=None

        if np.shape(pos)[1] == 6:
            pos, _ = pos[:, :3], pos[:, 3:]  # TODO: vel
        else:
            # vel = None  # TODO
            pass

        # TODO get agama units !
        masses = masses * u.solMass
        rep = coord.CartesianRepresentation(*pos.T * u.kpc)

        samples = SkyCoord(rep, frame=frame)
        samples.mass = masses
        samples.potential = self._sampler

        return samples

    # /def


# /class

# -------------------------------------------------------------------


##############################################################################
# END
