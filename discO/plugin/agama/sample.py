# -*- coding: utf-8 -*-

""":mod:`~agama` Potential Sampler."""

__all__ = [
    "AGAMAPotentialSampler",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

# PROJECT-SPECIFIC
import discO.type_hints as TH
from discO.core.sample import PotentialSampler
from discO.utils.random import RandomLike

##############################################################################
# CODE
##############################################################################


class AGAMAPotentialSampler(PotentialSampler, key="agama"):
    """Sample a :mod:`~agama` Potential.

    Parameters
    ----------
    potential : `~agama.potential`
        The potential object.

    frame : frame-like or None (optional, keyword-only)
        The preferred frame in which to sample.

    """

    def __call__(
        self,
        n: int = 1,
        representation_type: TH.OptRepresentationLikeType = None,
        random: RandomLike = None,
        **kwargs
    ) -> TH.SkyCoordType:
        """Sample.

        Parameters
        ----------
        n : int
            number of samples
        frame : frame-like or None
            output frame of samples
        **kwargs:
            ignored.

        Returns
        -------
        SkyCoord

        """
        # Get preferred frame and representation
        frame = self.frame
        representation_type = self._infer_representation(representation_type)

        # TODO accepts a potential parameter. what does this do?
        # TODO confirm random seed.
        with self._random_context(random):
            pos, masses = self._potential.sample(n=n)

        # process the position and mass
        if np.shape(pos)[1] == 6:
            pos, vel = pos[:, :3], pos[:, 3:]  # TODO: vel
            differentials = dict(
                s=coord.CartesianDifferential(*vel.T * u.km / u.s),
            )
        else:
            differentials = None
        rep = coord.CartesianRepresentation(
            *pos.T * u.kpc, differentials=differentials
        )

        if representation_type is None:
            representation_type = rep.__class__
        samples = SkyCoord(
            frame.realize_frame(rep, representation_type=representation_type),
            copy=False,
        )
        samples.mass = masses * u.solMass
        samples.potential = self.potential

        return samples

    # /def


# /class

# -------------------------------------------------------------------


##############################################################################
# END
