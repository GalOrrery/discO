# -*- coding: utf-8 -*-

"""pynbody utilities."""

__all__ = [
    "representation_to_pynbody_snapshot",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord

##############################################################################
# PARAMETERS


##############################################################################
# CODE
##############################################################################


def representation_to_pynbody_snapshot(rep, mass=1.0, eps=0.0, **props):
    """Convert a |Representation| to a |pyNBody| snapshot.

    Parameters
    ----------
    rep : |Representation| or |CoordinateFrame| or |SkyCoord|

    mass : float or Sequence
        The masses of each particle. If Sequence, must be length 'rep'.
    eps : float or Sequence
        If Sequence, must be length 'rep'.

    **props : float or Sequence
        Other properties to set on the snapshot.
        If Sequence, must be length 'rep'.

    Returns
    -------
    snap : |pyNBody| snapshot
        In Cartesian coordinates.

    Raises
    ------
    KeyError
        If any kwargs have keys: "x", "y", "z", "vx", "vy", "vz"

    """
    import pynbody

    # first check that the kwargs are allowed
    protected = ("x", "y", "z", "vx", "vy", "vz")
    if not set(props.keys()).isdisjoint(protected):
        raise KeyError(f"kwargs {protected} can only be set by 'rep'")

    # ensure repsentation is in Cartesian coordinates.
    rep = rep.represent_as(
        coord.CartesianRepresentation, s=coord.CartesianDifferential
    )

    # make snapshot
    snap = pynbody.new(star=len(rep))
    snap["x"], snap["y"], snap["z"] = rep.xyz
    snap["vx"], snap["vy"], snap["vz"] = rep.differentials["s"].d_xyz

    snap["mass"] = mass
    snap["eps"] = eps  # should this be 0?

    for n, v in props.items():
        snap[n] = v

    return snap


# /def


# -------------------------------------------------------------------


##############################################################################
# END
