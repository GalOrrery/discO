# -*- coding: utf-8 -*-

"""Utilities for making an interpolated error field."""

# __all__ = []


##############################################################################
# IMPORTS

# BUILT-IN
import os
import pathlib
import pickle
import typing as T

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import tqdm
from astropy_healpix import HEALPix
from scipy.interpolate import NearestNDInterpolator

##############################################################################
# PARAMETERS

RFS = T.TypeVar("RFS", coord.BaseRepresentation, coord.BaseCoordinateFrame, coord.SkyCoord)


##############################################################################
# CODE
##############################################################################


class NearestNDInterpolatorWithUnits(NearestNDInterpolator):
    def __init__(
        self,
        x: T.Union[np.ndarray, u.Quantity],
        y: T.Union[np.ndarray, u.Quantity],
        rescale: bool = False,
        tree_options: T.Optional[dict] = None,
        yunit: u.Unit = u.one,
    ) -> None:
        # process x value and units
        self._xunit: u.UnitBase = getattr(x, "unit", u.one)
        xv: np.ndarray = (x << self._xunit).value

        # process y value and units
        self._yunit: u.UnitBase = u.Unit(yunit)
        yv: np.ndarray = (y << self._yunit).value

        # bild interpolation
        super().__init__(xv, yv, rescale=rescale, tree_options=tree_options)

    def __call__(self, x: T.Union[np.ndarray, u.Quantity]) -> u.Quantity:
        xv: np.ndarray = (x << self._xunit).value
        return super().__call__(x) << self._yunit


class SphericalLogParallaxNearestNDInterpolator(NearestNDInterpolatorWithUnits):
    def __init__(
        self,
        c: RFS,
        y: T.Union[np.ndarray, u.Quantity],
        rescale: bool = False,
        tree_options: T.Optional[dict] = None,
        yunit: u.Unit = u.one,
    ) -> None:
        x = make_X(c)
        super().__init__(x, y, rescale=rescale, tree_options=tree_options, yunit=yunit)

    def __call__(self, c: RFS) -> u.Quantity:
        return super().__call__(make_X(c))


def make_healpix_los_unitsphere_grid(healpix) -> coord.SkyCoord:
    """Unit sphere grid.

    .. todo::

        Allow for more than one point per pixel.
        Possibly by going one further order and merging to get desired number
        of points.

    Parameters
    ----------
    healpix : `~astropy_healpix.HEALPix`
        The HEALPix instance.

    Returns
    -------
    (N,) `~astropy.coordinates.SkyCoord`
    """
    pixel_ids: np.ndarray = np.arange(healpix.npix, dtype=int)  # get all pixels
    # TODO! support more than one point per pixel
    dxs, dys = [0.5], [0.5]

    temp_dim = np.zeros((len(pixel_ids), len(dxs)))
    temp_r = coord.UnitSphericalRepresentation(temp_dim * u.rad, temp_dim * u.rad)
    healpix_sc = coord.SkyCoord(healpix.frame.realize_frame(temp_r))

    for i, dx in enumerate(dxs):
        healpix_sc[:, i] = healpix.healpix_to_skycoord(pixel_ids, dx=dx)

    return healpix_sc.flatten()


def make_los_sphere_grid(unitsphere: RFS, distances: u.Quantity = np.arange(1, 20) * u.kpc) -> RFS:
    """Make a spherical grid given the unit layer.

    Parameters
    ----------
    unitsphere : (N,) Representation, CoordinateFrame, or SkyCoord
        Unit spherical grid.
    distances : (M,) |Quantity|
        Distances to which to scale the unit-spherical grid.

    Returns
    -------
    (N, M) Representation or CoordinateFrame or SkyCoord
        Same type as 'unitsphere'. In spherical coordinates.
    """
    # translate to the unit layer
    us: coord.UnitSphericalRepresentation
    us = unitsphere.represent_as(coord.UnitSphericalRepresentation)

    # create an empty spherical grid
    placeholder = np.zeros((len(us), len(distances)))
    grid = coord.SphericalRepresentation(
        placeholder * u.rad, placeholder * u.rad, placeholder * u.kpc
    )

    # fill in coordinates, at different distances
    for i, distance in enumerate(distances):
        grid[:, i] = us * distance

    if isinstance(unitsphere, coord.BaseRepresentation):
        return grid
    elif isinstance(unitsphere, coord.BaseCoordinateFrame):
        return unitsphere.realize_frame(grid)
    else:
        return coord.SkyCoord(unitsphere.realize_frame(grid))


def make_X(c: RFS) -> np.ndarray:
    """Make coordinates for evaluating `scipy` interpolations.

    Parameters
    ----------
    sr : (N, M) BaseRepresentation, BaseCoordinateFrame, SkyCoord

    Returns
    -------
    (NxM, 3) ndarray[float]
        columns are flattened dimensions of 'sr':
        - longitude in deg,
        - latitude in deg
        - log10 parallax
    """
    # change to spherical representation
    sr = c.represent_as(coord.SphericalRepresentation)
    # [lon, lat, log10(p)]
    X = np.c_[
        sr.lon.to_value(u.deg).flat,
        sr.lat.to_value(u.deg).flat,
        np.log10(sr.distance.parallax.to_value(u.mas).flat),
    ]
    return X


def interpolate_errfield_on_los_sphere_grid(
    directory: T.Union[str, os.PathLike], healpix: HEALPix, sphere_grid: RFS
) -> NearestNDInterpolatorWithUnits:
    """Evaluate error field on spherical grid.

    Parameters
    ----------
    directory : str or path-like
    healpix : `~astropy_healpix.HEALPix`
    sphere_grid : (N, M) Representation or CoordinateFrame or SkyCoord
        For example, see `make_los_sphere_grid`.

    Returns
    -------
    `~scipy.interpolate.ndgriddata.NearestNDInterpolator`
        Interpolation of the evaluation of the saved patch fits
        on the LOS spherical grid.
        See :func:`make_X` for how the grid is interpreted.
        For a given input in the same coordinates, returns the
        predicted :math:`\log_{10}{\delta{\text{parallax}} / \text{parallax}}`.
    """
    if isinstance(sphere_grid, coord.BaseRepresentation) and not isinstance(
        sphere_grid, coord.SphericalRepresentation
    ):
        raise ValueError("`sphere_grid` must be a `SphericalRepresentation`.")
    elif (
        hasattr(sphere_grid, "representation_type")
        and sphere_grid.representation_type is not coord.SphericalRepresentation
    ):
        raise ValueError("`sphere_grid` must be in a spherical representation.")

    # data directory
    datadir = pathlib.Path(directory).expanduser().resolve()

    # Work with spherical representation, LOS
    sr = sphere_grid.represent_as(coord.SphericalRepresentation)

    # Start with empty prediction (N, M)
    ypred = np.full(sr.shape, np.nan)

    # iterate through the LOS
    for i, los in enumerate(tqdm.tqdm(sr)):

        # Get ID. Indices are (sphere, distance) so only need (sphere, )
        los_hp_id = healpix.skycoord_to_healpix(sphere_grid.realize_frame(los[0]))

        # Open correct file
        with open(datadir / f"fit_{los_hp_id:010}.pkl", mode="rb") as f:
            patchfit = pickle.load(f)

        # Build coordinates to evaluate scipy interpolation object
        # [lon, lat, log10(parallax)]
        X = make_X(los)

        # Evaluate object, filling in the LOS
        ypred[i, :] = patchfit.predict(X)

    # Make ND interpolation
    interp = SphericalLogParallaxNearestNDInterpolator(
        sr, ypred.flat, rescale=True, yunit=u.dex(u.one)
    )

    return interp


##############################################################################
# END
