# -*- coding: utf-8 -*-

"""Gaia Error Field Script.

This script can be run from the command line with the following parameters:

Parameters
----------
o, order : int (optional, keyword-only)
    The HEALPix order. Default 6.
n, ngroups : int (optional, keyword-only)
    The number of total groups. Default 200.

allsky : bool, keyword-only
    A flag indicating all HEALPix pixel ids are to be queried and fit.
    If used, 'allsky' and 'pixels_range' cannot be included.
pixels : tuple of int, keyword-only
    The set HEALPix pixel ids to query and fit.
    If passed as a kwarg, 'allsky' and 'pixels_range' cannot be included.
r, pixels_range : tuple of int, keyword-only
    2 integers setting the range of HEALPix ids to query and fit.
    If passed as a kwarg, 'allsky' and 'pixels' cannot be included.

i, random_index : int or None (optional, keyword-only)
    Limit the number of queried stars to within the random index.
    This can be used to speed up test queries without impacting which pixels
    are queried and fit.
rng : int (optional, keyword-only)
    The random number generator seed.
use_local : bool (optional, keyword-only)
    Whether to perform the queries on Gaia's server or locally.
    See :mod:`gaia_tools` for details.

plot : bool (optional, keyword-only)
    Whether to make plots.
filter_warnings : bool (optional, keyword-only)
    Whether to filter warnings.
v, verbose : bool (optional, keyword-only)
    Script verbosity.

saveloc : str (optional, keyword-only)
    The save location for the data.
"""

__all__ = [
    # script
    "make_parser",
    "main",
    # functions
    "fit_pixel",
    "query_and_fit_pixel_set",
]


##############################################################################
# IMPORTS

# BUILT-IN
import argparse
import pathlib
import pickle
import typing as T
import warnings

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import astropy_healpix
import healpy
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tqdm
from astropy.table import QTable, Row
from gaia_tools.query import query as do_query
from numpy.random import Generator
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.metrics._regression import UndefinedMetricWarning
from sklearn.utils import shuffle
from astropy_healpix import nside2npix, order2nside

# PROJECT-SPECIFIC
from .sky_distribution import main as sky_distribution_main

##############################################################################
# PARAMETERS

RandomStateType = T.Union[None, int, np.random.RandomState, np.random.Generator]
AxesSubplotType = T.TypeVar("AxesSubplotType", bound=plt.axes._subplots.AxesSubplot)

THIS_DIR = pathlib.Path(__file__).parent

# gaia_tools doesn't have ``GAIA_HEALPIX_INDEX``, so we use the equivalent
# formula source_id / 2^(35 + (12 - order) * 2)
# see https://www.gaia.ac.uk/data/gaia-data-release-1/adql-cookbook
ADQL_QUERY = """
SELECT
source_id, hpx{healpix_order},
parallax, parallax_error,
ra, ra_error,
dec, dec_error

FROM (
    SELECT
    source_id, random_index,
    CAST(FLOOR(source_id/POWER(2, 35+(12-{healpix_order})*2)) AS BIGINT) AS hpx{healpix_order},
    parallax, parallax_error,
    ra, ra_error,
    dec, dec_error

    FROM gaiadr2.gaia_source AS gaia
) AS gaia

WHERE hpx{healpix_order} IN {pixel_ids}
AND parallax >= 0
"""

##############################################################################
# CODE
##############################################################################


def _fit_linear(
    X: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    train_size: int,
    weight: T.Union[bool, npt.NDArray[np.float_]] = True,
    *,
    random_state: RandomStateType = None,
) -> T.Tuple[npt.NDArray[np.float_], LinearRegression]:
    """Fit data with linear regression model.

    Parameters
    ----------
    X : (N, 3) ndarray[float]
        The data with columns of
        [:math:`\alpha`, :math:`\delta`, :math:`\log_{10}(\rm{parallax})`]
    y : (N, ) ndarray[float]
        Log10 of the fractional parallax error.
    train_size : int
        Number of samples to generate. If left to None this is automatically
        set to the first dimension of the arrays. It should not be larger than
        the length of arrays.
        See `sklearn.utils.shuffle`.
    weight : bool or ndarray[float], optional
        Individual weights for each sample.
        See :meth:`sklearn.linear_model.LinearRegression.fit`
    random_state : `numpy.random.Generator`, `numpy.random.RandomState`, int, or None (optional)
        The random number generator or constructor thereof.
        Passed directly to `sklearn.utils.shuffle`.

    Returns
    -------
    ypred : ndarray[float]
        Predicted labels.
    model : `~sklearn.linear_model.LinearRegression`
        The fit linear regression model.
    """
    model = LinearRegression()

    # randomize the data order
    idx: npt.NDArray[np.int_] = np.arange(0, len(X))
    order: npt.NDArray[np.int_] = shuffle(idx, random_state=random_state, n_samples=train_size)

    # create weight for fitting
    if weight is True:
        xy: npt.NDArray[np.float_] = np.vstack([X[:, 2], y])
        wgt: npt.NDArray[np.float_] = gaussian_kde(xy)(xy)
        sample_weight = (1 / wgt)[order]
    elif isinstance(weight, np.ndarray):
        sample_weight = (1 / weight)[order]
    else:  # weight False
        sample_weight = None

    # fit data, with weights
    model.fit(X[order], y[order], sample_weight=sample_weight)

    # get predictions: ra & dec are at median value. log10 parallax is linear.
    Xp: npt.NDArray[np.float_] = np.c_[
        np.full(100, np.median(X[:, 0])),  # ra
        np.full(100, np.median(X[:, 1])),  # dec
        np.linspace(X[:, 2].min(), X[:, 2].max(), 100),  # log10(p)
    ]
    ypred: npt.NDArray[np.float_] = model.predict(Xp)

    return ypred, model


def fit_pixel(
    pixel: QTable, pixel_id: int, *, saveloc: pathlib.Path, ax: T.Optional[AxesSubplotType] = None
) -> None:
    """Fit pixel with linear models.

    The two linear models are 1) with and 2) without an inverse sample density
    weighing.

    Parameters
    ----------
    pixel : `~astropy.table.QTable`
        Must have columns 'ra', 'dec', 'parallax', 'parallax_frac_error'
    pixel_id : int
       Healpix index for the 'pixel'.

    saveloc : `pathlib.Path`, keyword-only
        Where to save the fit to the 'pixel'.
    ax : `matplotlib.axes._subplots.AxesSubplot` or None (optional, keyword-only)
        Plot axes onto which to plot the data and fits.
        If `None`, nothing is plotted.
        See `plot_parallax_prediction`.
    """
    pixel = pixel[np.isfinite(pixel["parallax"])]  # filter out NaN  # TODO! in query

    # construct the signal array
    X: npt.NDArray[np.float_]
    y: npt.NDArray[np.float_]
    X = np.c_[
        u.Quantity(pixel["ra"], u.deg, copy=False).value,
        u.Quantity(pixel["dec"], u.deg, copy=False).value,
        np.log10(u.Quantity(pixel["parallax"], u.mas, copy=False).value),
    ]
    y = np.log10(pixel["parallax_frac_error"].value.reshape(-1, 1))[:, 0]

    # get signal density of the parallax
    xy: npt.NDArray[np.float_] = np.vstack([X[:, 2], y])
    kde = gaussian_kde(xy)(xy)

    # fit a few different ways
    yregkde, reg = _fit_linear(X, y, train_size=int(len(pixel) * 0.8), weight=kde)
    yreguw, reg1 = _fit_linear(X, y, train_size=int(len(pixel) * 0.8), weight=False)

    # save weighted fit
    with open(saveloc / f"fit_{pixel_id:010}.pkl", mode="wb") as f:
        pickle.dump(reg, f)  # the weighted linear regression

    if ax is not None:
        plot_parallax_prediction(
            X,
            y,
            kde,
            yregkde,
            yreguw,
            pixel_id=pixel_id,
            ax=ax,
            labels=("linear model: density-weighting", "linear model: no density weight"),
        )


def query_and_fit_pixel_set(
    pixel_ids: tuple[int, ...],
    healpix_order: int,
    random_index: T.Optional[int] = 1_000_000,
    *,
    plot: bool = True,
    use_local: bool = True,
    saveloc: pathlib.Path = THIS_DIR
) -> None:
    """Query and fit a set of sky pixels (healpix pixels).

    Parameters
    ----------
    pixel_ids : tuple[int]
        Set of Healpix indices, at order.
    healpix_order : int
        The healpix order. See :func:`order2nside`
    random_index : int or None, optional
        The Gaia random index depth in the query. `None` will query the whole
        database. An integer (default 10^6) will limit the depth and make the
        query much faster.

    plot : bool (optional, keyword-only)
        Whether to plot the set of pixels.
    use_local : bool (optional, keyword-only)
        Whether to perform the query on a local database (`True`, default) or
        on Gaia's servers (`False`).
    saveloc : `pathlib.Path` (optional, keyword-only)
        Where to save the fit to the 'pixel'.
    """
    # create directories
    FOLDER = saveloc / f"order_{healpix_order}"
    FOLDER.mkdir(exist_ok=True)

    PLOT_DIR = FOLDER / "figures"
    PLOT_DIR.mkdir(exist_ok=True)

    DATA_DIR = FOLDER / "pixel_fits"
    DATA_DIR.mkdir(exist_ok=True)

    # -----------------------
    # Query batch

    # make query string
    hpl = f"hpx{healpix_order}"  # column name for healpix index
    adql_query = ADQL_QUERY.format(healpix_order=healpix_order, pixel_ids=pixel_ids)
    if random_index is not None:
        adql_query += f"AND random_index < {int(random_index)}"

    # perform query using `gaia_tools`
    # if the query fails to return anything, stop there.
    result = do_query(adql_query, local=use_local, use_cache=False, verbose=True, timeit=True)
    if len(result) == 0:
        warnings.warn(f"no data in pixels: {pixel_ids}")
        return

    # compute and add the fractional error to the table
    result["parallax_frac_error"] = result["parallax_error"] / result["parallax"]

    # reorganize the results to group stars by pixel
    pixels: QTable = result.group_by(hpl)

    # plot the pixels
    if plot:
        fig = plt.figure()
        plot_mollview(pixel_ids, healpix_order, fig=fig)

        shortened = hash(pixel_ids)  # TODO! do better. Put in PDF metadata
        with open(PLOT_DIR / f"mollview-{shortened}.txt", mode="w") as f:
            f.write(str(pixel_ids))

        fig.savefig(PLOT_DIR / f"mollview-{shortened}.pdf")

    # -----------------------
    # Fits to each pixel

    axs: npt.NDArray[np.object_]  # axes or 0s
    if plot:  # set up parallax plots
        rows, remainder = np.divmod(len(pixel_ids), 4)
        width = remainder if (rows == 0) else 4
        if remainder > 0:
            rows += 1
        fig, axs = plt.subplots(rows, width, figsize=(5 * width, 5 * rows))
    else:
        axs = np.array([None] * len(pixels.groups))  # noop for iteration

    pixel: QTable
    ax: T.Union[plt.axes._subplots.AxesSubplot, None]
    for pixel, ax in zip(pixels.groups, axs.flat):  # iter thru pixels
        fit_pixel(pixel, int(pixel[hpl][0]), saveloc=DATA_DIR, ax=ax)

    # save plot of all the pixels
    if plot:
        plt.tight_layout()
        fig.savefig(PLOT_DIR / f"parallax-{shortened}.pdf")


def make_groups(
    sky: QTable, healpix_order: int, numgroups: int = 200
) -> T.List[npt.NDArray[np.int_]]:
    """Group pixels together s.t. groups have approximate the same number of stars.

    Parameters
    ----------
    sky : `~astropy.table.QTable`
        Table of stars, grouped by healpix pixel ID.
    healpix_order : int
        The healpix order. See :func:`astropy_healpix.order2nside`
    numgroups : int, optional
        The number of groups to make.

    Returns
    -------
    groupsids : list[ndarray[int]]
        List of grouped pixels.
    """
    npix: int = nside2npix(order2nside(healpix_order))  # the number of sky pixels

    # get healpix column name. it depends on the order, but is the group key.
    colname = sky.groups.keys.colnames[0]

    # get unique ids
    pixelids, hpx_indices, num_counts_per_pixel = np.unique(
        sky[colname].value, return_index=True, return_counts=True
    )

    allpixelids = np.arange(npix)
    pixelnums = np.zeros(npix)
    pixelnums[pixelids] = num_counts_per_pixel
    pixelnums[pixelnums == 0] = 1  # set minimum number of 'counts' to 1

    # sort by number of counts
    sorter = np.argsort(pixelnums)[::-1]
    pixelnums = pixelnums[sorter]
    allpixelids = allpixelids[sorter]

    groupsids = [allpixelids[i::numgroups] for i in range(numgroups)]

    return groupsids


# ============================================================================
# Plotting


def plot_parallax_prediction(
    Xtrue: npt.NDArray[np.float_],
    ytrue: npt.NDArray[np.float_],
    kde: gaussian_kde,
    *ypred: npt.NDArray[np.float_],
    pixel_id: int,
    ax: T.Optional[plt.Axes] = None,
    labels: T.Tuple[str, ...],
) -> plt.Figure:
    """Plot predicted parallax.

    Parameters
    ----------
    Xtrue : ndarray[float]
    ytrue : ndarray[float]
    kde : `scipy.stats.gaussian_kde`
    *ypred : ndarray[float]
    pixel_id : int, keyword-only
    ax : `matplotlib.pyplot.Axes` or None, keyword-only

    Returns
    -------
    `matplotlib.pyplot.Figure`
    """
    # Get figure from axes. If None, make new.
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    # make average coordinates which approximate the location where `ypred`
    # were evaluated. This is just spread out better than the real location.
    Xpred = np.c_[
        np.full(100, np.median(Xtrue[:, 0])),  # ra
        np.full(100, np.median(Xtrue[:, 1])),  # dec
        np.linspace(Xtrue[:, 2].min(), Xtrue[:, 2].max(), 100),  # p
    ]

    # plot the coordinates and evaluations
    ax.scatter(Xtrue[:, -1], ytrue, s=5, label="data", alpha=0.3, c=kde)
    for i, y in enumerate(ypred):
        ax.scatter(Xpred[:, -1], y, s=5, label=r"$y_{pred}$ " + str(i))

    # set axes labels and adjust properties
    ax.set_xlabel(r"$\log_{10}$ parallax [mas]")
    ax.set_ylabel(r"$\log_{10}$ parallax fractional error")
    ax.set_title(f"Patch={pixel_id}")
    # distance label is secondary to parallax
    secax = ax.secondary_xaxis(
        "top",
        functions=(
            lambda logp: np.log10(
                coord.Distance(parallax=10 ** logp * u.mas).to_value(u.pc),
            ),
            lambda logd: np.log10(
                coord.Distance(10 ** logd * u.pc).parallax.to_value(u.mas),
            ),
        ),
    )
    secax.set_xlabel(r"$\log_{10}$ Distance [kpc]")

    ax.set_ylim(-3, 3)
    ax.invert_xaxis()
    ax.legend()

    return fig


# FIXME! this doesn't seem to be plotting correctly
def plot_mollview(
    pixel_ids: tuple[int, ...], healpix_order: int, fig: T.Optional[plt.Figure] = None
) -> plt.Figure:
    """Plot Mollweide view with pixels on sky.

    Parameters
    ----------
    pixel_ids : tuple[int]
        Set of pixel ids (int).
    healpix_order : int
        The healpix order.  See :func:`order2nside`

    Returns
    -------
    `matplotlib.pyplot.Figure`
    """
    npix = nside2npix(order2nside(healpix_order))

    # background plot
    m = np.arange(npix)
    alpha = np.zeros_like(m) + 0.5
    alpha[pixel_ids[0] : pixel_ids[-1]] = 1
    healpy.mollview(m, nest=True, coord=["C"], cbar=False, cmap="inferno", fig=fig, alpha=alpha)

    # pixel plot
    m[pixel_ids[0] : pixel_ids[-1]] = 3 * npix // 4
    alpha[: pixel_ids[0]] = 0
    alpha[pixel_ids[-1] :] = 0
    healpy.mollview(
        m,
        title=f"Mollview image (RING, order={healpix_order})\nPatches {pixel_ids}",
        nest=True,
        coord=["C"],
        cbar=False,
        cmap="Greens",
        fig=fig,
        reuse_axes=True,
        alpha=alpha,
    )

    return fig


##############################################################################
# Command Line
##############################################################################


def make_parser(*, inheritable: bool = False) -> argparse.ArgumentParser:
    """Expose ArgumentParser for ``main``.

    Parameters
    ----------
    inheritable: bool, optional, keyword only
        whether the parser can be inherited from (default False).
        if True, sets ``add_help=False`` and ``conflict_hander='resolve'``

    plot : bool, optional, keyword only
        Whether to produce plots, or not.

    verbose : int, optional, keyword only
        Script logging verbosity.

    Returns
    -------
    parser: `~argparse.ArgumentParser`
        The parser with arguments:
        - plot
        - verbose
    """
    parser = argparse.ArgumentParser(
        description="",
        add_help=not inheritable,
        conflict_handler="resolve" if not inheritable else "error",
    )

    # order
    parser.add_argument("-o", "--order", default=6, type=int, help="healpix order")

    # pixels are done in groups.
    parser.add_argument(
        "-n",
        "--ngroups",
        default=200,
        type=int,
        help="number of total groups",
    )

    # which pixels
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--allsky", action="store_true", help="fit all sky pixels")
    group.add_argument(
        "--pixels",
        action="append",
        type=int,
        nargs="+",
        help="only fit specified sky pixels by ID",
    )
    group.add_argument(
        "-r",
        "--pixels_range",
        type=int,
        nargs=2,
        help="fit specified sky pixels within range",
    )

    # stars in gaia
    parser.add_argument(
        "-i",
        "--random_index",
        default=None,
        type=int,
        help="limit queried stars within random index",
    )

    # random number generator
    parser.add_argument("--rng", default=0, type=int, help="random number generator seed")

    # gaia_tools
    parser.add_argument("--use_local", action="store_true", help="gaia_tools local query")

    # plot or not
    parser.add_argument("--plot", default=True, type=bool, help="plot")

    # script verbosity
    parser.add_argument("--filter_warnings", action="store_true", help="filter warnings")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")

    # save location
    parser.add_argument("--saveloc", type=str, default=THIS_DIR)

    return parser


# ------------------------------------------------------------------------


def main(
    args: T.Union[list[str], str, None] = None,
    opts: T.Optional[argparse.Namespace] = None,
) -> None:
    """Script Function.

    Parameters
    ----------
    args : list or str or None, optional
        an optional single argument that holds the sys.argv list,
        except for the script name (e.g., argv[1:])

    opts : `~argparse.Namespace` or None, optional
        Pre-constructed results of parsed args.
        Used ONLY if args is None.

    Warns
    -----
    UserWarning
        If 'args' and 'opts' are not None
    """
    # parse the input / command-line options
    ns: argparse.Namespace
    if opts is not None and args is None:
        ns = opts
    else:
        if opts is not None:
            warnings.warn("Not using `opts` because `args` are given")
        if isinstance(args, str):
            args = args.split()

        parser = make_parser()
        ns = parser.parse_args(args)

    # -----------------------
    # Make background distribution
    # This loads a table of 2 million stars, organized by healpix pixel number.
    sky: QTable = sky_distribution_main(opts=ns)

    # construct the list of groups of healpix pixels.
    # [ (pixel_1, pixel_2, ...),  (pixel_i, pixel_i+1, ...)]
    list_of_groups: T.List[T.Tuple[int, ...]]
    if ns.allsky:
        # groups the pixels together so that each group will have
        # approximately the same number of stars.
        list_of_groups = make_groups(sky, healpix_order=ns.order, numgroups=ns.ngroups)
    elif ns.pixels_range:
        pi, pf = ns.pixels_range
        if pi >= pf:
            raise ValueError("`pixels_range` must be [start, stop], with stop > start.")
        list_of_groups = np.array_split(np.arange(pi, pf), ns.ngroups)
    elif ns.pixels:
        list_of_groups = ns.pixels

    # -----------------------
    # optionally ignore warnings
    with warnings.catch_warnings():
        if ns.filter_warnings:
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            warnings.simplefilter("ignore", category=UserWarning)

        for batch in tqdm.tqdm(list_of_groups):
            query_and_fit_pixel_set(
                tuple(batch),
                healpix_order=ns.order,
                random_index=ns.random_index,
                plot=ns.plot,
                use_local=ns.use_local,
                saveloc=pathlib.Path(ns.saveloc),
            )


# ------------------------------------------------------------------------

if __name__ == "__main__":

    # call script
    main(args=None, opts=None)  # all arguments except script name


##############################################################################
# END
