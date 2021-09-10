# -*- coding: utf-8 -*-

"""Gaia Error Field Script.

This script can be run from the command line with the following parameters:

Parameters
----------

"""

__all__ = [
    # script
    "make_parser",
    "main",
    # functions
    "fit_kernel_ridge",
    "fit_gaussian_process",
    "fit_support_vector",
    "fit_linear",
    # querying
    "query_and_fit_patch_set",
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
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tqdm  # TODO! make optional
from astropy import table
from astroquery.gaia import Gaia
from scipy.stats import gaussian_kde
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.metrics._regression import UndefinedMetricWarning
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.utils import shuffle

##############################################################################
# PARAMETERS

RandomStateType = T.Union[None, int, np.random.RandomState, np.random.Generator]

# General
_PLOT: bool = True  # Whether to plot the output

THIS_DIR = pathlib.Path(__file__).parent
PLOT_DIR = THIS_DIR / "figures"
PLOT_DIR.mkdir(exist_ok=True)

DATA_DIR = THIS_DIR / "pk_reg"
DATA_DIR.mkdir(exist_ok=True)

##############################################################################
# CODE
##############################################################################


def fit_kernel_ridge(
    X: npt.NDArray,
    y: npt.NDArray,
    train_size: int,
    random_state: RandomStateType = None,
) -> (npt.NDArray, KernelRidge):
    """Kernel-Ridge Regression code.

    Parameters
    ----------
    X : ndarray
    y : ndarray
    train_size : int
    random_state : `numpy.random.Generator`, `numpy.random.RandomState`, int, or None (optional)

    Returns
    -------
    ykr : ndarray
    kr : `~sklearn.kernel_ridge.KernelRidge`
    """
    # construct grid-search for optimal parameters
    kr = GridSearchCV(
        KernelRidge(alpha=1, kernel="linear", gamma=0.1),
        param_grid={
            "alpha": [1e0, 0.1, 1e-2, 1e-3],
            "gamma": np.logspace(-2, 2, 5),
        },
    )

    # randomize the data order
    idx = shuffle(np.arange(0, len(X)), random_state=random_state, n_samples=train_size)

    # Fitting using the Kernel-Ridge Regression
    kr.fit(X[idx], y[idx])
    # get predictions: ra & dec are at median value. parallax is linear
    Xp = np.array(
        [
            np.ones(100) * np.median(X[:, 0]),  # ra
            np.ones(100) * np.median(X[:, 1]),  # dec
            np.linspace(X[:, 2].min(), X[:, 2].max(), 100),  # p
        ],
    ).T
    ykr = kr.predict(Xp)

    return ykr, kr


# /def


# def fit_gaussian_process(
#     X: npt.NDArray,
#     y: npt.NDArray,
#     train_size: int,
#     random_state: RandomStateType = None,
# ) -> (npt.NDArray, GaussianProcessRegressor):
#     """Gaussian-Process Regression code.
# 
#     Parameters
#     ----------
#     X : ndarray
#     y : ndarray
#     train_size : int
#     random_state : `numpy.random.Generator`, `numpy.random.RandomState`, int, or None (optional)
# 
#     Returns
#     -------
#     ykr : ndarray
#     kr : `~sklearn.gaussian_process.GaussianProcessRegressor`
#     """
#     # estimator
#     gpr = GaussianProcessRegressor(kernel=None)
# 
#     # randomize the data order
#     idx = shuffle(np.arange(0, len(X)), random_state=random_state, n_samples=train_size)
# 
#     # fit
#     gpr.fit(X[idx], y[idx])
#     ygp = gpr.predict(Xp)
# 
#     return ygp, gpr
# 
# 
# # /def


def fit_support_vector(
    X: npt.NDArray,
    y: npt.NDArray,
    train_size: int,
    random_state: RandomStateType = None,
) -> (npt.NDArray, SVR):
    """support-vector regression.

    Parameter
    ---------
    X : ndarray
    y : ndarray
    train_size : int
    random_state : `numpy.random.Generator`, `numpy.random.RandomState`, int, or None (optional)

    Returns
    -------
    ysv : ndarray
    svr : `~sklearn.svm.SVR`
    """
    svr = GridSearchCV(
        SVR(kernel="linear", gamma=0.1),
        param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)},
    )

    # randomize the data order
    idx = shuffle(np.arange(0, len(X)), random_state=random_state, n_samples=train_size)

    # Fitting using the Support Vector
    svr.fit(X[idx], y[idx])
    # get predictions: ra & dec are at median value. parallax is linear
    Xp = np.array(
        [
            np.ones(100) * np.median(X[:, 0]),  # ra
            np.ones(100) * np.median(X[:, 1]),  # dec
            np.linspace(X[:, 2].min(), X[:, 2].max(), 100),  # p
        ],
    ).T
    ysv = svr.predict(Xp)

    return ysv, svr


# /def


def fit_linear(
    X: npt.NDArray,
    y: npt.NDArray,
    train_size: int,
    weight: T.Union[bool, npt.NDArray] = True,
    random_state: RandomStateType = None,
) -> (npt.NDArray, LinearRegression):
    """Linear regression model.

    Parameters
    ----------
    X : ndarray
    y : ndarray
    train_size : int
    weight : bool or ndarray, optional
    random_state : `numpy.random.Generator`, `numpy.random.RandomState`, int, or None (optional)

    Returns
    -------
    ysv : ndarray
    svr : `~sklearn.linear_model.LinearRegression`
    """
    lr = LinearRegression()

    # randomize the data order
    idx = shuffle(np.arange(0, len(X)), random_state=random_state, n_samples=train_size)

    # fit, optionally with weights
    if weight is True or isinstance(weight, np.ndarray):  # True or kde
        if weight is True:
            xy = np.vstack([X[:, 2], y])
            weight = gaussian_kde(xy)(xy)
        lr.fit(X[idx], y[idx], sample_weight=(1 / weight)[idx])
    else:
        lr.fit(X[idx], y[idx])

    # get predictions: ra & dec are at median value. parallax is linear
    Xp = np.array(
        [
            np.ones(100) * np.median(X[:, 0]),  # ra
            np.ones(100) * np.median(X[:, 1]),  # dec
            np.linspace(X[:, 2].min(), X[:, 2].max(), 100),  # p
        ],
    ).T
    ylr = lr.predict(Xp)

    return ylr, lr


# /def


# ===================================================================


def plot_parallax_prediction(
    Xtrue: npt.NDArray,
    ytrue: npt.NDArray,
    kde,
    ypred1: npt.NDArray,
    ypred2: npt.NDArray,
    ypred3: npt.NDArray,
    fids,
    ax=None
) -> plt.Figure:
    """Plot predicted parallax.

    Parameters
    ----------
    Xtrue
    ytrue
    kde
    ypred1
    ypred2
    ypred3
    fids

    Returns
    -------
    `matplotlib.pyplot.Figure`
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot()
    else:
        fig = ax.figure

    ax.set_xlabel(r"$\log_{10}$ parallax [mas]")
    ax.set_ylabel(r"$\log_{10}$ parallax fractional error")
    ax.set_title(f"Patch={fids}")

    # distance label
    secax = ax.secondary_xaxis(
        "top",
        functions=(
            lambda logp: np.log10(coord.Distance(parallax=10 ** logp * u.mas).to_value(u.pc)),
            lambda logd: np.log10(coord.Distance(10 ** logd * u.pc).parallax.to_value(u.mas)),
        ),
    )
    secax.set_xlabel(r"$\log_{10}$ Distance [kpc]")

    Xpred = np.array(
        [
            np.ones(100) * np.median(Xtrue[:, 0]),  # ra
            np.ones(100) * np.median(Xtrue[:, 1]),  # dec
            np.linspace(Xtrue[:, 2].min(), Xtrue[:, 2].max(), 100),  # p
        ],
    ).T

    ax.scatter(Xtrue[:, -1], ytrue, s=5, label="data", alpha=0.3, c=kde)
    ax.scatter(Xpred[:, -1], ypred1, s=5, label="kernel-ridge")
    ax.scatter(Xpred[:, -1], ypred2, s=5, label="linear model: density-weighting")
    ax.scatter(Xpred[:, -1], ypred3, s=5, label="linear model: no density weight")

    ax.set_ylim(-3, 3)
    ax.invert_xaxis()
    ax.legend()

    return fig


# /def


def plot_mollview(patch_ids, order, fig=None):
    """Plot Mollweide view with patches on sky."""
    nside = hp.order2nside(order)
    npix = hp.nside2npix(nside)

    # background plot
    m = np.arange(npix)
    alpha = np.zeros_like(m) + 0.5
    alpha[patch_ids[0] : patch_ids[-1]] = 1
    hp.mollview(
        m,
        nest=True,
        coord=["C"],
        cbar=False,
        cmap="inferno",
        fig=fig,
        alpha=alpha,
    )

    # patch plot
    m[patch_ids[0] : patch_ids[-1]] = 3 * npix // 4
    alpha[:patch_ids[0]] = 0
    alpha[patch_ids[-1]:] = 0
    hp.mollview(
        m,
        title=f"Mollview image (RING, order={order})\nPatches {patch_ids}",
        nest=True,
        coord=["C"],
        cbar=False,
        cmap="Greens",
        fig=fig,
        reuse_axes=True,
        alpha=alpha,
    )
    fig = plt.gcf()

    return fig


def query_and_fit_patch_set(patch_ids: tuple[int, ...], order: int, plot=bool, random_index: T.Optional[int]=1000000) -> None:
    """Query and fit a set of sky patches.

    Parameters
    ----------
    patch_ids : tuple[int]
        Set of patch ids (int).
    order : int
        The healpix order.  See :func:`healpy.order2nside`

    """
    # create Gaia query
    hpl = f"healpix{order}"  # column name
    query = f"""
    SELECT
    source_id, GAIA_HEALPIX_INDEX({order}, source_id) AS {hpl},
    parallax AS parallax, parallax_error AS parallax_error,
    ra, ra_error AS ra_err,
    dec, dec_error AS dec_err

    FROM gaiadr2.gaia_source

    WHERE GAIA_HEALPIX_INDEX({order}, source_id) IN {patch_ids}
    AND parallax >= 0
    """
    if random_index is not None:
        query += f"AND random_index < {random_index}"

    job = Gaia.launch_job_async(
        query,
        dump_to_file=False,
        verbose=False,
    )
    # perform query and
    r = table.QTable(job.get_results(), copy=False)
    rgr = r.group_by(hpl)  # group stars by patch

    # plot the patches
    if plot:
        fig = plot_mollview(patch_ids, order)
        # TODO! allow for plot directory
        fig.savefig(PLOT_DIR / f"mollview-{'-'.join(map(str, patch_ids))}.pdf")

    # parallax plot
    if plot:
        rows, remainder = np.divmod(len(patch_ids), 4)
        if rows == 0:
            width = remainder
        else:
            width = 4
        if remainder > 0:
            rows += 1
        fig, axs = plt.subplots(rows, width, figsize=(5 * width, 5 * rows))
    else:
        axs = np.zeros(len(rgr.groups))

    key: table.Row
    group: table.Table
    for grp, ax in zip(rgr.groups, axs.flat):
        patch_id: int = grp[hpl][0]

        grp = grp[np.isfinite(grp["parallax"])]  # filter out NaN  # TODO! in query
        # group = group[group["parallax"] > 0]  # positive parallax
        
        # add the fractional error
        grp["parallax_frac_error"] = grp["parallax_error"] / grp["parallax"]

        X = np.array(
            [
                grp["ra"].to_value(u.deg),
                grp["dec"].to_value(u.deg),
                np.log10(grp["parallax"].to_value(u.mas)),
            ],
        ).T
        y = np.log10(grp["parallax_frac_error"].value.reshape(-1, 1))[:, 0]

        xy = np.vstack([X[:, 2], y])
        kde = gaussian_kde(xy)(xy)

        # fit a few different ways
        ykr, kr = fit_kernel_ridge(X, y, train_size=int(len(grp) * 0.8))
        ysv, svr = fit_support_vector(X, y, train_size=int(len(grp) * 0.8))
        yreg, reg = fit_linear(X, y, train_size=int(len(grp) * 0.8), weight=kde)
        yreg1, reg1 = fit_linear(X, y, train_size=int(len(grp) * 0.8), weight=False)

        with open(DATA_DIR / f"pk_{patch_id}.pkl", mode="wb") as f:
            pickle.dump(reg, f)  # the weighted linear regression

        if plot:
            plot_parallax_prediction(X, y, kde, ykr, yreg, yreg1, patch_id, ax=ax)

    if plot:
        plt.tight_layout()
        fig.savefig(PLOT_DIR / f"parallax-{'-'.join(map(str, patch_ids))}.pdf")


##############################################################################
# Command Line
##############################################################################


def make_parser(
    *, inheritable: bool = False, plot: bool = _PLOT,
) -> argparse.ArgumentParser:
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
    parser: |ArgumentParser|
        The parser with arguments:

        - plot
        - verbose

    ..
      RST SUBSTITUTIONS

    .. |ArgumentParser| replace:: `~argparse.ArgumentParser`

    """
    parser = argparse.ArgumentParser(
        description="",
        add_help=not inheritable,
        conflict_handler="resolve" if not inheritable else "error",
    )

    # order
    parser.add_argument("-o", "--order", default=4, type=int)

    # patches are done in batches. Need to decide the size
    parser.add_argument("-b", "--batch_size", default=10, type=int)

    # which patches
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--allsky", action="store_true",
                       help="Do all sky patches.")
    group.add_argument("--patches", action="append", type=int, nargs='+',
                       help="sky patch ids.")
    group.add_argument("-r", "--patches_range", type=int, nargs=2)

    # stars in gaia
    parser.add_argument("--random_index", default=None, type=int)

    # plot or not
    parser.add_argument("--plot", action="store", default=_PLOT, type=bool)

    # # script verbosity
    parser.add_argument("--filter_warnings", action="store_true")

    return parser


# /def


# ------------------------------------------------------------------------


def main(
    args: T.Union[list, str, None] = None,
    opts: T.Optional[argparse.Namespace] = None,
):
    """Script Function.

    Parameters
    ----------
    args : list or str or None, optional
        an optional single argument that holds the sys.argv list,
        except for the script name (e.g., argv[1:])
    opts : `~argparse.Namespace`| or None, optional
        pre-constructed results of parsed args
        if not None, used ONLY if args is None

        - nside

    """
    if opts is not None and args is None:
        pass
    else:
        if opts is not None:
            warnings.warn("Not using `opts` because `args` are given")
        if isinstance(args, str):
            args = args.split()

        parser = make_parser()
        opts = parser.parse_args(args)

    # /if

    # construct the list of batches of sky patches
    # [ (patch_1, patch_2, ...),  (patch_i, patch_i+1, ...)]
    if opts.allsky:
        nside = hp.order2nside(opts.order)
        npix = hp.nside2npix(nside)  # the number of sky patches
        nbatches = npix // opts.batch_size
        list_of_batches = np.array_split(np.arange(npix), nbatches)
    elif opts.patches_range:
        pi, pf = opts.patches_range
        if pi >= pf:
            raise ValueError("`patches_range` must be [start, stop], with stop > start.")
        nbatches = (pf - pi) // opts.batch_size
        list_of_batches = np.array_split(np.arange(pi, pf), nbatches)
    elif opts.patches:
        list_of_batches = opts.patches

    # optionally ignore warnings
    with warnings.catch_warnings():
        if opts.filter_warnings:
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)  # TODO!
            warnings.simplefilter("ignore", category=UserWarning)  # TODO!

        for batch in tqdm.tqdm(list_of_batches):
            query_and_fit_patch_set(tuple(batch), order=opts.order, plot=opts.plot, random_index=opts.random_index)


# /def


# ------------------------------------------------------------------------

if __name__ == "__main__":

    # call script
    main(args=None, opts=None)  # all arguments except script name


# /if


##############################################################################
# END
