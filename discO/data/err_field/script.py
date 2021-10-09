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

# PROJECT-SPECIFIC
from .sky_distribution import main as sky_distribution_main

##############################################################################
# PARAMETERS

RandomStateType = T.Union[
    None,
    int,
    np.random.RandomState,
    np.random.Generator,
]

# General
THIS_DIR = pathlib.Path(__file__).parent

##############################################################################
# CODE
##############################################################################


def fit_kernel_ridge(
    X: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    train_size: int,
    random_state: RandomStateType = None,
) -> T.Tuple[npt.NDArray[np.float_], KernelRidge]:
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
    idx = shuffle(
        np.arange(0, len(X)),
        random_state=random_state,
        n_samples=train_size,
    )

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


def fit_support_vector(
    X: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    train_size: int,
    random_state: RandomStateType = None,
) -> T.Tuple[npt.NDArray[np.float_], SVR]:
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
    idx = shuffle(
        np.arange(0, len(X)),
        random_state=random_state,
        n_samples=train_size,
    )

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
    X: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    train_size: int,
    weight: T.Union[bool, npt.NDArray[np.float_]] = True,
    random_state: RandomStateType = None,
) -> T.Tuple[npt.NDArray[np.float_], LinearRegression]:
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
    idx = shuffle(
        np.arange(0, len(X)),
        random_state=random_state,
        n_samples=train_size,
    )

    # fit, optionally with weights
    if weight is True:
        xy: npt.NDArray[np.float_] = np.vstack([X[:, 2], y])
        wgt: npt.NDArray[np.float_] = gaussian_kde(xy)(xy)
        lr.fit(X[idx], y[idx], sample_weight=(1 / wgt)[idx])
    elif isinstance(weight, np.ndarray):
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


# ============================================================================


def plot_parallax_prediction(
    Xtrue: npt.NDArray[np.float_],
    ytrue: npt.NDArray[np.float_],
    kde: gaussian_kde,
    ypred1: npt.NDArray[np.float_],
    ypred2: npt.NDArray[np.float_],
    ypred3: npt.NDArray[np.float_],
    patch_id: int,
    ax: T.Optional[plt.Axes] = None,
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
    patch_id

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
    ax.set_title(f"Patch={patch_id}")

    # distance label
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


def plot_mollview(
    patch_ids: tuple[int, ...], order: int, fig: T.Optional[plt.Figure] = None
) -> plt.Figure:
    """Plot Mollweide view with patches on sky.

    Parameters
    ----------
    patch_ids : tuple[int]
        Set of patch ids (int).
    order : int
        The healpix order.  See :func:`healpy.order2nside`
    """
    npix = hp.nside2npix(hp.order2nside(order))

    # background plot
    m = np.arange(npix)
    alpha = np.zeros_like(m) + 0.5
    alpha[patch_ids[0] : patch_ids[-1]] = 1
    hp.mollview(m, nest=True, coord=["C"], cbar=False, cmap="inferno", fig=fig, alpha=alpha)

    # patch plot
    m[patch_ids[0] : patch_ids[-1]] = 3 * npix // 4
    alpha[: patch_ids[0]] = 0
    alpha[patch_ids[-1] :] = 0
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

    return fig


# /def


# ============================================================================


def query_and_fit_patch_set(
    patch_ids: tuple[int, ...],
    order: int,
    plot: bool,
    random_index: T.Optional[int] = 1000000,
) -> None:
    """Query and fit a set of sky patches.

    Parameters
    ----------
    patch_ids : tuple[int]
        Set of patch ids (int).
    order : int
        The healpix order.  See :func:`healpy.order2nside`
    """
    # create directories
    FOLDER = THIS_DIR / f"order_{order}"
    FOLDER.mkdir(exist_ok=True)

    PLOT_DIR = FOLDER / "figures"
    PLOT_DIR.mkdir(exist_ok=True)

    DATA_DIR = FOLDER / "pk_reg"
    DATA_DIR.mkdir(exist_ok=True)

    # -----------------------
    # Query batch

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
        query += f"AND random_index < {int(random_index)}"

    job = Gaia.launch_job_async(
        query,
        dump_to_file=False,
        verbose=False,
    )
    # perform query and...
    result = table.QTable(job.get_results(), copy=False)
    if len(result) == 0:
        warnings.warn(f"no data in patches: {patch_ids}")
        return

    rgr: table.QTable = result.group_by(hpl)  # group stars by patch

    # plot the patches
    if plot:
        fig = plt.figure()
        plot_mollview(patch_ids, order, fig=fig)
        fig.savefig(PLOT_DIR / f"mollview-{'-'.join(map(str, patch_ids))}.pdf")

    # -----------------------
    # Fits to each patch

    ax: T.Union[plt.Axes, None]
    axs: npt.NDArray[np.object_]  # axes or 0s
    if plot:  # set up parallax plots
        rows, remainder = np.divmod(len(patch_ids), 4)
        width = remainder if (rows == 0) else 4
        if remainder > 0:
            rows += 1
        fig, axs = plt.subplots(rows, width, figsize=(5 * width, 5 * rows))
    else:
        axs = np.array([None] * len(rgr.groups))  # noop for iteration

    key: table.Row
    grp: table.Table
    for grp, ax in zip(rgr.groups, axs.flat):  # iter thru patches
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


# /def


def make_groups(sky: table.QTable, order: int):
    """Make groups.

    Parameters
    ----------
    sky : `~astropy.table.QTable`
    order : int

    Returns
    -------
    groupsids : list[ndarray]
    """
    nside = hp.order2nside(order)
    npix = hp.nside2npix(nside)  # the number of sky patches

    # get healpix column name. it depends on the order, but is the group key.
    keyname = rgr.groups.keys.colnames[0]

    # get unique ids
    patchids, hpx_indices, num_counts_per_patch = np.unique(
        sky[keyname].value, return_index=True, return_counts=True
    )

    allpatchids = np.arange(npix)
    patchnums = np.ones(npix)
    patchnums[patchids] = num_counts_per_patch
    patchnums[patchnums == 0] = 1  # set minimum number of 'counts' to 1

    # sort by number of counts
    sorter = np.argsort(patchnums)[::-1]
    patchnums = patchnums[sorter]
    allpatchids = allpatchids[sorter]

    numgroups = 200
    threshold = patchnums.sum() // numgroups

    # split arrays into numgroups
    patchnums_split = np.array_split(patchnums, numgroups)
    allpatchids_split = np.array_split(allpatchids, numgroups)

    # reverse every other, to try and even out the addition a little
    patchnums_split = [
        (group if not i % 2 else group[::-1]) for i, group in enumerate(patchnums_split)
    ]
    allpatchids_split = [
        (group if not i % 2 else group[::-1]) for i, group in enumerate(allpatchids_split)
    ]

    # turn back into 1 array
    patchnums = np.concatenate(patchnums_split)
    allpatchids = np.concatenate(allpatchids_split)

    groupsids = [allpatchids[i::numgroups] for i in range(numgroups)]

    # # plot the distribution of groups
    # groups = [patchnums[i::numgroups] for i in range(numgroups)]

    return groupsids


# /def

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

    # patches are done in batches. Needed unless all-sky.
    parser.add_argument(
        "-b",
        "--batch_size",
        default=30,
        type=int,
        help="number of patches in a batch",
    )

    # which patches
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--allsky", action="store_true", help="fit all sky patches")
    group.add_argument(
        "--patches",
        action="append",
        type=int,
        nargs="+",
        help="only fit specified sky patches by ID",
    )
    group.add_argument(
        "-r",
        "--patches_range",
        type=int,
        nargs=2,
        help="fit specified sky patches within range",
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
    parser.add_argument("--rng", default=0, type=int, help="random number generator")

    # plot or not
    parser.add_argument("--plot", default=True, type=bool, help="plot")

    # script verbosity
    parser.add_argument("--filter_warnings", action="store_true", help="filter warnings")

    # parallelize
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="whether to parallelize fitting the batches",
    )
    parser.add_argument(
        "--numcores",
        type=int,
        default=None,
        help="number of computer cores to use, if parallelizing",
    )

    # local query for background
    parser.add_argument("--use_local", default=True, type=bool, help="local query or not")

    return parser


# /def


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
    opts : `~argparse.Namespace`| or None, optional
        pre-constructed results of parsed args
        if not None, used ONLY if args is None

        - nside
    """
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

    # /if

    # -----------------------
    # make background distribution

    sky = sky_distribution_main(opts=ns)

    # -----------------------

    # random number generator
    rng = np.random.default_rng(ns.rng)

    # construct the list of batches of sky patches
    # [ (patch_1, patch_2, ...),  (patch_i, patch_i+1, ...)]
    if ns.allsky:
        list_of_batches = make_groups(sky, order=ns.order)
    elif ns.patches_range:
        # TODO! get sky-weighted groups
        pi, pf = ns.patches_range
        if pi >= pf:
            raise ValueError("`patches_range` must be [start, stop], with stop > start.")
        nbatches = (pf - pi) // ns.batch_size
        list_of_batches = np.array_split(np.arange(pi, pf), nbatches)
    elif ns.patches:
        list_of_batches = ns.patches

    list_of_batches = np.array(list_of_batches, dtype=object)

    # optionally ignore warnings
    with warnings.catch_warnings():
        if ns.filter_warnings:
            warnings.simplefilter(
                "ignore",
                category=UndefinedMetricWarning,
            )  # TODO!
            warnings.simplefilter("ignore", category=UserWarning)  # TODO!

        if ns.parallel:
            # TODO! not have galpy dependency just for this util
            # PROJECT-SPECIFIC
            from .multi import parallel_map

            def wrapped_query_and_fit_patch_set(batch: tuple[int, ...]) -> tuple[int, ...]:
                if len(batch) != 0:  # skip empty batch
                    query_and_fit_patch_set(
                        tuple(batch),
                        order=ns.order,
                        plot=False,  # FIXME! doesn't work with parallel map
                        random_index=ns.random_index,
                    )
                pbar.update(n=1)
                pbar.refresh()
                return batch

            # /def

            with tqdm.tqdm(total=len(list_of_batches)) as pbar:
                # TODO! switch to
                # https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.multiprocessing.Pool.map
                parallel_map(wrapped_query_and_fit_patch_set, list_of_batches, numcores=ns.numcores)

        else:
            for batch in tqdm.tqdm(list_of_batches):
                query_and_fit_patch_set(
                    tuple(batch),
                    order=ns.order,
                    plot=ns.plot,
                    random_index=ns.random_index,
                )


# /def


# ------------------------------------------------------------------------

if __name__ == "__main__":

    # call script
    main(args=None, opts=None)  # all arguments except script name


# /if


##############################################################################
# END
