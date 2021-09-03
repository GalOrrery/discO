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
]


##############################################################################
# IMPORTS

# BUILT-IN
import argparse
import typing as T
import warnings

# THIRD PARTY
import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from astroquery.gaia import Gaia
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.utils import shuffle

##############################################################################
# PARAMETERS

RandomStateType = T.Union[None, int, np.random.RandomState, np.random.Generator]

# General
_PLOT: bool = True  # Whether to plot the output

# Log file
_VERBOSE: int = 0  # Degree of logfile verbosity

##############################################################################
# CODE
##############################################################################


def fit_kernel_ridge(
    X: npt.NDArray, y: npt.NDArray, train_size: int, random_state: RandomStateType = None
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
        param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)},
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
        ]
    ).T
    ykr = kr.predict(Xp)

    return ykr, kr


# /def


def fit_gaussian_process(
    X: npt.NDArray, y: npt.NDArray, train_size: int, random_state: RandomStateType = None
) -> (npt.NDArray, GaussianProcessRegressor):
    """Gaussian-Process Regression code.

    Parameters
    ----------
    X : ndarray
    y : ndarray
    train_size : int
    random_state : `numpy.random.Generator`, `numpy.random.RandomState`, int, or None (optional)

    Returns
    -------
    ykr : ndarray
    kr : `~sklearn.gaussian_process.GaussianProcessRegressor`
    """
    # estimator
    gpr = GaussianProcessRegressor(kernel=None)

    # randomize the data order
    idx = shuffle(np.arange(0, len(X)), random_state=random_state, n_samples=train_size)

    # fit
    gpr.fit(X[idx], y[idx])
    ygp = gpr.predict(Xp)

    return ygp, gpr


# /def


def fit_support_vector(
    X: npt.NDArray, y: npt.NDArray, train_size: int, random_state: RandomStateType = None
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
        ]
    ).T
    ysv = svr.predict(Xp)

    return ysv, svr


# /def


def fit_linear(
    X: npt.NDArray,
    y: npt.NDArray,
    train_size: int,
    weight: bool = True,
    random_state: RandomStateType = None,
) -> (npt.NDArray, LinearRegression):
    """Linear regression model.

    Parameters
    ----------
    X : ndarray
    y : ndarray
    train_size : int
    weight : bool, optional
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
    if weight == True:
        xy = np.vstack([X[:, 2], y])
        kde = gaussian_kde(xy)(xy)
        lr.fit(X[idx], y[idx], sample_weight=(1 / kde)[idx])
    else:
        lr.fit(X[idx], y[idx])

    # get predictions: ra & dec are at median value. parallax is linear
    Xp = np.array(
        [
            np.ones(100) * np.median(X[:, 0]),  # ra
            np.ones(100) * np.median(X[:, 1]),  # dec
            np.linspace(X[:, 2].min(), X[:, 2].max(), 100),  # p
        ]
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
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(
        xlabel=r"$\log_{10}$ parallax [mas]",
        ylabel=r"$\log_{10}$ parallax fractional error",
    )
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
        ]
    ).T

    ax.scatter(Xtrue[:, -1], ytrue, s=5, label="data", alpha=0.3, c=kde)
    ax.scatter(Xpred[:, -1], ypred1, s=5, label="kernel-ridge")
    ax.scatter(Xpred[:, -1], ypred2, s=5, label="linear model: density-weighting")
    ax.scatter(Xpred[:, -1], ypred3, s=5, label="linear model: no density weight")
    ax.set_title(str(fids))

    ax.set_ylim(-3, 3)
    ax.invert_xaxis()
    ax.legend()

    return fig


# /def


def plot_mollview(setofids, nside):

    npix = hp.nside2npix(nside)
    m = np.arange(npix)
    m[setofids[0] : setofids[-1]] = m.max()

    hp.mollview(
        m,
        title="Mollview image RING",
        nest=True,
        coord=["C"],
        cbar=False,
        cmap=cm,
    )

    fig = plt.gcf()
    return fig


def query_and_fit_patch_set():
    """Query and fit a set of sky patches.

    Parameters
    ----------
    
    """

    job = Gaia.launch_job_async(
        f"""
    SELECT
    source_id, GAIA_HEALPIX_INDEX(4, source_id) AS healpix4,
    parallax AS parallax, parallax_error AS parallax_error,
    ra, ra_error AS ra_err,
    dec, dec_error AS dec_err

    FROM gaiadr2.gaia_source

    WHERE GAIA_HEALPIX_INDEX(4, source_id) IN {setofids}
    AND parallax >= 0
    AND random_index < 1000000
    """,
        dump_to_file=False,
        verbose=False,
    )

    r = job.get_results()
    rgr = r.group_by("healpix4")

    plot_mollview(setofids, opts.nside)

    for j in range(0, len(setofids)):
        rg = rgr[rgr["healpix4"] == setofids[j]]

        print(setofids[j], len(rg))

        # DOING STUFF HERE
        # with catch_warnings(UserWarning):
        df = table.QTable(rg)

        df = df[np.isfinite(df["parallax"])]  # filter out NaN
        df = df[df["parallax"] > 0]  # positive parallax

        # add the fractional error
        df["parallax_frac_error"] = df["parallax_error"] / df["parallax"]

        X = np.array(
            [
                df["ra"].to_value(u.deg),
                df["dec"].to_value(u.deg),
                np.log10(df["parallax"].to_value(u.mas)),
            ]
        ).T
        y = np.log10(df["parallax_frac_error"].value.reshape(-1, 1))[:, 0]

        xy = np.vstack([X[:, 2], y])
        kde = gaussian_kde(xy)(xy)

        ykr, kr = kernel_ridge(X, y, train_size=int(len(rg) * 0.8))
        # ygp, gpr = Gauss_process(X,y, train_size)
        ysv, svr = support_vector(X, y, train_size=int(len(rg) * 0.8))
        yreg, reg = linear(X, y, train_size=int(len(rg) * 0.8))
        yreg1, reg1 = linear(X, y, train_size=int(len(rg) * 0.8), weight=False)

        with open("pk_reg/pk_" + str(setofids[j]) + ".pkl", mode="wb") as f:
            pickle.dump(reg, f)

        plot_parallax_prediction(X, y, kde, ykr, yreg, yreg1, setofids[j])


##############################################################################
# Command Line
##############################################################################


def make_parser(
    *, inheritable: bool = False, plot: bool = _PLOT, verbose: int = _VERBOSE
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

    # plot or not
    parser.add_argument("--plot", action="store", default=_PLOT, type=bool)

    # script verbosity
    parser.add_argument("-v", "--verbose", action="store", default=0, type=int)

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

    if hasattr(opts, "norder"):
        norder = opts.norder
        opts.nside = hp.order2nside(norder)  # converts norder to nside

    breakpoint()

    return

    for setofids in tqdm.tqdm(groups_of_setofids):
        query_and_fit_patch_set(setofids)


# /def


# ------------------------------------------------------------------------

if __name__ == "__main__":

    # call script
    main(args=None, opts=None)  # all arguments except script name


# /if


##############################################################################
# END
