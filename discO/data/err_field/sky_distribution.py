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
import typing as T

# THIRD PARTY
import healpy as hp
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from astropy import table
from gaia_tools.query import query as do_query

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

# gaia_tools doesn't have ``GAIA_HEALPIX_INDEX``, so we use the equivalent
# formula source_id / 2^(35 + (12 - order) * 2)
# see https://www.gaia.ac.uk/data/gaia-data-release-1/adql-cookbook
ADQL_QUERY = """
SELECT
source_id, hpx{order},
parallax, parallax_error,
ra, ra_error,
dec, dec_error

FROM (
    SELECT
    source_id, random_index,
    TO_INTEGER(FLOOR(source_id/POWER(35+(12-{order})*2, 2))) AS hpx{order},
    parallax, parallax_error,
    ra, ra_error,
    dec, dec_error

    FROM gaiadr2.gaia_source AS gaia
) AS gaia

WHERE parallax >= 0
{random_index}

ORDER BY hpx{order};
"""

##############################################################################
# CODE
##############################################################################


def query_sky_distribution(
    order: int = 6,
    random_index: T.Optional[int] = None,
    *,
    plot: bool = True,
    use_local: bool = True,
    user: str = "postgres",
) -> None:
    """Query Sky and save number count.

    Parameters
    ----------
    order : int, optional
    random_index : int, optional

    plot : bool (optional, keyword-only)
    use_local : bool (optional, keyword-only)

    Returns
    -------
    sky : `~astropy.tables.QTable`
        Grouped by
    """
    # make ADQL
    random_index = "" if random_index is None else f"AND random_index < {int(random_index)}"
    adql_query = ADQL_QUERY.format(order=order, random_index=random_index)

    # data folder
    FOLDER = THIS_DIR / f"order_{order}"
    FOLDER.mkdir(exist_ok=True)

    # data file
    DATA_DIR = FOLDER / f"sky_distribution_{order}.ecsv"

    try:
        result = table.QTable.read(DATA_DIR)
    except Exception as e:
        result = do_query(
            adql_query, local=use_local, use_cache=False, user=user, verbose=True, timeit=True
        )
        result.write(DATA_DIR)

    # group by healpix index
    sky = result.group_by(f"hpx{order}")

    if plot:

        PLOT_DIR = FOLDER / "figures"
        PLOT_DIR.mkdir(exist_ok=True)

        # get unique ids
        patchids, hpx_indices, num_counts_per_pixel = np.unique(
            sky[f"hpx{order}"].value, return_index=True, return_counts=True
        )

        # ----------------
        # plot mollweide

        fig = plt.figure()
        ax = fig.add_subplot(
            title="Number of Counts per Pixel",
            xlabel="Number of Counts",
            ylabel=f"Frequency / {num_counts_per_pixel.sum()}",
        )
        ax.hist(num_counts_per_pixel, bins=50, log=True)
        fig.savefig(PLOT_DIR / f"num_counts_per_pixel_{order}.pdf")
        plt.close(fig)

        # ----------------
        # plot mollweide

        fig = plt.figure(figsize=(10, 10), facecolor="white")
        nside = hp.order2nside(order)
        npix = hp.nside2npix(nside)

        ma = np.zeros(npix)
        ma[patchids] = num_counts_per_pixel / num_counts_per_pixel.sum()
        ma[ma == 0] = hp.UNSEEN

        hp.mollview(
            ma,
            nest=True,
            coord=["C"],
            cbar=True,
            cmap="Greens",
            fig=fig,
            title=f"Star Count Fraction (Nest {order}, Mollweide)",
            norm=colors.LogNorm(),
            badcolor="white",
        )
        fig.savefig(PLOT_DIR / f"sky_distribution_{order}.pdf")
        plt.close(fig)

    return sky


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

    # stars in gaia
    parser.add_argument(
        "-i",
        "--random_index",
        default=int(2e6),
        type=int,
        help="limit queried stars within random index",
    )

    # plot or not
    parser.add_argument("--plot", default=True, type=bool, help="make plots or not")

    # gaia_tools
    parser.add_argument("--use_local", action="store_true", help="gaia_tools local query")
    parser.add_argument(
        "--username", default="postgres", type=str, help="gaia_tools query username"
    )

    return parser


# /def


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

    sky = query_sky_distribution(
        order=ns.order,
        random_index=ns.random_index,
        plot=ns.plot,
        use_local=ns.use_local,
        user=ns.username,
    )

    return sky


# /def


# ------------------------------------------------------------------------

if __name__ == "__main__":

    # call script
    main(args=None, opts=None)  # all arguments except script name


# /if

##############################################################################
# END
