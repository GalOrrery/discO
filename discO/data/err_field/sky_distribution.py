# -*- coding: utf-8 -*-

"""Gaia Error Field Script.

This script can be run from the command line with the following parameters:

Parameters
----------

"""

__all__ = ["make_parser", "main"]


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
from astropy.table import QTable
from gaia_tools.query import query as do_query

##############################################################################
# PARAMETERS

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
    CAST(FLOOR(source_id/POWER(2, 35+(12-{order})*2)) AS BIGINT) AS hpx6,
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
    verbose: bool = True,
) -> None:
    """Query sky and save number count.

    Parameters
    ----------
    order : int, optional
    random_index : int, optional

    plot : bool (optional, keyword-only)
        Whether to make plots from the query results.
    use_local : bool (optional, keyword-only)
        Perform the query on a local database or the Gaia server.
        See :func:`gaia_tools.query.query` for details.
    verbose : bool (optional, keyword-only)
        Script verbosity.

    Returns
    -------
    sky : `~astropy.tables.QTable`
        Grouped by healpix index.
    """
    # ----------------------
    # data folder
    FOLDER = THIS_DIR / f"order_{order}"
    FOLDER.mkdir(exist_ok=True)

    # data file
    DATA_DIR = FOLDER / f"sky_distribution_{order}.ecsv"

    if verbose:
        print(f"data will be saved to / read from {DATA_DIR}")

    # ----------------------
    # Perform query or load from file

    # make ADQL
    random_index = "" if random_index is None else f"AND random_index < {int(random_index)}"
    adql_query = ADQL_QUERY.format(order=order, random_index=random_index)

    try:
        result = QTable.read(DATA_DIR)
    except Exception as e:
        if verbose:
            print("starting query.")
        result = do_query(
            adql_query, local=use_local, use_cache=False, user=user, verbose=True, timeit=True
        )
        if verbose:
            print("finished query.")

        # ensure tight columns are int
        result["source_id"].dtype = int
        result[f"hpx{order}"].dtype = int

        # write so next time don't need to query
        if verbose:
            print("saving sky distribution table.")
        result.write(DATA_DIR)
    else:
        if verbose:
            print("loaded sky distribution table.")

    # group by healpix index
    sky = result.group_by(f"hpx{order}")

    if plot:
        if verbose:
            print("making plots.")

        # save plots in the same location as the data
        PLOT_DIR = FOLDER / "figures"
        PLOT_DIR.mkdir(exist_ok=True)

        # get healpix counts
        patchids, hpx_indices, num_counts_per_pixel = np.unique(
            sky[f"hpx{order}"].value, return_index=True, return_counts=True
        )

        # histogram of counts per pixel
        plot_hist_pixel_count(num_counts_per_pixel, saveloc=PLOT_DIR)

        # plot mollweide of sky colored by count
        plot_sky_mollview(num_counts_per_pixel, order, saveloc=PLOT_DIR)

    return sky


def plot_hist_pixel_count(num_counts_per_pixel: np.ndarray, saveloc: pathlib.Path) -> None:
    """Plot histogram of counts per pixel.

    Parameters
    ----------
    num_counts_per_pixel : ndarray[int]
    saveloc : path-like
    """
    # make plot
    fig = plt.figure()
    ax = fig.add_subplot(
        title="Number of Counts per Pixel",
        xlabel="Number of Counts",
        ylabel=f"Frequency / {num_counts_per_pixel.sum()}",
    )
    # plot histogram
    ax.hist(num_counts_per_pixel, bins=50, log=True)
    # save and close
    fig.savefig(saveloc / f"num_counts_per_pixel_{order}.pdf")
    plt.close(fig)


def plot_sky_mollview(num_counts_per_pixel: np.ndarray, order: int, saveloc: pathlib.Path) -> None:
    """Plot mollweide of sky colored by pixel count.

    Parameters
    ----------
    num_counts_per_pixel : ndarray[int]
    order : int
    saveloc : path-like
    """
    fig = plt.figure(figsize=(10, 10), facecolor="white")

    # calculate npix from order
    nside = hp.order2nside(order)
    npix = hp.nside2npix(nside)

    # create pixel map
    pmap = np.zeros(npix)
    pmap[patchids] = num_counts_per_pixel / num_counts_per_pixel.sum()
    pmap[pmap == 0] = hp.UNSEEN

    # plot
    hp.mollview(
        pmap,
        nest=True,
        coord=["C"],
        cbar=True,
        cmap="Greens",
        fig=fig,
        title=f"Star Count Fraction (Nest {order}, Mollweide)",
        norm=colors.LogNorm(),
        badcolor="white",
    )
    # save and close
    fig.savefig(saveloc / f"sky_distribution_{order}.pdf")
    plt.close(fig)


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
    """
    # make the argument parser
    parser = argparse.ArgumentParser(
        description="Query Gaia for the approximate density distribution of stars across the sky.",
        add_help=not inheritable,
        conflict_handler="resolve" if not inheritable else "error",
    )

    # healpix order. Order 6 has approximately 1 pixel per square degree.
    parser.add_argument("-o", "--order", default=6, type=int, help="healpix order")

    # random index = depth to query of stars in gaia
    parser.add_argument(
        "-i",
        "--random_index",
        default=int(2e6),
        type=int,
        help="limit queried stars within random index",
    )

    # plot or not
    parser.add_argument("--plot", default=True, type=bool, help="make plots or not")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose")

    # gaia_tools
    parser.add_argument(
        "--use_local", action="store_true", help="perform a local database query or query gaia"
    )
    parser.add_argument(
        "--username", default="postgres", type=str, help="gaia_tools query username"
    )

    return parser


# ------------------------------------------------------------------------


def main(
    args: T.Union[list[str], str, None] = None,
    opts: T.Optional[argparse.Namespace] = None,
) -> None:
    """Query Gaia for distribution of stars on the sky.

    Parameters
    ----------
    args : list or str or None, optional
        an optional single argument that holds the sys.argv list,
        except for the script name (e.g., argv[1:])
    opts : `~argparse.Namespace`| or None, optional
        pre-constructed results of parsed args
        if not None, used ONLY if args is None.

    Returns
    -------
    `astropy.table.QTable`
    """
    # parse args
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

    if verbose:
        print("Starting script for the sky distribution of stars in Gaia.")

    # query or load from
    sky = query_sky_distribution(
        order=ns.order,
        random_index=ns.random_index,
        plot=ns.plot,
        use_local=ns.use_local,
        user=ns.username,
        verbose=np.verbose,
    )

    return sky


# ------------------------------------------------------------------------

if __name__ == "__main__":

    # call script
    main(args=None, opts=None)  # all arguments except script name


##############################################################################
# END
