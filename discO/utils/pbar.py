# -*- coding: utf-8 -*-

"""Progress bar, modified from :mod:`~emcee`."""

# STDLIB
import logging

__all__ = ["get_progress_bar"]
__credits__ = ["emcee"]

##############################################################################
# IMPORTS

# LOCAL
from discO.setup_package import HAS_TQDM

if HAS_TQDM:
    # THIRD PARTY
    import tqdm

##############################################################################
# CODE
##############################################################################


class _NoOpProgressBar:
    """This class implements the progress bar interface but does nothing"""

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def update(self, count):
        pass


# /class

# -------------------------------------------------------------------


def get_progress_bar(display, total):
    """Get a progress bar interface with given properties

    If the tqdm library is not installed, this will always return a "progress
    bar" that does nothing.

    Parameters
    ----------
    display : bool or str
        Should the bar actually show the progress? Or a string to indicate which
        tqdm bar to use.
    total : int
        The total size of the progress bar.

    """
    if display:
        if not HAS_TQDM:
            logging.warning(
                "Install the tqdm library to use the progress bar.",
            )
            return _NoOpProgressBar()
        else:
            if display is True:
                return tqdm.tqdm(total=total)
            else:
                return getattr(tqdm, display).tqdm(total=total)

    return _NoOpProgressBar()


# /def
