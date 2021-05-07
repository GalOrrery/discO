# -*- coding: utf-8 -*-

"""Progress bar, modified from :mod:`~emcee`."""

# BUILT-IN
import logging

__all__ = ["get_progress_bar"]
__credits__ = ["emcee"]

##############################################################################
# IMPORTS

from discO.setup_package import HAS_TQDM

if HAS_TQDM:
    import tqdm

##############################################################################
# CODE
##############################################################################


class _NoOpPBar(object):
    """This class implements the progress bar interface but does nothing"""

    def __init__(self):
        pass

    # /def

    def __enter__(self, *args, **kwargs):
        return self

    # /def

    def __exit__(self, *args, **kwargs):
        pass

    # /def

    def update(self, count):
        pass

    # /def


# /class

# -------------------------------------------------------------------


def get_progress_bar(display, total):
    """Get a progress bar interface with given properties

    If the tqdm library is not installed, this will always return a "progress
    bar" that does nothing.

    Args:
        display (bool or str): Should the bar actually show the progress? Or a
                               string to indicate which tqdm bar to use.
        total (int): The total size of the progress bar.

    """
    if display:
        if not HAS_TQDM:
            logging.warning(
                "You must install the tqdm library to use progress "
                "indicators with emcee"
            )
            return _NoOpPBar()
        else:
            if display is True:
                return tqdm.tqdm(total=total)
            else:
                return getattr(tqdm, "tqdm_" + display)(total=total)

    return _NoOpPBar()


# /def
