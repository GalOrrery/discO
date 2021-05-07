# -*- coding: utf-8 -*-

"""Random Number Generators."""

__all__ = [
    "NumpyRNGContext",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T
import warnings

# THIRD PARTY
import numpy as np

##############################################################################
# PARAMETERS

RandomLike = T.Union[int, np.random.RandomState, None]

##############################################################################
# CODE
##############################################################################


class NumpyRNGContext:
    """
    A context manager (for use with the ``with`` statement) that will seed the
    numpy random number generator (RNG) to a specific value, and then restore
    the RNG state back to whatever it was before.

    This is primarily intended for use in the astropy testing suit, but it
    may be useful in ensuring reproducibility of Monte Carlo simulations in a
    science context.

    Parameters
    ----------
    seed : int or :class:`~numpy.random.RandomState` instance
        The value to use to seed the numpy RNG.

        .. warning::

            If using :class:`~numpy.random.RandomState`, this can be changed
            externally.

    Examples
    --------
    A typical use case might be::

        with NumpyRNGContext(<some seed value you pick>):
            from numpy import random

            randarr = random.randn(100)
            ... run your test using `randarr` ...

        #Any code using numpy.random at this indent level will act just as it
        #would have if it had been before the with statement - e.g. whatever
        #the default seed is.


    """

    def __init__(self, seed: RandomLike):
        self.seed = seed

    # /def

    def __enter__(self):
        """Start random state."""
        # store old state
        self.startstate = np.random.get_state()

        if isinstance(self.seed, int) or self.seed is None:
            np.random.seed(seed=self.seed)
        elif isinstance(self.seed, np.random.RandomState):
            state = self.seed.get_state()
            np.random.set_state(state)

        elif isinstance(self.seed, np.random.Generator):
            warnings.warn("Can't set random context from Generator.")

    # /def

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context-manager.

        deletes stored random state.

        """
        # need to advance the random state by the same amount
        # if it was a random state
        if isinstance(self.seed, np.random.RandomState):
            self.seed.set_state(np.random.get_state())

        # reset global stat to starting value
        np.random.set_state(self.startstate)

        del self.seed

    # /def


# /class


# -------------------------------------------------------------------

##############################################################################
# END
