# -*- coding: utf-8 -*-

"""Testing :mod:`~discO.utils.pbar`."""

__all__ = [
    "Test__NoOpProgressBar",
    "Test_get_progress_bar",
]


##############################################################################
# IMPORTS

# BUILT-IN
import logging

# THIRD PARTY
import pytest

# PROJECT-SPECIFIC
from discO.setup_package import HAS_TQDM
from discO.utils import pbar

##############################################################################
# PARAMETERS

LOGGER = logging.getLogger(__name__)

##############################################################################
# TESTS
##############################################################################


class Test__NoOpProgressBar:

    #################################################################
    # Method Tests

    def test___enter__and___exit__(self):
        """Test method ``__enter__``."""
        obj = pbar._NoOpProgressBar()
        with obj as pb:
            assert pb is obj

    # /def

    def test_update(self):
        """Test method ``update``."""
        obj = pbar._NoOpProgressBar()
        obj.update(1000)  # just don't fail

    # /def


# /class


# -------------------------------------------------------------------


class Test_get_progress_bar:
    """Test :func:`~discO.utils.pbar.get_progress_bar`."""

    def test_noop(self):
        pb = pbar.get_progress_bar(display=False, total=2)
        assert isinstance(pb, pbar._NoOpProgressBar)

    @pytest.mark.skipif(HAS_TQDM, reason="only if not has `tqdm`")
    def test_op_no_tqdm(self, caplog):
        pb = pbar.get_progress_bar(display=True, total=2)
        assert isinstance(pb, pbar._NoOpProgressBar)
        assert "Install the tqdm library" in caplog.text

    @pytest.mark.skipif(not HAS_TQDM, reason="only if has `tqdm`")
    def test_op_tqdm(self):
        # THIRD PARTY
        import tqdm

        pb = pbar.get_progress_bar(display=True, total=2)
        assert isinstance(pb, tqdm.tqdm)

        pb = pbar.get_progress_bar(display="notebook", total=2)
        assert isinstance(pb, tqdm.notebook.tqdm_notebook)


# /def


##############################################################################
# END
