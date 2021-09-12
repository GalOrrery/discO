# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Samplers and stuff for 3rd-party packages."""


__all__ = []


# LOCAL
from discO.setup_package import HAS_AGAMA, HAS_GALPY

if HAS_AGAMA:
    # LOCAL
    from . import agama

    __all__ += ["agama"]


if HAS_GALPY:
    # LOCAL
    from . import galpy

    __all__ += ["galpy"]


##############################################################################
# END
