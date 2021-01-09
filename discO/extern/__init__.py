# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Samplers and stuff for 3rd-party packages."""


__all__ = []


# PROJECT-SPECIFIC
from discO.setup_package import HAS_AGAMA, HAS_GALPY

if HAS_AGAMA:
    from .agama import AGAMAPotentialSampler

    __all__ += ["AGAMAPotentialSampler"]


if HAS_GALPY:
    from .galpy import GalpyPotentialSampler

    __all__ += ["GalpyPotentialSampler"]


##############################################################################
# END
