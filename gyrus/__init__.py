"""Gyrus recursively generates large-scale Nengo models using NumPy semantics."""

import logging

from .version import version as __version__

# Gyrus namespace (API).
from .auto import vectorize
from .base import asoperator, fold
from .operators import (
    apply,
    broadcast_scalar,
    decode,
    filter,
    integrate,
    join,
    lti,
    multiply,
    pre,
    reduce_transform,
    slice,
    split,
    stimulus,
    transform,
)

probe = base.Operator.probe
register_method = base.Operator.register_method
register_ufunc = base.Operator.register_ufunc


logger = logging.getLogger(__name__)
try:
    # Prevent output if no handler set
    logger.addHandler(logging.NullHandler())
except AttributeError:
    pass

__copyright__ = "2021, Applied Brain Research"
__license__ = "Free for non-commercial use; see LICENSE.rst"
