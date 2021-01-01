"""Gyrus recursively generates large-scale Nengo models using NumPy semantics."""

import logging

# Gyrus namespace (API).
from .auto import configure, vectorize
from .base import asoperator, fold
from .operators import (
    apply,
    broadcast_scalar,
    convolve,
    decode,
    filter,
    integrate,
    join,
    lti,
    multiply,
    neurons,
    pre,
    reduce_transform,
    slice,
    split,
    stimulus,
    transform,
)
from .optional import layer, tensor_node
from .version import version as __version__

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
