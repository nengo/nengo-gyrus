"""Gyrus recursively generates large-scale Nengo models using NumPy semantics."""

import logging

# Gyrus namespace (API).
from .auto import configure, vectorize
from .base import asoperator, fold
from .neurons import Parabola
from .operators import (
    apply,
    broadcast_scalar,
    bundle,
    convolve,
    decode,
    filter,
    integrate,
    lti,
    multiply,
    neurons,
    reduce_transform,
    slice,
    stimuli,
    stimulus,
    transform,
    unbundle,
)
from .optional import KerasOptimizerSynapse, layer, tensor_node
from .version import version as __version__

probe = base.Operator.probe
register_method = base.Operator.register_method
register_ufunc = base.Operator.register_ufunc


logger = logging.getLogger(__name__)
try:
    # Prevent output if no handler set
    logger.addHandler(logging.NullHandler())
except AttributeError:  # pragma: no cover
    pass

__copyright__ = "2021, Applied Brain Research"
__license__ = "Free for non-commercial use; see LICENSE.rst"
