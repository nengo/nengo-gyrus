import sys

import nengo

from gyrus.auto import vectorize
from gyrus.base import Operator


def _import_or_fail(module_name, fail_msg):
    """Imports and returns the module with the given name, or raises a RuntimeError."""
    try:
        __import__(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"could not import {module_name}; {fail_msg}\n"
            f"missing module(s) may be installed via the extra named 'optional'"
        ) from exc
    return sys.modules[module_name]


@Operator.register_method("layer")
@vectorize("Layer", excluded=["shape_in"])
def layer(node, layer_func, *, transform=nengo.params.Default, shape_in=None):
    """Operator that applies a nengo_dl.Layer to each output."""
    nengo_dl = _import_or_fail(
        "nengo_dl", fail_msg="nengo_dl.Layer is required by the 'layer' operator"
    )
    return nengo_dl.Layer(layer_func)(node, transform=transform, shape_in=shape_in)


@Operator.register_method("tensor_node")
@vectorize("TensorNode", excluded=["shape_out"])
def tensor_node(
    node, tensor_func, *, shape_out=nengo.params.Default, pass_time=nengo.params.Default
):
    """Operator that applies a nengo_dl.TensorNode to each output."""
    nengo_dl = _import_or_fail(
        "nengo_dl",
        fail_msg="nengo_dl.TensorNode is required by the 'tensor_node' operator",
    )
    out = nengo_dl.TensorNode(
        tensor_func, shape_in=(node.size_out,), shape_out=shape_out, pass_time=pass_time
    )
    nengo.Connection(node, out, synapse=None)
    return out
