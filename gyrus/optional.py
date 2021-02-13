import sys

import nengo
import numpy as np
from nengo.rc import rc

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
        tensor_func,
        shape_in=(node.size_out,),
        shape_out=shape_out,
        pass_time=bool(pass_time),  # https://github.com/nengo/nengo/issues/1667
    )
    nengo.Connection(node, out, synapse=None)
    return out


class KerasOptimizerSynapse(nengo.synapses.Synapse):
    """Nengo synapse that updates its state using a Keras optimizer.

    This can be used to implement gradient descent over time within the state of a
    Nengo synapse. [1]_ Note that the gradient's sign is flipped from the usual
    convention in Keras/TensorFlow, and scaled by the time-step (``dt``). This makes it
    somewhat interchangeable with typical synapses in Nengo within the context of
    Principle 3 from the Neural Engineering Framework.

    References
    ----------
    .. [1] Synaptic Descent. US Provisional Patent Application. 63/078,200.
       Voelker and Eliasmith, 2020.
    """

    def __init__(self, optimizer, *args, **kwargs):
        self.optimizer = optimizer
        self.tf = _import_or_fail(
            "tensorflow",  # dependency of nengo-dl
            fail_msg=f"tensorflow is required by the '{type(self)}' synapse",
        )
        super().__init__(*args, **kwargs)

    def make_state(self, shape_in, shape_out, dt, dtype=None, y0=0):
        assert shape_in == shape_out
        dtype = rc.float_dtype if dtype is None else np.dtype(dtype)
        X = y0 * np.ones(shape_out, dtype=dtype)
        return {"X": X}

    def make_step(self, shape_in, shape_out, dt, rng, state):
        # TODO: This probably doesn't do the right thing if you reset/pickle the
        #  simulator (since we're creating a new state variable, instead of using the
        #  underlying signal).
        x = self.tf.Variable(state["X"])

        def step(t, signal):
            self.optimizer.apply_gradients([(-dt * signal, x)])
            return x.numpy()

        return step


# Explicitly copy the signature since functools.wraps would require the tf module to
# already be in the namespace, and that is an optional dependency.
def Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
    **kwargs,
):
    """Wraps the ``tf.keras.optimizers.Adam`` optimizer in a Nengo synapse."""
    tf = _import_or_fail(
        "tensorflow",  # dependency of nengo-dl
        fail_msg=f"tensorflow is required by the Adam optimizer",
    )
    return KerasOptimizerSynapse(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
            **kwargs,
        )
    )
