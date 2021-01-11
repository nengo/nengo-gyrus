import nengo
import numpy as np
import pytest

from gyrus import Adam, stimulus, vectorize
from gyrus.nengo_helpers import explicit_passthrough
from gyrus.optional import _import_or_fail


def test_import_or_fail():
    with pytest.raises(
        RuntimeError, match="could not import _nonexistent_package; unit test check"
    ):
        _import_or_fail("_nonexistent_package", fail_msg="unit test check")


def test_tensornode():
    nengo_dl = pytest.importorskip("nengo_dl")
    import tensorflow as tf  # required by nengo_dl

    u = np.linspace(-1, 1, 1000)
    x = stimulus(u)
    y = x.tensor_node(tf.exp, pass_time=False)

    with tf.device("/cpu:0"):
        out = y.run(1, 1, simulator=nengo_dl.Simulator)

    assert np.allclose(out.squeeze(axis=0), np.exp(u))


def test_layer():
    nengo_dl = pytest.importorskip("nengo_dl")
    import tensorflow as tf  # required by nengo_dl

    u = np.linspace(-1, 1, 1000)
    x = stimulus(u)
    y = x.layer(tf.exp, shape_in=u.shape)

    with tf.device("/cpu:0"):
        out = y.run(1, 1, simulator=nengo_dl.Simulator)

    assert np.allclose(out.squeeze(axis=0), np.exp(u))


def test_keras_optimizer_synapse(plt):
    tf = pytest.importorskip("tensorflow")

    @vectorize
    def descent(size_out, function, synapse=Adam()):
        out = explicit_passthrough(size_in=size_out)
        nengo.Connection(out, out, function=function, synapse=synapse)
        return out

    # Local optimum of this gradient is (0.5, -0.25).
    x = descent(2, function=lambda x: [0.5 - x[0], -0.25 - x[1]])
    with tf.device("/cpu:0"):
        y = np.asarray(x.run(2))

    plt.plot(y)

    assert np.allclose(y[0], [0, 0])
    assert np.allclose(y[-1], [0.5, -0.25])
