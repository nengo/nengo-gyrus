import numpy as np
import pytest

from gyrus import stimulus
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
