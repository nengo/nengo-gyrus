import nengo
import numpy as np
import pytest
from nengo.exceptions import ValidationError

from gyrus.nengo_helpers import get_transform_size_out, validate_function_size


def test_validate_function_size_basic():
    def f(x, a=0):
        return x + a

    assert validate_function_size(f, input_shape=10) == 10


def test_validation_function_size_invalid():
    def f():
        return 0

    with pytest.raises(ValidationError, match="must accept a single np.array"):
        validate_function_size(f, input_shape=1)

    def g(x):
        return x[:, None]

    with pytest.raises(ValidationError, match="array_like with at most one axis"):
        validate_function_size(g, input_shape=1)


def _check_nengo_transform(tr, size_in, size_out):
    with nengo.Network():
        a = nengo.Node(np.zeros(size_in))
        b = nengo.Node(size_in=size_out)
        nengo.Connection(a, b, transform=tr)


def test_transform_sizes():
    tr = 1
    size_in = 1
    size_out = get_transform_size_out(tr, size_in)
    assert size_out == size_in
    _check_nengo_transform(tr, size_in, size_out)

    size_in = 3
    size_out = get_transform_size_out(tr, size_in)
    assert size_out == size_in
    _check_nengo_transform(tr, size_in, size_out)

    tr = [1, 1, 1]
    size_in = 1
    with pytest.raises(ValueError, match="to have size_in=1, not size_in=3"):
        get_transform_size_out(tr, size_in)
    with pytest.raises(
        ValidationError, match=r"does not match expected shape \(3, 1\)"
    ):
        _check_nengo_transform(tr, size_in, size_out)

    size_in = 3
    size_out = get_transform_size_out(tr, size_in)
    assert size_out == size_in
    _check_nengo_transform(tr, size_in, size_out)

    size_out = 3
    size_in = 4
    tr = nengo.transforms.Dense(shape=(size_out, size_in), init=nengo.dists.Choice([0]))
    assert size_out == get_transform_size_out(tr, size_in)
    _check_nengo_transform(tr, size_in, size_out)

    with pytest.raises(ValueError, match="to have size_in=5, not size_in=4"):
        get_transform_size_out(tr, size_in + 1)
    with pytest.raises(
        ValidationError,
        match=r"Transform input size \(4\) not equal to Node output size \(5\)",
    ):
        _check_nengo_transform(tr, size_in + 1, size_out)

    tr = np.ones((size_out, size_in))
    assert size_out == get_transform_size_out(tr, size_in)
    _check_nengo_transform(tr, size_in, size_out)

    with pytest.raises(ValueError, match="to have size_in=5, not size_in=4"):
        get_transform_size_out(tr, size_in + 1)
    with pytest.raises(
        ValidationError, match=r"does not match expected shape \(3, 5\)"
    ):
        _check_nengo_transform(tr, size_in + 1, size_out)

    tr = np.ones((1, 1, 1))
    with pytest.raises(ValueError, match="to have ndim of at most 2, not 3"):
        get_transform_size_out(tr, size_in=1)
    with pytest.raises(
        ValidationError, match=r"does not match expected shape \(1, 1\)"
    ):
        _check_nengo_transform(tr, 1, 1)
