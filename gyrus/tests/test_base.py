import numpy as np
import pytest

from gyrus import asoperator, pre, stimulus
from gyrus.base import Fold, Operator, lower_folds


def test_invalid_args():
    with pytest.raises(TypeError, match="must be an iterator of Operator types"):
        Operator([None])


def test_base_attributes():
    op = Operator([])
    assert op.input_ops == ()
    assert op.ndim == 0
    assert op.shape == ()
    assert str(op) == "Operator()"
    assert repr(op) == "Operator([])"


def test_scalar_input_op_attributes():
    stim = pre(np.ones(3))
    op = stim.decode()
    assert op.input_ops == (stim,)
    assert op.ndim == 0
    assert op.shape == ()
    assert op.size_out == 3
    assert str(op) == "Decode(Pre())"
    assert repr(op) == "Decode([Pre([])])"


def test_fold_attributes():
    op = Fold([])
    assert op.input_ops == ()
    assert op.ndim == 1
    assert op.shape == (0,)
    assert str(op) == "Fold()"
    assert repr(op) == "Fold([])"
    assert not len(op.array)
    assert not len(op.size_out)


def test_array_input_op_attributes():
    shape = (4, 5, 1)
    op = stimulus(np.ones(shape))
    assert op.input_ops == tuple(op)
    assert op.ndim == len(shape)
    assert op.shape == shape
    assert type(op[0, 0, 0]).__name__ == "Stimulus"

    n = 0
    for input_op in op:
        assert input_op.ndim == len(shape) - 1
        assert input_op.shape == shape[1:]
        n += 1
    assert len(op) == n == shape[0]

    assert np.all(op.size_out == np.ones(shape))


def test_foldable():
    valid = (pre(1), pre(2))
    assert Fold.is_foldable(valid)

    invalid = Fold(valid)
    assert not Fold.is_foldable(invalid)

    with pytest.raises(TypeError, match="and not be a Fold"):
        Fold(invalid)


def test_array_not_implemented():
    with pytest.raises(TypeError, match="no implementation found"):
        np.vdot(stimulus(np.eye(3)), np.ones(3))


def test_fold_functional():
    u = np.arange(3 * 4 * 5).reshape((3, 4, 5))
    stim = stimulus(u)
    assert isinstance(stim, Fold)
    assert np.all(np.asarray(stim.run(1, 1)).squeeze(axis=(-2, -1)) == u)


def test_str():
    assert str(stimulus(np.ones((1, 2)))) == "Fold(Fold(Stimulus(), Stimulus()))"
    assert str(stimulus(np.ones((1, 1, 1)))) == "Fold(Fold(Fold(Stimulus())))"
    assert (
        str(stimulus(np.ones((1, 1, 2)))) == "Fold(Fold(Fold(Stimulus(), Stimulus())))"
    )
    assert str(stimulus(np.ones((1, 1, 1, 1)))) == "Fold(Fold(Fold(Fold(...))))"

    assert (
        stimulus(np.ones(3)).join().__str__(max_width=2)
        == "Join1D(Stimulus(), Stimulus(), ...)"
    )
    assert (
        pre(np.ones(2)).split().__str__(max_depth=2, max_width=2)
        == "Fold(Slice(...), Slice(...))"
    )


def test_repr():
    assert (
        repr(stimulus([1, 1]) * 3)
        == "Fold([Transforms([Stimulus([])]), Transforms([Stimulus([])])])"
    )


def test_asoperator():
    stim = stimulus(0)
    assert asoperator(stim) is stim

    input_ops = (stimulus(0), stimulus(1))
    op = asoperator(input_ops)
    assert isinstance(op, Fold)
    assert op.input_ops == input_ops

    assert asoperator(np.asarray(stim)) is stim

    with pytest.raises(TypeError, match="expected array scalar .* to be an Operator"):
        asoperator(np.asarray(0))


def test_lower_folds_not_implemented():
    a = pre(0)
    with pytest.raises(TypeError, match="all returned NotImplemented"):
        a @ a


def test_lower_folds():
    @lower_folds
    def f(a, b=0):
        assert not isinstance(a, Fold)
        assert not isinstance(b, Fold)
        return a

    f(stimulus(np.ones((3, 1))))
    f(pre(0), b=stimulus(np.ones((3, 1))))
