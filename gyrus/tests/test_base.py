import numpy as np
import pytest

from gyrus import asoperator, stimuli, stimulus
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
    stim = stimulus(np.ones(3))
    op = stim.decode()
    assert op.input_ops == (stim,)
    assert op.ndim == 0
    assert op.shape == ()
    assert op.size_out == 3
    assert str(op) == "Decode(Stimulus())"
    assert repr(op) == "Decode([Stimulus([])])"


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
    op = stimuli(np.ones(shape))
    assert op.input_ops == tuple(op)
    assert op.ndim == len(shape)
    assert op.shape == shape
    assert type(op[0, 0, 0]).__name__ == "Stimuli"

    n = 0
    for input_op in op:
        assert input_op.ndim == len(shape) - 1
        assert input_op.shape == shape[1:]
        n += 1
    assert len(op) == n == shape[0]

    assert np.all(op.size_out == np.ones(shape))


def test_foldable():
    valid = (stimulus(1), stimulus(2))
    assert Fold.is_foldable(valid)

    invalid = Fold(valid)
    assert not Fold.is_foldable(invalid)

    with pytest.raises(TypeError, match="and not be a Fold"):
        Fold(invalid)


def test_array_not_implemented():
    with pytest.raises(TypeError, match="no implementation found"):
        np.vdot(stimuli(np.eye(3)), np.ones(3))


def test_fold_functional():
    u = np.arange(3 * 4 * 5).reshape((3, 4, 5))
    stim = stimuli(u)
    assert isinstance(stim, Fold)
    assert np.all(np.asarray(stim.run(1, 1)).squeeze(axis=(-2, -1)) == u)


def test_str():
    assert str(stimuli(np.ones((1, 2)))) == "Fold(Fold(Stimuli(), Stimuli()))"
    assert str(stimuli(np.ones((1, 1, 1)))) == "Fold(Fold(Fold(Stimuli())))"
    assert str(stimuli(np.ones((1, 1, 2)))) == "Fold(Fold(Fold(Stimuli(), Stimuli())))"
    assert str(stimuli(np.ones((1, 1, 1, 1)))) == "Fold(Fold(Fold(Fold(...))))"

    assert (
        stimuli(np.ones(3)).bundle().__str__(max_width=2)
        == "Bundle1D(Stimuli(), Stimuli(), ...)"
    )
    assert (
        stimulus(np.ones(2)).unbundle().__str__(max_depth=2, max_width=2)
        == "Fold(Slice(...), Slice(...))"
    )


def test_repr():
    assert (
        repr(stimuli([1, 1]) * 3)
        == "Fold([Transforms([Stimuli([])]), Transforms([Stimuli([])])])"
    )


def test_asoperator():
    stim = stimuli(0)
    assert asoperator(stim) is stim

    input_ops = (stimuli(0), stimuli(1))
    op = asoperator(input_ops)
    assert isinstance(op, Fold)
    assert op.input_ops == input_ops

    assert asoperator(np.asarray(stim)) is stim

    with pytest.raises(TypeError, match="expected array scalar .* to be an Operator"):
        asoperator(np.asarray(0))


def test_lower_folds_not_implemented():
    a = stimulus(0)
    with pytest.raises(TypeError, match="all returned NotImplemented"):
        a @ a


def test_lower_folds():
    @lower_folds
    def f(a, b=0):
        assert not isinstance(a, Fold)
        assert not isinstance(b, Fold)
        return a

    f(stimuli(np.ones((3, 1))))
    f(stimulus(0), b=stimuli(np.ones((3, 1))))


def test_lower_folds_recursive():
    @lower_folds
    def f(a):
        assert isinstance(a, list)
        assert not any(isinstance(a_i, Fold) for a_i in a)
        return a

    f([stimuli(np.ones((3, 1))), stimuli(np.ones((3, 1)))])
