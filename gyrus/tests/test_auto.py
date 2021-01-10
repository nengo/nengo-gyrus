import nengo
import numpy as np
import pytest

from gyrus import configure, fold, stimuli, stimulus, vectorize
from gyrus.auto import Configure
from gyrus.base import Operator


def test_invalid_vectorize_op_input():
    class InvalidInput(Operator):
        @property
        def size_out(self):
            return [1, 1]

    with pytest.raises(ValueError, match="all input_ops produce an integer size_out"):
        stimuli(InvalidInput([]))


def test_invalid_vectorize_op_output():
    @vectorize
    def f():
        pass

    with pytest.raises(ValueError, match="output from generate defines size_out"):
        f()


def test_invalid_vectorize_folds():
    @vectorize("F", excluded=[0])
    def f(a):
        return a

    with pytest.raises(TypeError, match="instantiated with a Fold"):
        f(stimuli([0, 1]))


def test_invalid_vectorize_folds_jagged():
    @vectorize
    def f(a):
        return a

    with pytest.raises(TypeError, match="instantiated with a Fold"):
        f(fold([stimuli([0, 1]), stimuli(2)]))


def test_invalid_parameter():
    stim = stimuli(1)
    with pytest.raises(ValueError, match="not configurable"):
        stim.configure(dimensions=2)


def test_vectorize_kwargs():
    @vectorize("F")
    def f(node_a, node_b, *, tr=1):
        """f docstring"""
        assert node_a.size_out == node_b.size_out
        out = nengo.Node(size_in=node_a.size_out)
        nengo.Connection(node_a, out, transform=tr, synapse=None)
        nengo.Connection(node_b, out, transform=tr, synapse=None)
        return out

    op = f(stimulus(1), node_b=stimulus(2), tr=3)
    assert type(op).__name__ == "F"
    assert type(op).__doc__ == "f docstring"

    assert op.run(1, 1).squeeze(axis=1) == 9


def test_vectorize_bad_generate():
    stim = stimuli(0)
    with pytest.raises(RuntimeError, match="mismatch between number of op_indices"):
        stim.generate(stim)


def test_vectorize_example():
    def multiply_by_two(x):
        y = nengo.Node(size_in=x.size_out)
        nengo.Connection(x, y, transform=2, synapse=None)
        return y

    x = stimuli([1, 2, 3])
    y = vectorize(multiply_by_two)(x)
    assert np.all(np.asarray(y.run(1, dt=1)).squeeze(axis=(1, 2)) == [2, 4, 6])


def test_config_basic():
    config = dict(n_neurons=400, radius=3, normalize_encoders=False)
    stim = stimuli(0).configure(**config)

    extra = dict(radius=4, neuron_type=nengo.RectifiedLinear())
    x = stim.decode(**extra)
    assert x._impl_kwargs == {**config, **extra}

    with nengo.Network() as model:
        x.make()

    ensembles = model.all_ensembles
    assert len(ensembles) == 1
    assert ensembles[0].n_neurons == 400
    assert ensembles[0].radius == 4  # extra takes precedence
    assert ensembles[0].normalize_encoders == False
    assert ensembles[0].neuron_type == nengo.RectifiedLinear()


def test_config_fold():
    stim = configure(stimuli([0, 1, 2]), n_neurons=250).decode()
    assert len(stim) == 3
    for op in stim:
        assert op._impl_kwargs == {"n_neurons": 250}


def test_config_left_to_right_precedence():
    x1 = stimuli(0).configure(n_neurons=300)
    x2 = stimuli(1).configure(n_neurons=500)

    assert (x1 * x2)._impl_kwargs == {"n_neurons": 300}
    assert (x2 * x1)._impl_kwargs == {"n_neurons": 500}

    assert Configure.resolve_config([x1, x2]) == {"n_neurons": 300}
    assert Configure.resolve_config([x2, x1]) == {"n_neurons": 500}


def test_config_downstream_precedence():
    stim = stimuli(0).configure(n_neurons=200)
    assert stim.config == {"n_neurons": 200}
    x = stim.configure(n_neurons=300)
    y = x.transform(2)
    z = y ** 2
    assert z._impl_kwargs == {"n_neurons": 300}
    assert Configure.resolve_config([z]) == {"n_neurons": 300}
    assert z in Configure._cache

    a = stimuli(0).configure(n_neurons=400, radius=2)
    assert Configure.resolve_config([z + a]) == {"n_neurons": 300, "radius": 2}
    assert Configure.resolve_config([a + z]) == {"n_neurons": 400, "radius": 2}

    # Test hard reset of config downstream.
    b = (z + a).configure(reset=True, radius=3)
    assert b.config == {"radius": 3}
    assert Configure.resolve_config([b]) == {"radius": 3}
