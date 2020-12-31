import nengo
import pytest

from gyrus import configure, stimulus
from gyrus.auto import Configure


def test_invalid_parameter():
    stim = stimulus(1)
    with pytest.raises(ValueError, match="not configurable"):
        stim.configure(dimensions=2)


def test_config_basic():
    config = dict(n_neurons=400, radius=3, normalize_encoders=False)
    stim = stimulus(0).configure(**config)

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
    stim = configure(stimulus([0, 1, 2]), n_neurons=250).decode()
    assert len(stim) == 3
    for op in stim:
        assert op._impl_kwargs == {"n_neurons": 250}


def test_config_left_to_right_precedence():
    x1 = stimulus(0).configure(n_neurons=300)
    x2 = stimulus(1).configure(n_neurons=500)

    assert (x1 * x2)._impl_kwargs == {"n_neurons": 300}
    assert (x2 * x1)._impl_kwargs == {"n_neurons": 500}

    assert Configure.resolve_config([x1, x2]) == {"n_neurons": 300}
    assert Configure.resolve_config([x2, x1]) == {"n_neurons": 500}


def test_config_downstream_precedence():
    stim = stimulus(0).configure(n_neurons=200)
    assert stim.config == {"n_neurons": 200}
    x = stim.configure(n_neurons=300)
    y = x.transform(2)
    z = y ** 2
    assert z._impl_kwargs == {"n_neurons": 300}
    assert Configure.resolve_config([z]) == {"n_neurons": 300}
    assert z in Configure._cache

    a = stimulus(0).configure(n_neurons=400, radius=2)
    assert Configure.resolve_config([z + a]) == {"n_neurons": 300, "radius": 2}
    assert Configure.resolve_config([a + z]) == {"n_neurons": 400, "radius": 2}

    # Test hard reset of config downstream.
    b = (z + a).configure(reset=True, radius=3)
    assert b.config == {"radius": 3}
    assert Configure.resolve_config([b]) == {"radius": 3}
