import nengo
import numpy as np
from nengo.utils.ensemble import response_curves

from gyrus import Parabola


def test_parabola_response_curve(seed):
    lower, upper = 50, 100
    with nengo.Network(seed=seed) as model:
        x = nengo.Ensemble(
            250,
            1,
            neuron_type=Parabola(),
            max_rates=nengo.dists.Uniform(lower, upper),
        )

    with nengo.Simulator(model) as sim:
        _, a = response_curves(x, sim)

    max_rates = a.max(axis=0)
    assert max_rates.shape == (x.n_neurons,)
    assert np.all(max_rates >= lower)
    assert np.all(max_rates <= upper)

    min_rates = a.min(axis=0)
    assert np.all(min_rates >= 0)
    assert np.mean(min_rates) <= 1e-2


def test_parabola_max_rates_intercepts(rng):
    n_neurons = 1000
    neuron_type = Parabola()

    max_rates = rng.uniform(low=50, high=100, size=n_neurons)
    intercepts = rng.uniform(low=-1, high=1, size=n_neurons)

    gain, bias = neuron_type.gain_bias(max_rates, intercepts)

    check_max_rates, check_intercepts = neuron_type.max_rates_intercepts(gain, bias)

    assert np.allclose(max_rates, check_max_rates)
    assert np.allclose(intercepts, check_intercepts)


def test_perfect_square(plt):
    with nengo.Network() as model:
        u = nengo.Node(lambda t: 2 * t - 1)
        x = nengo.Ensemble(1, 1, neuron_type=Parabola(), encoders=[[1]], intercepts=[0])
        y = nengo.Node(size_in=1)

        nengo.Connection(u, x, synapse=None)
        nengo.Connection(
            x, y, function=np.square, solver=nengo.solvers.Lstsq(), synapse=None
        )

        p_u = nengo.Probe(u)
        p_y = nengo.Probe(y)

    with nengo.Simulator(model) as sim:
        sim.run(1)

    plt.plot(sim.data[p_y])
    plt.plot(sim.data[p_u] ** 2)

    assert np.allclose(sim.data[p_y], sim.data[p_u] ** 2)
