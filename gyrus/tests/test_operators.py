import nengo
import numpy as np
import pytest
from nengo.utils.numpy import is_array, rms

from gyrus import broadcast_scalar, bundle, convolve, fold, probe, stimuli, stimulus
from gyrus.auto import Configure
from gyrus.base import Operator
from gyrus.operators import Transforms, _Fold_array_functions


def setup_module():
    # To make the Nengo vs. Gyrus networks match exactly, we fix all seeds to be
    # zero for all ensembles (gyrus.decode) and product networks (gyrus.multiply).
    assert "seed" not in Configure._base_config
    Configure._base_config["seed"] = 0


def teardown_module():
    # Set the configuration back to default/random seed after these unit tests.
    Configure._base_config.pop("seed")


def test_broadcast_scalar():
    stim = fold(
        [
            [stimulus(np.zeros(2)), stimulus(np.zeros(3))],
            [stimulus(np.zeros(4)), stimulus(np.zeros(1))],
        ]
    )
    assert np.all(stim.size_out == [[2, 3], [4, 1]])
    (a11, a12), (a21, a22) = broadcast_scalar(2, stim.size_out).run(1, 1)
    assert np.all(a11 == [[2, 2]])
    assert np.all(a12 == [[2, 2, 2]])
    assert np.all(a21 == [[2, 2, 2, 2]])
    assert np.all(a22 == [[2]])

    with pytest.raises(TypeError, match="expected scalar, but got array"):
        broadcast_scalar(np.eye(3), size_out=(3, 3, 3))


def test_communication_channel(tau=0.1):
    input_function = lambda t: np.sin(2 * np.pi * t)
    tau = 0.1

    # Approach using Nengo.
    with nengo.Network(seed=0) as model:
        stim = nengo.Node(output=input_function)

        x = nengo.Ensemble(100, stim.size_out, seed=0)
        y = nengo.Node(size_in=stim.size_out)

        nengo.Connection(stim, x, synapse=None)
        nengo.Connection(x, y, synapse=tau)

        p_y = nengo.Probe(y)

    with nengo.Simulator(model, dt=0.005) as sim:
        sim.run(1)

    stim = stimuli(input_function)
    y = stim.decode(lambda x: x, n_neurons=100, seed=0).filter(tau)
    out = y.run(1, dt=0.005, seed=0)

    assert np.allclose(sim.data[p_y], out)


def test_multiple_outputs():
    input_functions = [
        lambda t: np.sqrt(t) * np.cos(2 * np.pi * t),
        lambda t: np.sqrt(t) * np.sin(2 * np.pi * t),
    ]
    n_neurons = 100

    with nengo.Network(seed=0) as model:
        model.config[nengo.Connection].synapse = None

        stim1 = nengo.Node(output=input_functions[0])
        stim2 = nengo.Node(output=input_functions[1])

        x1 = nengo.Ensemble(n_neurons, stim1.size_out, seed=0)
        x2 = nengo.Ensemble(n_neurons, stim2.size_out, seed=0)

        y = nengo.Node(size_in=1)
        y_hat = nengo.Node(size_in=1)
        error = nengo.Node(size_in=1)

        nengo.Connection(stim1, y, function=np.square)
        nengo.Connection(stim2, y, function=np.square)
        nengo.Connection(y, error)

        nengo.Connection(stim1, x1)
        nengo.Connection(stim2, x2)
        nengo.Connection(x1, y_hat, function=np.square)
        nengo.Connection(x2, y_hat, function=np.square)
        nengo.Connection(y_hat, error, transform=-1)

        p_y = nengo.Probe(y)
        p_y_hat = nengo.Probe(y_hat)
        p_error = nengo.Probe(error)

    with nengo.Simulator(model, dt=0.005) as sim:
        sim.run(1)

    out1 = sim.data[p_y], sim.data[p_y_hat], sim.data[p_error]

    # An explicit approach in Gyrus.
    x = stimuli(input_functions)
    y = x[0].apply(np.square) + x[1].apply(np.square)
    y_hat = x[0].decode(np.square, n_neurons=n_neurons, seed=0) + x[1].decode(
        np.square, n_neurons=n_neurons, seed=0
    )
    out2 = fold([y, y_hat, y - y_hat]).run(1, dt=0.005, seed=0)

    # Alternative operator overload approach in Gyrus.
    x = stimuli(input_functions)
    y = np.sum(x.apply(np.square))
    y_hat = np.sum(x ** 2)
    out3 = fold([y, y_hat, y - y_hat]).run(1, dt=0.005, seed=0)

    assert np.allclose(out1, out2)
    assert np.allclose(out2, out3)


def test_subtract_scalar():
    input_function = lambda t: (2 * t - 1) * np.pi
    n_neurons = 100
    tau = 0.005

    with nengo.Network(seed=0) as model:
        stim = nengo.Node(input_function)
        one = nengo.Node(1)

        x = nengo.Ensemble(n_neurons, stim.size_out, radius=np.pi, seed=0)
        y = nengo.Node(size_in=1)
        one_minus_y = nengo.Node(size_in=1)

        nengo.Connection(stim, x, synapse=None)
        nengo.Connection(x, y, function=np.sin, synapse=tau)

        nengo.Connection(one, one_minus_y, synapse=None)
        nengo.Connection(y, one_minus_y, synapse=None, transform=-1)

        p_one_minus_y = nengo.Probe(one_minus_y)

    with nengo.Simulator(model) as sim:
        sim.run(1.0)

    out1 = sim.data[p_one_minus_y]

    x = stimuli(input_function)
    y = x.decode(np.sin, n_neurons=n_neurons, radius=np.pi, seed=0).filter(tau)
    out2 = (1 - y).run(1)
    out3 = ((-y) + 1).run(1)

    assert np.allclose(out1, out2)
    assert np.allclose(out2, out3)


def test_shapes():
    input_function = [
        lambda t: [0, 1, 2],
        lambda t: [3],
        lambda t: [4, 5, 6, 7],
        lambda t: [8, 9],
    ]

    x = stimuli(input_function)
    assert len(x) == 4
    assert x.shape == (4,)
    assert np.all([inp.size_out for inp in x] == [3, 1, 4, 2])
    assert np.all(x.size_out == [3, 1, 4, 2])

    y = x.bundle()
    assert y.size_out == 10
    assert y[3:].size_out == 7
    assert y[5].size_out == 1

    y2 = x[:2].bundle()
    assert y2.size_out == 4

    assert np.all(y.run(1, dt=1) == np.arange(10))


def test_bundle_unbundle():
    data = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    stim = stimuli(data)
    t = 6

    y = stim.bundle()
    assert y.shape == (3, 4)
    out = np.asarray(y.run(t, 1))
    y_split = y.unbundle()
    assert y_split.shape == (3, 4, 5)
    y_check = y_split.bundle()
    assert y_check.shape == (3, 4)
    out_check = np.asarray(y_check.run(t, 1))
    assert np.allclose(out, out_check)
    out_check = np.asarray(y_split.run(t, 1)).squeeze(axis=-1).transpose((0, 1, 3, 2))
    assert out.shape == (3, 4, t, 5)
    assert np.allclose(out_check, out)
    for i in range(t):
        assert np.allclose(out[:, :, i, :], data)

    y = bundle(stim, axis=-2)
    assert y.shape == (3, 5)
    out = np.asarray(y.run(t, 1))
    y_split = y.unbundle()
    assert y_split.shape == (3, 5, 4)
    out_check = np.asarray(y_split.run(t, 1)).squeeze(axis=-1).transpose((0, 1, 3, 2))
    assert out.shape == (3, 5, t, 4)
    assert np.allclose(out_check, out)
    for i in range(t):
        assert np.allclose(out[:, :, i, :], data.transpose((0, 2, 1)))

    y_split = y.unbundle(axis=-2)
    assert y_split.shape == (3, 4, 5)
    out_check = np.asarray(y_split.run(t, 1)).squeeze(axis=-1)
    for i in range(t):
        assert np.allclose(out_check[..., i], data)

    y = bundle(stim, axis=0)
    assert y.shape == (4, 5)
    out = np.asarray(y.run(t, 1))
    y_split = y.unbundle()
    assert y_split.shape == (4, 5, 3)
    out_check = np.asarray(y_split.run(t, 1)).squeeze(axis=-1).transpose((0, 1, 3, 2))
    assert out.shape == (4, 5, t, 3)
    assert np.allclose(out_check, out)
    for i in range(t):
        assert np.allclose(out[:, :, i, :], data.transpose((1, 2, 0)))


def test_join_invalid():
    with pytest.raises(ValueError, match="expected all input ops to be an Operator"):
        bundle([stimuli(np.ones(3)), stimuli(np.ones(2))])

    with pytest.raises(ValueError, match="cannot bundle zero nodes"):
        bundle([])


def test_transpose_reshape_squeeze():
    shape = (4, 1, 2, 1, 6)
    data = np.arange(np.prod(shape)).reshape(shape)
    stim = stimuli(data)

    axes = (2, 0, 1, 3, 4)
    out = np.asarray(np.transpose(stim, axes=axes).run(1, 1)).squeeze(axis=(-2, -1))
    assert np.allclose(data.transpose(axes), out)

    newshape = (np.prod(shape[:3]), -1, shape[-1])
    out = np.asarray(stim.reshape(newshape=newshape).run(1, 1)).squeeze(axis=(-2, -1))
    assert np.allclose(data.reshape(newshape), out)

    axis = (1, 3)
    out = np.asarray(stim.squeeze(axis=axis).run(1, 1)).squeeze(axis=(-2, -1))
    assert np.allclose(data.squeeze(axis=axis), out)


def test_transpose():
    data = np.arange(3 * 4).reshape((3, 4))
    stim = stimuli(data)
    assert stim.T.shape == np.transpose(stim).shape == (4, 3)
    assert np.allclose(stim.T.run(1, 1), data.T[..., None, None])


def test_flatten():
    data = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    stim = stimuli(data)
    assert stim.shape == data.shape
    flat = stim.flatten()
    assert flat.shape == (data.size,)
    assert np.allclose(flat.run(1, 1), data.flatten()[:, None, None])


def test_multidimensional_function():
    input_function = lambda t: [t, -np.sqrt(t), t ** 3]
    n_neurons = 2000
    tau = 0.005

    with nengo.Network() as model:
        stim = nengo.Node(output=input_function)

        x = nengo.Ensemble(n_neurons, 3, radius=np.sqrt(3), seed=0)
        y = nengo.Node(size_in=1)

        nengo.Connection(stim, x, synapse=None)
        nengo.Connection(x, y, function=np.prod, synapse=tau)

        p = nengo.Probe(y)

    with nengo.Simulator(model) as sim:
        sim.run(1)

    out1 = sim.data[p]

    x = (
        stimuli(input_function)
        .decode(np.prod, n_neurons=n_neurons, radius=np.sqrt(3))
        .filter(tau)
    )
    out2 = x.run(1)

    assert np.allclose(out1, out2)


def test_multiple_ensembles():
    input_functions = [
        lambda t: [np.sin(t), np.cos(t), t, 1 - 2 * t ** 2],
        lambda t: [t, np.sin(t), 1 - 2 * np.cos(t) ** 2, t ** 3],
    ]
    d = 4
    tau1 = 0.01
    tau2 = 0.1

    with nengo.Network() as model:
        stim1 = nengo.Node(output=input_functions[0])
        stim2 = nengo.Node(output=input_functions[1])

        x1 = nengo.Ensemble(100, d, seed=0)
        x2 = nengo.Ensemble(100, d, seed=0)
        y = nengo.Ensemble(100, d, seed=0)

        nengo.Connection(stim1, x1, synapse=None)
        nengo.Connection(stim2, x2, synapse=None)

        nengo.Connection(x1, y, synapse=tau1)
        nengo.Connection(x2, y, synapse=tau1)

        p = nengo.Probe(y, synapse=tau2)

    with nengo.Simulator(model) as sim:
        sim.run(1)

    out1 = sim.data[p]

    stim = stimuli(input_functions)
    x1, x2 = stim.decode()
    y = (x1 + x2).filter(tau1).decode().filter(tau2)
    out2 = y.run(1)

    assert np.allclose(out1, out2)


def test_np_prod():
    input_functions = [
        lambda t: [1, -1, 0, 0.5],
        lambda t: [1, 0.5, -1, 1],
    ]
    d = 4
    tau = 0.1

    with nengo.Network() as model:
        stim1 = nengo.Node(output=input_functions[0])
        stim2 = nengo.Node(output=input_functions[1])

        x = nengo.networks.Product(200, dimensions=d, seed=0)

        nengo.Connection(stim1, x.input_a, synapse=None)
        nengo.Connection(stim2, x.input_b, synapse=None)

        p = nengo.Probe(x.output, synapse=tau)

    with nengo.Simulator(model) as sim:
        sim.run(1)

    out1 = sim.data[p]

    out2 = np.prod(stimuli(input_functions)).filter(tau).run(1)

    assert np.allclose(out1, out2)


def test_recursive_fold():
    shape = (2, 3, 4, 5)
    constants = np.arange(np.prod(shape)).reshape(shape)

    @np.vectorize
    def make_lambdas(x):
        return lambda _: x * np.ones(7)

    input_functions = make_lambdas(constants)
    assert input_functions.shape == shape
    stim = stimuli(input_functions)
    assert stim.shape == shape

    r = np.asarray(stim.run(6, dt=1))
    assert r.shape == shape + (6, 7)
    assert np.allclose(r, constants[..., None, None])

    # Test jagged dimensions across fold
    out1, out2 = stimuli([lambda t: [1, 3, 4], lambda t: [1, 2]]).run(1, dt=1)
    assert np.allclose(out1, [1, 3, 4])
    assert np.allclose(out2, [1, 2])


def test_stimulus_decorator():
    @stimuli
    def f(t):
        return (t + 1) ** 2

    assert np.allclose(f.run(1, 1), 4)


def matrix_vector_gyrus(A, x, t, tau):
    assert ((stimuli(np.ones(3)) @ np.ones(3)).run(1, 1)) == [[3.0]]

    y_true = A @ x

    A_check = stimuli(A).run(1, 1)
    assert np.allclose(A, np.asarray(A_check).squeeze(axis=(-2, -1)))

    y = (stimuli(A) @ x).run(1, 1)
    assert np.allclose(y_true, np.asarray(y).squeeze(axis=(-2, -1)))

    y = (A @ stimuli(x)).run(1, 1)
    assert np.allclose(y_true, np.asarray(y).squeeze(axis=(-2, -1)))

    y = (A.tolist() @ stimuli(x)).run(1, 1)  # invokes __rmatmul__
    assert np.allclose(y_true, np.asarray(y).squeeze(axis=(-2, -1)))

    y = (stimuli(A) @ stimuli(x)).filter(synapse=tau).run(t)
    return np.asarray(y).reshape(y_true.size, -1).T


def matrix_vector_nengo(A, x, t, tau):
    assert A.ndim == 3
    assert x.ndim == 1
    assert A.shape[-1] == x.shape[-1]

    with nengo.Network() as model:
        A_stims = np.empty(A.shape, dtype=nengo.Node)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                for k in range(A.shape[2]):
                    A_stims[i, j, k] = nengo.Node(A[i, j, k])

        x_stims = [nengo.Node(x_i) for x_i in x]

        y = nengo.Node(size_in=A.shape[0] * A.shape[1])
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                y_index = i * A.shape[1] + j
                for k in range(A.shape[2]):
                    product = nengo.networks.Product(200, dimensions=1, seed=0)
                    nengo.Connection(A_stims[i, j, k], product.input_a, synapse=None)
                    nengo.Connection(x_stims[k], product.input_b, synapse=None)
                    nengo.Connection(product.output, y[y_index], synapse=None)

        p_y = nengo.Probe(y, synapse=tau)

    with nengo.Simulator(model) as sim:
        sim.run(t)

    return sim.data[p_y]


def test_matrix_vector_multiply(rng):
    A = rng.uniform(-1, 1, size=(4, 2, 3))
    x = rng.uniform(-1, 1, size=3)
    t = 0.05
    tau = 0.005
    out1 = matrix_vector_nengo(A, x, t, tau)
    out2 = matrix_vector_gyrus(A, x, t, tau)
    assert np.allclose(out1, out2)


def test_jagged_probes():
    f = stimuli([lambda _: [1, 2, 3], lambda _: [4, 5]])
    y1, y2 = f.run(1, 1)
    assert np.allclose(y1.squeeze(axis=0), [1, 2, 3])
    assert np.allclose(y2.squeeze(axis=0), [4, 5])


def test_nengo_interoperability():
    input_function = lambda t: np.sin(2 * np.pi * t)

    def subnetwork(x):
        return (stimulus(x) ** 2).filter(synapse=0.005)

    with nengo.Network() as model:
        stim = nengo.Node(input_function)

        y = subnetwork(stim).make()

        p = nengo.Probe(y)
        p_func = probe(y)

    y_compare = subnetwork(input_function).run(1)

    with nengo.Simulator(model) as sim:
        sim.run(1)
        assert np.allclose(y_compare, sim.data[p])
        assert np.allclose(y_compare, p_func(sim))


def oscillator(hertz):
    radians = 2 * np.pi * hertz
    return [[0, -radians], [radians, 0]], [1, 0]


def test_oscillator():
    frequency = 4
    t = 1
    A, _ = oscillator(frequency)

    input_function = nengo.processes.Piecewise({0: [10, 0], 0.1: [0, 0]})
    kick = stimuli(input_function)

    x = kick.integrate(lambda x: x.decode().transform(A))
    y = x.run(t)

    with nengo.Network() as model:
        ens = nengo.Ensemble(100, dimensions=2, seed=0)
        kick = nengo.Node(input_function)
        psc = nengo.Node(size_in=2)

        integrator = nengo.synapses.LinearFilter([1], [1, 0])
        nengo.Connection(kick, psc, synapse=integrator)
        nengo.Connection(ens, psc, transform=A, synapse=integrator)
        nengo.Connection(psc, ens, synapse=None)

        ens_probe = nengo.Probe(psc)

    with nengo.Simulator(model) as sim:
        sim.run(t)

    assert np.allclose(sim.data[ens_probe], y)


def test_integrand_invalid():
    u = stimulus(0)
    with pytest.raises(TypeError, match="expected integrand to generate a single Node"):
        u.integrate(integrand=lambda x: fold([x, x]))

    with pytest.raises(TypeError, match="integrand returned Node with size_out=2"):
        u.integrate(integrand=lambda x: fold([x, x]).bundle())


def test_lti():
    dt = 0.001

    # The first time-step is t=dt.
    kick = stimuli(lambda t: 1 / dt if t <= dt else 0)

    hertz = 4
    t = 1
    system = oscillator(hertz=hertz)
    x1 = kick.lti(system).run(t, dt=dt)
    x2 = kick.lti(system, state=lambda x: x.decode(neuron_type=nengo.Direct())).run(
        t, dt=dt
    )

    # Ideal (does not suffer from ZOH integrating a piecewise step input):
    # phase = 2 * np.pi * hertz * np.arange(x1.shape[0] - 1) * dt
    # x = np.stack([np.cos(phase), np.sin(phase)], axis=-1)

    assert np.allclose(x1, x2)


def test_lti_invalid():
    u = stimulus(np.ones(3))
    A = -np.eye(2)
    B = np.ones((2, 3))
    x = u.lti(system=(A, B))

    with pytest.raises(ValueError, match="must be a square matrix"):
        u.lti(system=(B, B))

    with pytest.raises(ValueError, match="must be 1D or 2D"):
        u.lti(system=(A, 0))

    with pytest.raises(ValueError, match="must be an array of length 2"):
        u.lti(system=(A, B.T))

    with pytest.raises(ValueError, match="to have size_in=3, not size_in=2"):
        u.lti(system=(A, A))


def test_high_dimensional_integrator(plt):
    d = 32
    dt = 1e-3

    def f(t, hz):
        return np.cos(2 * np.pi * hz * (t - dt)) * (2 * np.pi * hz)

    input_functions = [lambda t, hz=hz: f(t, hz) for hz in np.linspace(1, 2, d)]
    u = stimuli(input_functions)
    x = u.integrate()

    y = np.asarray(x.run(1, dt=dt))
    y_hat = np.asarray(x.decode().filter(5e-3).run(1, dt=dt))
    assert y.shape == y_hat.shape

    t = (1 + np.arange(y.shape[1])) * dt
    y_check = np.cumsum(np.asarray([f(t) for f in input_functions]), axis=1) * dt
    assert np.allclose(y.squeeze(axis=2)[:, 1:], y_check[:, :-1])
    assert rms(y - y_hat) < 0.1

    for y_i, y_hat_i in zip(y, y_hat):
        plt.plot(y_i.squeeze(axis=1), linestyle="--")
        plt.plot(y_hat_i.squeeze(axis=1), alpha=0.7)
    plt.xlabel("Time-step")


def test_elementwise_multiply():
    d = 64
    a = np.linspace(-1, 1, d)
    b = a ** 2

    op_a = stimulus(a)
    assert op_a.size_out == d

    ab = op_a * b
    assert ab.size_out == d

    y = np.asarray(ab.run(1, 1)).squeeze(axis=0)
    assert y.shape == (d,)

    a2ab = fold([op_a, 2 * op_a]) * b
    y2 = np.asarray(a2ab.run(1, 1)).squeeze(axis=1)
    assert np.allclose(y2[0], y)
    assert np.allclose(y2[1], 2 * y2[0])

    y3 = np.asarray((b * op_a).run(1, 1)).squeeze(axis=0)
    assert np.allclose(y3, y)

    y_check = a ** 3
    assert np.allclose(y, y_check)

    # Test multiplying vector with a scalar constant.
    a10 = np.asarray((op_a * 10).run(1, 1)).squeeze(axis=0)
    assert np.allclose(a10, a * 10)


def test_multiply_invalid():
    a = stimulus(np.zeros(2))
    b = stimulus(np.zeros(3))
    with pytest.raises(ValueError, match="multiply operator size_out"):
        a * b

    with pytest.raises(TypeError, match="multiply size mismatch"):
        a * [0, 1, 2]

    with pytest.raises(TypeError, match="all returned NotImplemented"):
        a * np.eye(2)


def test_multiply_direct(rng):
    shape = (2, 1, 3)
    a = rng.randn(*shape)
    b = rng.randn(*shape)

    input_a = stimuli(a).configure(neuron_type=nengo.Direct())
    input_b = stimuli(b)

    y = input_a * input_b
    with nengo.Network() as model:
        y.make()

    for ens in model.all_ensembles:
        assert ens.neuron_type == nengo.Direct()
    assert len(model.all_ensembles) == 2 * np.prod(shape)

    out = np.asarray(y.run(1, 1))
    assert np.allclose(out.squeeze(axis=(-2, -1)), a * b)


def _numpy_convolve(a, b):
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b)).real


def test_convolve_direct(rng):
    dims = 64
    a = rng.randn(dims)
    b = rng.randn(dims)

    input_a = stimulus(a).configure(neuron_type=nengo.Direct())
    input_b = stimulus(b)

    y = convolve(input_a, input_b)
    assert y.size_out == dims

    out = y.run(1, 1).squeeze(axis=0)
    assert out.shape == (dims,)

    assert np.allclose(out, _numpy_convolve(a, b))


def test_convolve_invalid():
    a = stimulus(np.zeros(2))
    b = stimulus(np.zeros(3))
    with pytest.raises(ValueError, match="convolve operator size_out"):
        a.convolve(b)


def test_sum_dot():
    A = [[1, 2, 3], [4, 5, 6]]
    x = [7, 8, 9]
    y_check = np.dot(A, x)

    y1 = np.dot(stimuli(A), x)
    assert y1.shape == (2,)
    assert np.all(y1.size_out == [1, 1])
    out1 = np.asarray(y1.run(1, 1)).squeeze(axis=(-2, -1))
    assert np.allclose(out1, y_check)

    y2 = np.sum((stimuli(A).bundle() * np.asarray(x)).unbundle(), axis=1)
    assert y2.shape == (2,)
    out2 = np.asarray(y2.run(1, 1)).squeeze(axis=(-2, -1))
    assert np.allclose(out1, out2)

    y3 = (stimuli(A).bundle() * np.asarray(x)).transform(np.ones((1, len(x))))
    assert y3.shape == (2,)
    out3 = np.asarray(y3.run(1, 1)).squeeze(axis=(-2, -1))
    assert np.allclose(out2, out3)


def test_add():
    assert np.allclose(
        (stimulus([2, 3, 4]) + 1).run(1, 1).squeeze(axis=(0,)), [3, 4, 5]
    )
    assert np.allclose(
        ([2, 3, 4] + stimulus([1, 2, 3])).run(1, 1).squeeze(axis=(0,)), [3, 5, 7]
    )

    a = stimulus(1)
    assert a + 0 is 0 + a is a


def test_add_invalid():
    a = stimulus(np.ones(2))

    with pytest.raises(TypeError, match="add size mismatch"):
        a + [0, 1, 2]

    with pytest.raises(TypeError, match="all returned NotImplemented"):
        a + np.eye(2)

    with pytest.raises(TypeError, match="all returned NotImplemented"):
        a + "b"


def test_custom_subnetworks():
    synapse = 0.01

    def gyrus_custom_subnetwork(node_a, node_b, synapse):
        out_a = node_a ** 2
        out_b = (node_a * node_b - out_a) / 2
        return fold([out_a, out_b]).filter(synapse)

    def nengo_custom_subnetwork(node_a, node_b, synapse):
        assert node_a.size_out == node_b.size_out
        d = node_a.size_out

        product = nengo.networks.Product(200, dimensions=d, seed=0)
        ens = nengo.Ensemble(100, d, seed=0)

        nengo.Connection(node_a, ens, synapse=None)
        nengo.Connection(node_a, product.input_a, synapse=None)
        nengo.Connection(node_b, product.input_b, synapse=None)

        decode_a = nengo.Node(size_in=d)
        decode_b = nengo.Node(size_in=d)

        nengo.Connection(ens, decode_a, function=np.square, synapse=synapse)
        nengo.Connection(product.output, decode_b, transform=0.5, synapse=synapse)
        nengo.Connection(decode_a, decode_b, transform=-0.5, synapse=None)

        return [decode_a, decode_b]

    # Running it in Gyrus.
    input_function1 = lambda t: np.sin(2 * np.pi * t)
    input_function2 = lambda t: t

    x1 = stimuli(input_function1)
    x2 = stimuli(input_function2)

    y = gyrus_custom_subnetwork(x1, x2, synapse)

    out1, out2 = y.run(1)

    # Running it in Nengo.
    with nengo.Network() as model:
        stim1 = nengo.Node(input_function1)
        stim2 = nengo.Node(input_function2)

        y1, y2 = nengo_custom_subnetwork(stim1, stim2, synapse)

        p1 = nengo.Probe(y1)
        p2 = nengo.Probe(y2)

    with nengo.Simulator(model) as sim:
        sim.run(1)

    assert np.allclose(sim.data[p1], out1)
    assert np.allclose(sim.data[p2], out2)


def test_label():
    assert stimulus(1).label == "Stimulus()"
    assert (
        2 * (stimuli(np.ones(2)) * stimuli(np.ones(2)))
    ).label == "Fold(Transforms(...), ...)"
    assert stimuli(1).integrate().label == "Integrate(Stimuli())"


def test_convolution():
    shape = (3, 3, 2, 4)
    tr = nengo.Convolution(
        n_filters=4,
        input_shape=(5, 5, 2),
        init=np.arange(np.prod(shape)).reshape(shape),
    )
    inp = np.arange(tr.size_in)

    stim = stimulus(inp)
    out = stim.transform(tr).run(1, 1)

    with nengo.Network() as model:
        stim = nengo.Node(output=inp)
        x = nengo.Node(size_in=tr.size_out)
        nengo.Connection(stim, x, transform=tr, synapse=None)
        p = nengo.Probe(x)

    with nengo.Simulator(model) as sim:
        sim.step()

    assert np.allclose(sim.data[p], out)


def test_rtruediv():
    a = stimulus(0)
    with pytest.raises(TypeError, match="all returned NotImplemented"):
        1 / a


def test_rpow():
    a = stimulus(0)
    with pytest.raises(TypeError, match="all returned NotImplemented"):
        2 ** a


def test_transform():
    twod = stimuli([1, 2]).bundle()
    assert twod.size_out == 2
    assert np.all(twod.run(1, 1) == [[1, 2]])
    assert np.allclose(twod.transform([2, -3]).run(1, 1), [[2, -6]])
    assert np.allclose(twod.transform([[2, -3]]).run(1, 1), [[-4]])
    assert np.allclose(twod.transform([[2, -3], [4, -1]]).run(1, 1), [[-4, 2]])


def test_transform_invalid():
    with pytest.raises(TypeError, match="must be a Fold"):
        Transforms(stimulus(0), [1])

    with pytest.raises(
        ValueError,
        match=r"input operators \(2\) must equal the number of transforms \(3\)",
    ):
        Transforms(stimuli([2, 2]), np.ones(3))

    with pytest.raises(ValueError, match="expected a Fold with only a single axis"):
        Transforms(stimuli(np.eye(2)), np.ones(2))

    with pytest.raises(ValueError, match="input_ops must have all the same size"):
        Transforms(fold([stimulus(1), stimulus([1, 1])]), np.ones(2))


def test_transform_associativity():
    a = stimulus(1)
    b = stimulus(2)
    c = stimulus(3)
    out = 2 * (3 * a + 4 * (b + 5 * c))
    # == 6 * a + 8 * b + 40 * c

    with nengo.Network() as model:
        p = nengo.Probe(out.make())

    with nengo.Simulator(model) as sim:
        sim.step()

    assert sim.data[p].squeeze(axis=-1) == 6 * 1 + 8 * 2 + 40 * 3

    transforms = []
    for conn in model.all_connections:
        if conn.transform is not nengo.params.Default:
            transforms.append(conn.transform.init)
    assert np.all(transforms == [6, 8, 40])


def test_neurons():
    a = stimulus(1).configure(seed=0)
    tr = nengo.dists.Uniform(-1, 1)
    x = a.neurons(transform=tr)
    y = x.run(1)

    with nengo.Network() as model:
        stim = nengo.Node(1)
        ens = nengo.Ensemble(100, 1, seed=0)
        nengo.Connection(stim, ens.neurons, transform=tr, seed=0, synapse=None)
        p = nengo.Probe(ens.neurons)

    with nengo.Simulator(model) as sim:
        sim.run(1)

    assert np.allclose(y, sim.data[p])


@pytest.mark.parametrize("ufunc", [np.sin, np.cos, np.tanh, np.square])
def test_direct_ufuncs(ufunc):
    u = np.linspace(-1, 1, 1000)
    x = stimuli(nengo.processes.PresentInput(u, 1e-3)).configure(
        neuron_type=nengo.Direct()
    )
    y = ufunc(x)
    assert np.allclose(y.run(1).squeeze(axis=-1), ufunc(u))


class _Closure:
    """Encapsulates args and kwargs and creates folds out of NumPy arrays."""

    def __init__(self, *args, folded_args=None, folded_kwargs=None, **kwargs):
        self.args = args
        self.kwargs = kwargs

        if folded_args is None:
            folded_args = tuple(map(self._lift_arg, args))
        self.folded_args = folded_args

        if folded_kwargs is None:
            folded_kwargs = {
                key: self._lift_arg(kwarg) for key, kwarg in kwargs.items()
            }
        self.folded_kwargs = folded_kwargs

    @classmethod
    def _lift_arg(cls, arg):
        # A bit like the inverse of base.py::lower_folds::_lower_arg except direct mode
        # stimuli are automatically created from arrays for testing convenience.
        if is_array(arg):
            return stimuli(arg).configure(neuron_type=nengo.Direct())
        if isinstance(arg, (tuple, list)):
            return type(arg)(map(cls._lift_arg, arg))
        return arg


@pytest.mark.parametrize("array_function", _Fold_array_functions)
def test_array_functions(array_function):
    a = np.arange(3 * 1 * 5).reshape((3, 1, 5))
    b = a * 2 + 1
    c = np.arange(5)
    func1d = sum
    axis = -1
    indices_or_sections = 1
    newshape = (1, 15)

    closures = {
        # Functional programming routines
        np.apply_along_axis: _Closure(func1d, arr=a, axis=axis),
        # Math routines
        np.dot: _Closure(a, c),
        np.mean: _Closure(a),
        np.outer: _Closure(c, c),
        np.prod: _Closure(a, axis=1),
        np.sum: _Closure(a, axis=0),
        # Changing array shape
        np.reshape: _Closure(a, newshape=newshape),
        np.ravel: _Closure(a),
        # Transpose-like operations
        np.moveaxis: _Closure(a, source=0, destination=-1),
        np.rollaxis: _Closure(a, axis=1),
        np.swapaxes: _Closure(a, axis1=-1, axis2=1),
        np.transpose: _Closure(a, axes=(1, 0, 2)),
        # Changing number of dimensions
        np.atleast_1d: _Closure(a),
        np.atleast_2d: _Closure(a),
        np.atleast_3d: _Closure(a),
        np.broadcast_to: _Closure(a, shape=(7,) + a.shape),
        np.broadcast_arrays: _Closure(a, b),
        np.expand_dims: _Closure(a, axis=3),
        np.squeeze: _Closure(a, axis=1),
        # Joining arrays
        np.concatenate: _Closure((a, b)),
        np.stack: _Closure((a, b)),
        np.block: _Closure([a, b]),
        np.vstack: _Closure((a, b)),
        np.hstack: _Closure((a, b)),
        np.dstack: _Closure((a, b)),
        np.column_stack: _Closure((a, b)),
        # Splitting arrays
        np.split: _Closure(a, indices_or_sections=indices_or_sections),
        np.array_split: _Closure(a, indices_or_sections=indices_or_sections),
        np.dsplit: _Closure(a, indices_or_sections=indices_or_sections),
        np.hsplit: _Closure(a, indices_or_sections=indices_or_sections),
        np.vsplit: _Closure(a, indices_or_sections=indices_or_sections),
        # Tiling arrays
        np.tile: _Closure(a, reps=2),
        np.repeat: _Closure(a, repeats=2),
        # Rearranging elements
        np.flip: _Closure(a),
        np.fliplr: _Closure(a),
        np.flipud: _Closure(a),
        np.reshape: _Closure(a, newshape=newshape),
        np.roll: _Closure(a, shift=1),
        np.rot90: _Closure(a),
    }

    if array_function not in closures:
        pytest.fail()

    else:
        closure = closures[array_function]

        op = array_function(*closure.folded_args, **closure.folded_kwargs)
        assert isinstance(op, Operator)
        ideal = array_function(*closure.args, **closure.kwargs)

        out = np.asarray(op.run(1, 1)).squeeze(axis=(-2, -1))
        assert np.allclose(out, ideal)
