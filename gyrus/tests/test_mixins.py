import nengo
import numpy as np
import pytest

from gyrus import register_method, register_ufunc, stimuli, stimulus
from gyrus.base import Operator


def test_not_probeable():
    class BadGenerate(Operator):
        def generate(self, *nodes):
            return None

    op = BadGenerate([])
    with nengo.Network():
        with pytest.raises(ValueError, match="generate is expected to return"):
            op.make()


def test_make_no_context():
    op = stimulus(0)
    with pytest.raises(RuntimeError, match="make method is meant to be invoked within"):
        op.make()


def test_make_cache_basic():
    a = stimulus(0.5)
    b = a + a.filter(0.1) ** 2 - (0.5 * a) ** 3
    with nengo.Network() as model:
        b.make()

    roots = []
    for node in model.all_nodes:
        if node.size_in == 0:
            roots.append(node)

    # a should only be made once
    assert len(roots) == 1 and roots[0].output == 0.5


def test_make_context():
    x = stimuli(0).decode()

    with nengo.Network() as model:
        assert model not in Operator._network_to_cache
        a = x.make()
        assert model in Operator._network_to_cache
        # Test that next two makes are redundant.
        assert len(model.all_ensembles) == 1
        b = x.make()
        c = x.make()
        assert len(model.all_ensembles) == 1

        with nengo.Network() as subnet:
            assert subnet not in Operator._network_to_cache
            d = x.make()
            assert subnet in Operator._network_to_cache

        assert a is b is c
        assert c is not d

    # There should be two ensembles, one in model and one in subnet.
    assert len(model.all_ensembles) == 2
    assert len(model.ensembles) == 1
    assert len(subnet.ensembles) == 1


def test_run_context():
    a = stimulus(0)
    with nengo.Network() as model:
        with pytest.raises(RuntimeError, match="run method is not meant to be"):
            a.run(1)


def test_method_already_registered():
    decorator = register_method("filter")
    with pytest.raises(ValueError, match="already has a method with name"):

        @decorator
        def f(x):
            return x


def test_ufunc_already_registered():
    decorator = register_ufunc(np.negative)
    with pytest.raises(
        ValueError, match="class already implements <ufunc 'negative'>.__call__"
    ):

        @decorator
        def f(x):
            return x


def test_missing_ufunc():
    op = stimulus(6)
    with pytest.raises(TypeError, match="all returned NotImplemented"):
        np.gcd(op, 9)
