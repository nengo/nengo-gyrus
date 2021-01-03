from gyrus.utils import cached_property


class A:
    def __init__(self):
        self.invocations = 0

    @cached_property
    def x(self):
        self.invocations += 1
        return self.invocations


def test_cached_property():
    name = "_x_cached_property"

    a = A()
    assert a.x == 1
    assert a.x == 1
    assert a.invocations == 1
    assert getattr(a, name) == 1

    b = A()
    assert b.invocations == 0
    assert not hasattr(b, name)
    assert b.x == 1
    assert b.x == 1
    assert b.invocations == 1
    assert getattr(b, name) == 1
