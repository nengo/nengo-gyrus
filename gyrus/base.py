from functools import wraps

import numpy as np
from nengo.utils.numpy import is_array, is_iterable

from gyrus.mixins import NengoSimulatorMixin, RegisterOperatorsMixin
from gyrus.utils import cached_property


class Operator(NengoSimulatorMixin, RegisterOperatorsMixin):
    """Abstract base class for all Gyrus operators.

    A Gyrus operator consists of a tuple of input operators (i.e., dependencies),
    together with a generate method implemented by the subclass. The generate method
    is expected to contain generic code for creating Nengo objects within the
    context of some ``nengo.Network()`` in order to implement the operator.
    Each operator is also expected to be immutable, be free from side-effects, and only
    depend on the arguments to its constructor.

    Intuitively, a Gyrus operator can be understood as a 'function' that consumes
    Nengo nodes and produces Nengo nodes. Essentially like a Nengo subnetwork, but
    with the same base type as all other primitive operations. This lifts the concept
    of a Nengo network to the status of a first-class citizen in Gyrus. We direct the
    user to the Gyrus whitepaper for more high-level discussion and motivation.

    Only in advanced cases should this operator base class be directly subclassed.
    In situations where a user would like to implement a custom operator, it is
    recommended to use the ``@gyrus.vectorize(...)`` decorator to wrap their
    Nengo code (which dynamically creates a new operator type). The
    ``@gyrus.register_method(method_name)`` decorator may also be used to register a
    custom operator as a method on all other operators.

    The input to an operator's generate method is the output from generating each of
    its input operators (one input for each operator, in the same order). In the
    simplest scenario, each operator's generate method consumes a number of
    ``nengo.Node`` objects and produces a single ``nengo.Node`` object. There does not
    need to be any particular relationship between the dimensionality of each input and
    output node. The only assumption is that the generate method does not modify any
    external state (apart from adding Nengo objects to the current network context, and
    possibly even creating other Gyrus operators within the same context).

    A number of methods are provided to make it convenient to create operators
    out of other operators, and to generate and run Nengo models that implement the
    dependency graph of any operator and probe its returned output(s). In addition,
    NumPy ufuncs such as ``np.add`` are handled through the ``RegisterOperatorsMixin``
    and the special NumPy method, ``__array_ufunc__``.
    """

    # Set default recursion depth and width for str(self).
    _str_max_depth = 4
    _str_max_width = None

    def __init__(self, input_ops):
        if not all(isinstance(op, Operator) for op in input_ops):
            raise TypeError(
                f"input_ops ({input_ops}) must be an iterator of Operator types"
            )
        # Cast to tuple as a safety precaution to prevent mutation.
        self._input_ops = tuple(input_ops)

    @property
    def input_ops(self):
        """Tuple of operators that are consumed by this operator."""
        return self._input_ops

    @property
    def ndim(self):
        """Number of dimensions in operator when treated as a NumPy array."""
        # A single operator that simply returns a single Node is like a NumPy
        # scalar, regardless of how many input operators it consumes or the
        # size_out of the operator.
        return 0

    @property
    def shape(self):
        """Shape of operator when treated as a NumPy array."""
        # A NumPy scalar also has a shape of () corresponding to zero dimensions.
        return ()

    @property
    def size_out(self):
        """Output dimensionality of the Nengo object(s) produced by the operator.

        The shape of ``op.size_out`` is expected to match ``op.shape``. Each element
        corresponds to the size_out of the corresponding output Nengo object.
        """
        raise NotImplementedError(
            f"size_out property must be implemented by subclass of type: {type(self)}"
        )

    def generate(self, *nodes):
        """Generates the Nengo objects that implement the operator."""
        raise NotImplementedError(
            f"generate method must be implemented by subclass of type: {type(self)}"
        )

    def __str__(self, max_depth=_str_max_depth, max_width=_str_max_width):
        if max_depth <= 1:
            str_input_ops = "..." if len(self.input_ops) else ""
        else:
            ops = self.input_ops if max_width is None else self.input_ops[:max_width]
            str_input_ops = ", ".join(map(lambda op: op.__str__(max_depth - 1), ops))
            if len(ops) < len(self.input_ops):
                assert max_width is not None
                str_input_ops += ", ..."
        return f"{type(self).__name__}({str_input_ops})"

    def __repr__(self):
        repr_input_ops = ", ".join(map(repr, self.input_ops))
        return f"{type(self).__name__}([{repr_input_ops}])"


class Fold(Operator):
    """A recursive structure of Gyrus operators with NumPy array semantics.

    A fold is recursively defined as a tuple of operators that may optionally include
    other folds.

    This nested array of operators becomes a single operator, with the semantics
    of a NumPy array -- but one in which each element of the array is an operator.
    Operators can be vectorized across folds using the ``@gyrus.vectorize(...)``
    decorator---such that the implementation produces a single operator, while
    the outer function consumes and produces folds---analogous to the
    ``@np.vectorize`` decorator, where a NumPy array is a fold and a NumPy scalar
    is an operator.

    If called with an array of input operators then each level of nesting will be
    converted into a Fold (such that there are no arrays -- only folds containing
    tuple of operators).

    Under the hood, array methods are backed by the machinery of NumPy arrays with
    ``dtype=type(self)``. As well, NumPy array functions, such as ``np.sum``, are
    handled through the special NumPy method, ``__array_function__``. This allows
    folds to be treated as normal NumPy arrays in many situations, but one in which
    each element of the array is a Gyrus operator (each with its own tuple of input
    operator dependencies and separate output dimensionality).
    """

    def __init__(self, input_ops):
        if not self.is_foldable(input_ops):
            raise TypeError(
                f"input_ops ({input_ops}) should be an iterable and not be a Fold"
            )

        folded_input_ops = tuple(
            fold(input_ops=input_op) if self.is_foldable(input_op) else input_op
            for input_op in input_ops
        )
        super().__init__(folded_input_ops)

    def __array_function__(self, func, types, args, kwargs):
        """Handles NumPy array functions by delegating them to registered functions."""
        # https://numpy.org/devdocs/reference/arrays.classes.html
        # Note: Not every function in self._registered_methods get routed to this
        # handler. Only those that are NumPy array functions that operate on arrays,
        # such as reshape, sum, transpose, etc.
        if func in self._registered_methods:
            return lower_folds(func)(*args, **kwargs)
        return NotImplemented

    @classmethod
    def is_foldable(cls, input_ops):
        """Returns True iff input_ops is an acceptable argument to the constructor."""
        # There is no technical reason to disallow a fold to be the input, but
        # this would be a noop and likely not what the user intends to do.
        return is_iterable(input_ops) and not isinstance(input_ops, Fold)

    @cached_property
    def array(self):
        """Returns the structure of input operators as an immutable NumPy array."""
        arr = np.asarray(self.input_ops, dtype=type(self))
        arr.flags.writeable = False  # safety precaution
        return arr

    @property
    def ndim(self):
        return self.array.ndim

    @cached_property
    def shape(self):
        return self.array.shape

    @cached_property
    def size_out(self):
        def _size_out(input_op):
            return input_op.size_out

        out = np.vectorize(_size_out, otypes=[np.int])(self.array)
        assert out.shape == self.shape
        return out

    def generate(self, *nodes):
        return nodes

    def __getitem__(self, indices):
        return asoperator(self.array.__getitem__(indices))

    def __iter__(self):
        return iter(self.input_ops)

    def __len__(self):
        # Note: Defining __len__ and __getitem__ is sufficient for @np.vectorize
        # to automatically vectorize across instances of this type. But __getitem__
        # on its own (as in the Operator base class) is insufficient.
        return len(self.input_ops)


def fold(input_ops):
    """Return a fold from an iterable of operators or folds."""
    return Fold(input_ops)


def asoperator(x):
    """Returns x as a single Operator if possible, otherwise as a Fold."""
    if isinstance(x, Operator):
        op = x
    elif is_array(x) and x.shape == ():
        op = x.item()
        if not isinstance(op, Operator):
            raise TypeError(f"expected array scalar ({op}) to be an Operator")
    else:
        op = fold(x)
    return op


def lower_folds(function):
    """Wraps a function to substitute Folds with NumPy arrays.

    Any arguments (positional and/or keyword) that are Folds are replaced by their
    underlying arrays, before being passed to the decorated function. The value produced
    by the function is then lifted back up to an Operator or Fold. If no arguments can
    be converted, then ``NotImplemented`` is returned.

    This is supposed to be used within ``__array_ufunc__`` or ``__array_function__`` or
    some other dunder method that expects ``NotImplemented`` when the consumed types do
    not have an implementation defined. This allows Python's and/or NumPy's type
    inference machinery to work as intended with overloaded operators.
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        replaced = False
        new_args = []
        for arg in args:
            if isinstance(arg, Fold):
                arg = arg.array
                replaced = True
            new_args.append(arg)
        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, Fold):
                v = v.array
                replaced = True
            new_kwargs[k] = v
        if not replaced:
            return NotImplemented
        return asoperator(function(*new_args, **new_kwargs))

    return wrapper
