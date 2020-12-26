from functools import wraps

import numpy as np

import nengo
from nengo.utils.numpy import is_integer

from gyrus.base import asoperator, Fold, Operator
from gyrus.utils import cached_property


class AutoOperator(Operator):
    """Operator that is used to automatically wrap implementations via vectorize.

    This type automatically determines the size_out of the operator by calling the
    generate method with dummy zero Nengo nodes. In addition, it includes
    positional arguments and keyword arguments in the call to the underlying
    implementation. Specifically, the ``generate`` method calls
    ``impl(*impl_args, **impl_kwargs)`` but replaces each positional argument
    indexed by ``op_indices`` and each keyword argument indexed by ``op_keys``
    with the input nodes, in the corresponding order.
    """

    def __init__(self, input_ops, impl, impl_args, impl_kwargs, op_indices, op_keys):
        super().__init__(input_ops)
        self._impl = impl
        self._impl_args = impl_args
        self._impl_kwargs = impl_kwargs
        self._op_indices = op_indices
        self._op_keys = op_keys

        # Eager validation is critical in at least one subtle circumstance: for
        # instance, since __getitem__ overloads the Operator, all base operators
        # automatically support iteration. Then, when the split operator (for instance)
        # iterates over an operator it will loop infinitely, as each __getitem__ will
        # create a new operator without raising IndexError. To stop the iteration, we
        # need the IndexError to be raised, which happens if the underlying
        # implementation is called since that triggers it through the call to
        # nengo.Node(...)[indices]. One might think an alternative solution would
        # be the define __len__ for all base operators, but then @np.vectorize will
        # vectorize all the way through the base operator as well which is not wanted.
        self.size_out  # trigger validation

    @cached_property
    def size_out(self):
        # Note: This can be relatively expensive for operators that dynamically
        # create other operators inside of their generate methods (or depend on
        # operators that do). All such dependencies will be generated too, as will
        # any that they might generate in their generate methods. The result of this
        # property is therefore cached, but that is still less than ideal
        # as each size_out calculation can recreate some portion of the graph.
        with nengo.Network(add_to_container=False):
            input_nodes = []
            for input_op in self.input_ops:
                input_op_size_out = input_op.size_out
                if not is_integer(input_op_size_out):
                    raise ValueError(
                        f"default size_out property only works if all input_ops "
                        f"produce an integer size_out, but got: {input_op_size_out}"
                    )
                input_nodes.append(nengo.Node(np.zeros(input_op_size_out)))
            output = self.generate(*input_nodes)
            if not hasattr(output, "size_out"):
                # Expected to be a Nengo node or a sliced object view of a Nengo node.
                raise ValueError(
                    f"default size_out property only works if the output from generate "
                    f"defines size_out, but got: {output}"
                )
        return output.size_out

    def generate(self, *nodes):
        # Invokes the decorated function with the same signature but fills in each input
        # operator with the corresponding node (one node for each input_op, in the same
        # order). The rest of the arguments are left as they have been determined by
        # np.vectorize. Copies are created to keep the operator instance frozen.
        j = 0
        impl_args = list(self._impl_args)
        impl_kwargs = dict(self._impl_kwargs)
        for op_index in self._op_indices:
            impl_args[op_index] = nodes[j]
            j += 1
        for op_key in self._op_keys:
            impl_kwargs[op_key] = nodes[j]
            j += 1
        if j != len(nodes):
            raise RuntimeError(
                f"mismatch between number of op_indices plus op_keys ({j}) versus the"
                f"number of nodes ({len(nodes)})"
            )
        return self._impl(*impl_args, **impl_kwargs)


def vectorize(name, bases=(AutoOperator,), *vectorize_args, **vectorize_kwargs):
    """Dynamically creates a vectorized operator type with the given implementation.

    More specifically, using this to decorate some Nengo code results in a function
    that consumes some number of Operators or Folds and produces a single Operator or
    Fold with the same shape. The wrapped implementation is applied element-wise
    within the Operator or Fold.

    Under the hood, this is a combination of ``@np.vectorize`` on the outside and
    ``AutoOperator`` on the inside. That is, the decorator iterates across its inputs
    to create a NumPy array of operators. Each such element is in fact an instance of a
    newly created subclass of ``AutoOperator`` with the given name and the given
    implementation used for its ``generate`` method. By default, the size of each
    operator is determined automatically by calling the implementation with dummy zero
    Nengo nodes.
    """

    def decorator(impl):
        # Dynamically create a type for the target operator.
        # This only happens once in total per decorated function.
        dct = {
            "__module__": impl.__module__,
            "__doc__": impl.__doc__,
        }
        cls = type(name, bases, dct)

        # Define the function that will be vectorized using np.vectorize.
        def instantiate(*impl_args, **impl_kwargs):
            # Determine which args and kwargs are Gyrus operators.
            op_indices = [
                i for i, value in enumerate(impl_args) if isinstance(value, Operator)
            ]
            op_keys = [
                key for key, value in impl_kwargs.items() if isinstance(value, Operator)
            ]

            # Remember where the operators appear in the function signature
            # for use in instance.generate.
            input_ops = []
            for op_index in op_indices:
                input_ops.append(impl_args[op_index])
            for op_key in op_keys:
                input_ops.append(impl_kwargs[op_key])
            if any(isinstance(op, Fold) for op in input_ops):
                raise TypeError(
                    f"{name} is being instantiated with a Fold despite being "
                    f"vectorized -- perhaps {impl} was called with a jagged array?"
                )

            # The operators (input_ops) are the inputs to the target operator.
            # Each of those inputs will in turn be mapped onto some node
            # that becomes a positional argument to generate (in the same order).
            # This will become the generate implementation for the operator.
            # See AutoOperator.generate for more details.
            return cls(
                input_ops=input_ops,
                impl=impl,
                impl_args=impl_args,
                impl_kwargs=impl_kwargs,
                op_indices=op_indices,
                op_keys=op_keys,
            )

        # This creates a function which automatically vectorizes all inputs to
        # the decorated function using the normal numpy rules for broadcasting,
        # such that the implementation only needs to handle individual operators
        # (rather than folds). The return type is a numpy array consisting of
        # objects of the dynamically generated type (cls). Note that even though
        # the base object is iterable (because it supports __getitem__) it won't
        # be vectorized by np.vectorize as it also looks for __len__ to determine
        # whether to vectorize an element, which is currently supported by Fold
        # but not Operator.
        vectorized = np.vectorize(
            instantiate, otypes=[cls], *vectorize_args, **vectorize_kwargs
        )

        # The actual implementation (i.e., the returned wrapper) does the vectorization
        # and then converts the array output into a Fold -- unless it is just a single
        # operator, in which case just that single operator is returned.
        @wraps(impl)
        def wrapper(*args, **kwargs):
            array = vectorized(*args, **kwargs)
            op = asoperator(array)
            return op

        return wrapper

    return decorator
