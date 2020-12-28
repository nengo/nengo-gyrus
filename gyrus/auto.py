from functools import wraps
from weakref import WeakKeyDictionary

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


def vectorize(
    name, bases=(AutoOperator,), configurable=(), *vectorize_args, **vectorize_kwargs
):
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

    If any parameter names are specified for ``configurable`` then they will be
    automatically picked up through any upstream ``Configure```` operators
    (see ``configure`` for details). Currently this only works for keyword
    (not positional) arguments, and so the decorated function's signature should
    include an asterisk (*) delimiter to force any configurable parameters to be
    keyword-only.
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

            # Resolve the configuration for this list of input operators and then
            # use that to override any unspecified keyword arguments.
            config = Configure.resolve_config(input_ops)
            for key, value in config.items():
                if key in configurable:
                    # Keyword arguments explicitly provided take precedence.
                    impl_kwargs.setdefault(key, value)

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

        # Keep track of configurable parameters so that they can be checked in the
        # Configure constructor.
        Configure.register_configurable(configurable)

        return wrapper

    return decorator


class Configure(Operator):
    """Noop operator that propagates configuration settings downstream."""

    _base_config = {}
    _cache = WeakKeyDictionary()  # non-instances of Configure -> config
    _configurable = set()

    def __init__(self, input_op, reset=False, **config):
        super().__init__((input_op,))
        for key in config:
            if key not in self._configurable:
                raise ValueError(f"parameter '{key}' is not configurable")
        self._config = (
            {**self._base_config, **config}  # config takes precedence (PEP 448)
            if reset
            else self._combine_configs(input_op, downstream_config=config)
        )

    @classmethod
    def register_configurable(cls, names):
        """Globally registers the parameter names as being configurable."""
        cls._configurable |= set(names)

    @classmethod
    def resolve_config(cls, input_ops):
        """Returns the config for an operator with the given input operators."""
        # Combine from left -> right so that the precedence goes from left -> right.
        upstream_config = cls._base_config
        for input_op in input_ops:
            upstream_config = cls._combine_configs(
                upstream_op=input_op, downstream_config=upstream_config
            )
        return upstream_config

    @classmethod
    def _combine_configs(cls, upstream_op, downstream_config):
        """Combines downstream with all upstream configs, following precedence rules."""
        if isinstance(upstream_op, cls):
            upstream_config = upstream_op._config  # dict is copied below
        elif upstream_op in cls._cache:
            upstream_config = cls._cache[upstream_op]
        else:
            # Note that downstream_config is not used here, which makes this
            # context-free. Combining this fact with the immutability of operators
            # makes this cacheable.
            upstream_config = cls._cache[upstream_op] = cls.resolve_config(
                input_ops=upstream_op.input_ops
            )

        # Downstream takes precedence over upstream (see PEP 448).
        return {**upstream_config, **downstream_config}

    @property
    def config(self):
        return self._config.copy()  # safety precaution to keep operators immutable

    @property
    def size_out(self):
        assert len(self.input_ops) == 1
        return self.input_ops[0].size_out

    def generate(self, node):
        return node  # noop


@Operator.register_method("configure")
def configure(input_op, reset=False, **config):
    """Operator that applies configuration settings to all downstream operators.

    Applying the configure operator to any Operator or Fold results in a new Operator
    or Fold that contains the given keyword arguments as configuration. Moreover,
    the configuration is propagated downstream and picked up by applicable operators.

    Currently, configuration is only picked up by operators that have been decorated by
    ``@gyrus.vectorize``', and is restricted to keyword arguments that have been
    explicitly whitelisted as ``configurable`` by the decorator.

    Another limitation is that if any Gyrus operators are created within a generate
    call (for example, considering the ``integrand`` option to the integrator) then
    those operators are by default disconnected from the rest of the graph
    (in the case of the integrator, ``Pre(x)`` creates a disjoint root).

    Configurations obey the following natural rules for precedence:

      1. Keyword arguments explicitly provided to the operator have highest precedence.

      2. An operator's own configuration takes precedence over those of its input
         operators. However, if ``reset=True`` then the input operators are essentially
         ignored, and the configuration is 'reset' to what it would be if configuring a
         root in the graph.

      3. The input operators take precedence in left-to-right order.

    Since operators are immutable and form a directed acyclic graph (DAG), these
    rules are unambiguous, and give a consistent configuration setting for every
    operator in the DAG.
    """
    return asoperator(
        np.vectorize(Configure, otypes=[Configure])(
            input_op=input_op, reset=reset, **config
        )
    )
