import nengo
import numpy as np
from nengo.utils.filter_design import cont2discrete
from nengo.utils.numpy import is_array, is_array_like, is_iterable, is_number

from gyrus.auto import vectorize
from gyrus.base import Fold, Operator, asoperator, fold, lower_folds
from gyrus.nengo_helpers import (
    explicit_passthrough,
    get_params,
    get_transform_size_out,
    is_pre,
    validate_function_size,
)
from gyrus.utils import cached_property


def _node_or_identity(output):
    """Returns output if it is a valid pre object, otherwise Node(output)."""
    if is_pre(output):
        return output
    else:
        return nengo.Node(output=output)


@vectorize("Pre", excluded=[0, "output"])
def pre(output):
    """Operator that supplies input data given a Nengo object or a Node output.

    This is equivalent to ``stimulus`` but won't vectorize its argument. Only a single
    pre object is produced that supplies the given input, as is. This is primarily
    useful when ``output`` is like an array, and we want to keep it as such as opposed
    to creating a separate operator for each element of ``output``.
    """
    # Nodes are iterable (along the output dimension) due to them implementing
    # __getitem__ and __len__, so they automatically get vectorized by @np.vectorize.
    # As a result, stimulus(stim) will return a Fold with each element in the fold
    # corresponding to an output dimension from stim. This may be okay in some
    # situations. But the workaround for the more typical situation is to provide
    # stim[:] which is not iterable but is awkward and cannot be sliced again
    # (because Nengo ObjView cannot be sliced). So instead we made this version which
    # explicitly excludes the parameter from np.vectorize.
    return _node_or_identity(output)


@vectorize("Stimulus")
def stimulus(output):
    """Operator that supplies input data given Nengo objects or Node outputs.

    This is subtly different from ``pre`` in that the input can be vectorized to produce
    a Fold. For example, if an array is provided then it will be vectorized into a
    number of one-dimensional nodes. Use ``pre`` if this is not desired.
    """
    return _node_or_identity(output)


def broadcast_scalar(scalar, size_out):
    """Operator that creates a scalar Node and transforms it to match some shape.

    The parameter ``size_out`` is expected to be in the format of ``op.size_out``
    where ``op`` is an Operator (such as a Fold).
    """
    scalar = np.asarray(scalar)
    if not scalar.ndim == 0:
        raise TypeError(f"expected scalar, but got array with shape: {scalar.shape}")

    input_op = pre(scalar)

    def _project(_size_out):
        return input_op.transform(np.ones((_size_out, 1)))

    # This is vectorized across the elements of op.size_out. If op is just a basic
    # Operator then this trivially reduces to a single transform. If op is a Fold
    # then the elements of size_out correspond to the size_out of each of its operators
    # and a separate transform (from the same pre object) is created for each element.
    # For every repeated value of size_out, the same transform will be recreated.
    # An optimizer could reuse the transform if it makes sense to (it might not always
    # make sense to do so, for example, if a scalar is being communicated to multiple
    # different processing cores and it saves communication to do the transform after
    # communicating the scalar as opposed to doing it before and communicating a
    # high-dimensional vector to each core.
    return asoperator(np.vectorize(_project, otypes=[Transforms])(_size_out=size_out))


@Operator.register_method("__getitem__")
@vectorize("Slice")
def slice(node, indices):
    """Operator that slices each output using Nengo's object slicing."""
    # Note that fold has its own __getitem__ and so this overload applies to non-fold
    # operators (or can be called directly on a fold of course).
    return node[indices]


@Operator.register_method("filter")
@vectorize("Filter")
def filter(node, synapse):
    """Operator that applies a synaptic filter to each output."""
    out = nengo.Node(size_in=node.size_out)
    nengo.Connection(node, out, synapse=synapse)
    return out


@Operator.register_method("apply")
@vectorize("Apply")
def apply(node, function):
    """Operator that applies a function to each output ideally using a Node."""
    size_out = validate_function_size(function, input_shape=node.size_out)
    # An explicit passthrough is needed here because we are applying a function
    # to its output. Note we could also apply the function directly inside the
    # node but that requires adding a time parameter which would require
    # wrapping the user's function.
    copy = explicit_passthrough(size_in=node.size_out)
    nengo.Connection(node, copy, synapse=None)
    out = nengo.Node(size_in=size_out)
    nengo.Connection(copy, out, function=function, synapse=None)
    return out


@Operator.register_method("decode")
@vectorize(
    "Decode", configurable=get_params(nengo.Ensemble) - {"dimensions"} | {"seed"}
)
def decode(node, function=lambda x: x, *, n_neurons=100, label="Decode", **ens_kwargs):
    """Operator that approximates a function of each output using an Ensemble."""
    x = nengo.Ensemble(
        n_neurons=n_neurons, dimensions=node.size_out, label=label, **ens_kwargs
    )
    size_out = validate_function_size(function, input_shape=node.size_out)
    out = nengo.Node(size_in=size_out)
    nengo.Connection(node, x, synapse=None)
    nengo.Connection(x, out, function=function, synapse=None)
    return out


@Operator.register_method("neurons")
@vectorize(
    "Neurons",
    configurable=get_params(nengo.Ensemble) - {"dimensions"} | {"seed", "transform"},
)
def neurons(
    node,
    *,
    n_neurons=100,
    transform=nengo.params.Default,
    label="Neurons",
    **kwargs,
):
    """Operator that linearly projects to a layer of neurons."""
    # Note: A number of kwargs will essentially be ignored by the Ensemble, such as
    # encoders, normalize_encoders, eval_points, n_eval_points, and radius.
    x = nengo.Ensemble(n_neurons=n_neurons, dimensions=1, label=label, **kwargs)
    # Use the same seed so that if a distribution is provided for the transform then
    # it will seed that distribution.
    seed = kwargs.get("seed", None)
    nengo.Connection(node, x.neurons, transform=transform, seed=seed, synapse=None)
    return x.neurons


@Operator.register_method("multiply")
@vectorize(
    "Multiply", configurable={"n_neurons", "neuron_type", "input_magnitude", "seed"}
)
def multiply(
    node_a,
    node_b,
    *,
    n_neurons=100,
    neuron_type=nengo.params.Default,
    input_magnitude=1.0,
    label="Multiply",
    **net_kwargs,
):
    """Operator that approximates an element-wise product using a Product network.

    ``n_neurons`` is the number of neurons per Ensemble, for which there are two per
    multiplication.
    """
    if node_a.size_out != node_b.size_out:
        raise ValueError(
            f"multiply operator size_out of operand a ({node_a.size_out}) does not "
            f"match size_out of operand b ({node_b.size_out})"
        )
    product = nengo.networks.Product(
        n_neurons=2 * n_neurons,
        dimensions=node_a.size_out,
        input_magnitude=input_magnitude,
        label=label,
        **net_kwargs,
    )
    if neuron_type is not nengo.params.Default:
        # Product network doesn't support ensemble kwargs, and setting the config after
        # the network is created doesn't change it.
        for ensemble in product.all_ensembles:
            ensemble.neuron_type = neuron_type
    assert product.output.label == "output"
    product.output.label = None  # gets set automatically by NengoSimulatorMixin.make

    nengo.Connection(node_a, product.input_a, synapse=None)
    nengo.Connection(node_b, product.input_b, synapse=None)
    return product.output


@Operator.register_method("convolve")
@vectorize(
    "Convolve", configurable={"n_neurons", "neuron_type", "input_magnitude", "seed"}
)
def convolve(
    node_a,
    node_b,
    *,
    n_neurons=100,
    neuron_type=nengo.params.Default,
    invert_a=False,
    invert_b=False,
    input_magnitude=1.0,
    label="Convolve",
    **net_kwargs,
):
    """Operator that approximates the circular convolution using Product networks.

    ``n_neurons`` is the number of neurons per Ensemble, for which there are two per
    dimension.
    """
    if node_a.size_out != node_b.size_out:
        raise ValueError(
            f"convolve operator size_out of operand a ({node_a.size_out}) does not "
            f"match size_out of operand b ({node_b.size_out})"
        )
    convolution = nengo.networks.CircularConvolution(
        n_neurons=2 * n_neurons,
        dimensions=node_a.size_out,
        invert_a=invert_a,
        invert_b=invert_b,
        input_magnitude=input_magnitude,
        label=label,
        **net_kwargs,
    )
    if neuron_type is not nengo.params.Default:
        # CircularConvolution network doesn't support ensemble kwargs, and setting the
        # config after the network is created doesn't change it.
        for ensemble in convolution.all_ensembles:
            ensemble.neuron_type = neuron_type
    assert convolution.output.label == "output"
    convolution.output.label = None  # set automatically by NengoSimulatorMixin.make

    nengo.Connection(node_a, convolution.input_a, synapse=None)
    nengo.Connection(node_b, convolution.input_b, synapse=None)
    return convolution.output


@Operator.register_method("integrate")
@vectorize("Integrate")
def integrate(u, integrand=None):
    r"""Operator that solves \dot{x} = u + integrand(x) using Euler's method."""
    x = nengo.Node(size_in=u.size_out)
    integrator = nengo.synapses.LinearFilter([1], [1, 0])

    nengo.Connection(u, x, synapse=integrator)
    if integrand is not None:
        # Note: The operator that integrand(...) produces will be generated into the
        # current Nengo context, including all of its dependencies. This even includes
        # dependencies that may have already been built into a different subnetwork!
        # Only dependencies that have already been generated into the current Nengo
        # context will not be regenerated.
        op_dot_x = integrand(pre(x))
        dot_x = op_dot_x.make()
        if not isinstance(dot_x, nengo.Node):
            raise TypeError(
                f"expected integrand to generate a single Node, but got: {dot_x}"
            )
        elif dot_x.size_out != u.size_out:
            raise TypeError(
                f"integrand returned Node with size_out={dot_x.size_out} but expected "
                f"size_out={u.size_out}"
            )
        nengo.Connection(dot_x, x, synapse=integrator)

        # If op_dot_x.make() recurses back to pre(x), then it can set the label
        # to Pre() but we'd like it to be set according to str(op) where op is the
        # current Integrate operator. Setting the label back to None and returning
        # x will achieve this.
        if x.label is not None:
            x.label = None
    return x


@Operator.register_method("lti")
def lti(u, system, state=lambda x: x, dt=0.001, method="zoh"):
    r"""Operator that solves \dot{x} = A.state(x) + B.u. where A, B = system.

    The state parameter can be any callable function that consumes a Pre operator
    and produces a Gyrus operator that consumes said operator as an input. For instance,
    nonlinear dynamical systems may be implemented by specifying a nonlinear function
    for the ``state``.
    """
    if not is_iterable(system) or len(system) != 2:
        raise NotImplementedError(
            f"lti currently only supports systems as two-tuples: (A, B); not {system}"
        )

    # Reshape and validate A, B matrices.
    A, B = system
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A ({A}) must be a square matrix")
    size_out = A.shape[0]
    B = np.asarray(B)
    if B.ndim == 1:
        B = B[:, None]
    if B.ndim != 2:
        raise ValueError(f"B ({B}) must be 1D or 2D, but is {B.ndim}")
    if B.shape[0] != size_out:
        raise ValueError(
            f"B ({B}) must be an array of length {size_out}, not {B.shape[0]}"
        )
    C = np.zeros_like(B).T
    D = 0

    # Discretize the dynamical system, \dot{x} = Ax + Bu, to
    # x[t + dt] = Ax[t] + Bu[t], using some method (ZOH recommended for stability).
    Abar, Bbar, _, _, _ = cont2discrete((A, B, C, D), dt=dt, method=method)

    # Apply Voelker (2019) equation 5.30 with H(z) = dt / (z - 1).
    # This compensates for the discretized integrator such that the resulting
    # system is the one that was requested. In this particular case (with the
    # synapse being the discretized integrator) this reduces to the inverse
    # of Euler's method.
    Amap = (Abar - np.eye(len(Abar))) / dt
    Bmap = Bbar / dt

    # Finally express the Amap, Bmap system using vectorized Gyrus operators.
    return u.transform(Bmap).integrate(integrand=lambda x: state(x).transform(Amap))


@Operator.register_ufunc(np.add)
def __ufunc_add(a, b):
    """Operator that implements element-wise addition with Gyrus operator(s).

    If both operands are Gyrus operators, then they are added together by Nengo
    transform(s) that add every dimension, element-wise.

    If one of the operands is like an array, then it is expected to be either
    0D (scalar) or 1D (list). In the 0D case, the scalar is provided by a Nengo node,
    and broadcasted through Nengo transforms according to the shape of the other
    operand. In the 1D case, the list is expected to have the same number of elements
    as the size_out of each element in the other operand, such that the same vector can
    be added to every vector that is produced by the other operand.
    """
    if isinstance(a, Operator) and isinstance(b, Operator):
        return reduce_transform([a, b], trs=[1, 1], axis=0)
    elif isinstance(b, Operator):
        a, b = b, a
    # At least one of the two operands has to be an Operator.
    assert isinstance(a, Operator)
    assert not isinstance(b, Operator)  # due to first if statement
    if is_array_like(b):
        # This can be generalized to handle a wider variety of cases,
        # But, similar to __ufunc_multiply we are keeping the behaviour
        # as unambiguous as possible for now.
        b = np.asarray(b)
        if np.all(b == 0):
            return a
        elif b.ndim == 0:
            # Creates a single scalar Node and then transforms it
            # according to the size_out of each element in a.
            b = broadcast_scalar(b, size_out=a.size_out)
        elif b.ndim == 1:
            # Repeats the same Node for every element in a.
            if np.any(a.size_out != len(b)):
                raise TypeError(
                    f"add size mismatch for a (operator) + b (array): "
                    f"a.size_out ({a.size_out}) must match len(b) ({len(b)})"
                )
            b = asoperator(np.broadcast_to(pre(b), shape=a.shape))
        else:
            return NotImplemented
    elif not isinstance(b, Operator):
        return NotImplemented
    return np.add(a, b)  # recurse to first if statement


@Operator.register_ufunc(np.multiply)
def __ufunc_multiply(a, b):
    """Operator that implements element-wise multiplication with Gyrus operator(s).

    If both operands are Gyrus operators, then they are multiplied together by using
    ``gyrus.multiply``, which vectorizes an element-wise product network across both
    operands.

    If one of the operands is like an array, then it is expected to be either
    0D (scalar) or 1D (list). In either case, the operand becomes a Nengo transform
    that is applied to each Gyrus operator to scale its outputs element-wise.
    """
    # gyrus.multiply (used here and defined elsewhere) is a bit different from
    # the ufunc defined here, np.multiply. The former only supports multiplying two
    # operators. The latter delegates to the former (in the same way as __mul__), but
    # also delegates to transform to handle a wider variety of types. In particular, if
    # one of the two operands is not an operator, then it will be used as a transform on
    # the other operand. To make the semantics of this unambiguous, in a similar manner
    # to __ufunc_add, only 0D or 1D transforms are currently supported, such that the
    # transform is doing an element-wise multiplication on each element.
    if isinstance(a, Operator) and isinstance(b, Operator):
        return multiply(a, b)
    elif isinstance(b, Operator):
        a, b = b, a
    # At least one of the two operands has to be an Operator.
    assert isinstance(a, Operator)
    assert not isinstance(b, Operator)  # due to first if statement
    if is_array_like(b):
        b = np.asarray(b)
        # Scalars (b.ndim == 0) are fine, as is, as they naturally work with
        # nengo.Connection. 1D arrays also work, as is, so long as their length is equal
        # to the number of input dimensions in Nengo, which will also be the number of
        # output dimensions.
        if np.all(b == 1):
            return a
        elif b.ndim == 1 and np.any(a.size_out != len(b)):
            raise TypeError(
                f"multiply size mismatch for a (operator) * b (array): "
                f"a.size_out ({a.size_out}) must match len(b) ({len(b)})"
            )
        elif b.ndim >= 2:
            return NotImplemented
    return transform(a, tr=b)


@Operator.register_ufunc(np.subtract)
def __ufunc_subtract(a, b):
    """Operator that implements element-wise subtraction with Gyrus operator(s).

    See __ufunc_add for details.
    """
    # If b is an operator then this chains two transforms that get combined
    # through Transforms._combine_transforms.
    return np.add(a, np.negative(b))


@Operator.register_ufunc(np.negative)
def __ufunc_negative(x):
    """Operator that implements element-wise negation with Gyrus operator(s).

    See __ufunc_multiply for details.
    """
    assert isinstance(x, Operator)
    # Also equivalent to x.transform(-1).
    return np.multiply(x, -1)


@Operator.register_ufunc(np.matmul)
@lower_folds
def __ufunc_matmul(*args, **kwargs):
    """Operator that implements matrix multiplication with Gyrus operator(s)."""
    # This looks like infinite recursion but the lower_folds decorator makes progress
    # by replacing arguments with NumPy arrays, which causes the np.matmul to be
    # handled elsewhere. On the other hand, if none of the args get replaced then that
    # means this was called on an Operator and not a Fold. In this case it comes back
    # as NotImplemented, which is consistent with how np.matmul is not implemented for
    # scalars. If we weren't to return NotImplemented this then this method would
    # recurse infinitely as it would be making no progress through the chain of
    # np.matmul -> __ufunc_matmul.
    return np.matmul(*args, **kwargs)


@Operator.register_ufunc(np.divide)
def __ufunc_divide(a, b):
    """Operator that implements element-wise division with Gyrus operator(s).

    Currently only supports ``a / b`` where ``b`` is not an Operator. That is, the Gyrus
    operator must be the dividend (``a``).
    """
    if not isinstance(b, Operator):
        assert isinstance(a, Operator)
        return np.multiply(a, 1 / b)
    return NotImplemented


@Operator.register_ufunc(np.power)
def __ufunc_power(base, exponent):
    """Operator that implements raising to some power with Gyrus operator(s).

    Currently only supports ``base ** exponent`` where ``exponent`` is not an Operator.
    That is, the Gyrus operator is the base. The same exponent is applied to each
    element of the base.
    """
    if is_array_like(exponent):
        assert isinstance(base, Operator)
        return base.decode(lambda x: x ** exponent)
    return NotImplemented


@Operator.register_ufunc(np.sin)
def __ufunc_sin(op):
    """Operator that is an alias for op.decode(np.sin)."""
    return op.decode(np.sin)


@Operator.register_ufunc(np.cos)
def __ufunc_cos(op):
    """Operator that is an alias for op.decode(np.cos)."""
    return op.decode(np.cos)


@Operator.register_ufunc(np.tanh)
def __ufunc_tanh(op):
    """Operator that is an alias for op.decode(np.tanh)."""
    return op.decode(np.tanh)


@Operator.register_ufunc(np.square)
def __ufunc_square(op):
    """Operator that is an alias for op.decode(np.square)."""
    return op.decode(np.square)


class Bundle1D(Operator):
    """Operator that joins all outputs in a list of operators into a single output."""

    def __init__(self, input_ops):
        super().__init__(input_ops)
        if any(isinstance(op, Fold) for op in self.input_ops):
            raise ValueError(
                f"expected all input ops to be an Operator but not a Fold, but got: "
                f"{self.input_ops}"
            )
        if not len(self.input_ops):
            raise ValueError("cannot bundle zero nodes")

    @cached_property
    def size_out(self):
        return sum(op.size_out for op in self.input_ops)

    def generate(self, *nodes):
        size_out = sum(node.size_out for node in nodes)
        y = nengo.Node(size_in=size_out)
        i = 0
        for op, node in zip(self.input_ops, nodes):
            k = op.size_out
            nengo.Connection(node, y[i : i + k], synapse=None)
            i += k
        assert i == size_out
        return y


@Fold.register_method("bundle")
def bundle(input_ops, axis=-1):
    """Operator that joins all of the outputs along a given axis into a single output.

    This reduces the dimensionality of the Fold by one by applying ``Bundle1D`` along
    the chosen axis. When each element has a ``size_out`` of 1, this is the inverse of
    unbundle (when called with the same axis).
    """
    # This together with Bundle1D is a reasonably compact example of how to define a
    # custom operator without using @gyrus.vectorize. Instead of np.apply_along_axis
    # many custom operators might apply np.vectorize.
    return asoperator(np.apply_along_axis(func1d=Bundle1D, axis=axis, arr=input_ops))


@Operator.register_method("unbundle")
def unbundle(input_ops, axis=-1):
    """Operator that splits each output into a Fold of one-dimensional outputs.

    The expands the dimensionality by creating a Fold along the given axis, containing
    the outputs of each element. This is the inverse of bundle (when called with the
    same axis).
    """

    def _unbundle(input_op):
        assert input_op.ndim == 0
        # This tuple unpacking works because the Operator is iterable via __getitem__.
        # This produces a Slice operator.
        return fold([*input_op])

    return np.moveaxis(
        fold(np.vectorize(_unbundle, otypes=[Fold])(input_ops)),
        source=-1,
        destination=axis,
    )


class Transforms(Operator):
    """Operator that sums across each transform applied to each input operator."""

    def __init__(self, input_ops, trs):
        input_ops = asoperator(input_ops)
        if not isinstance(input_ops, Fold):
            raise TypeError(f"input_ops ({input_ops}) must be a Fold")
        trs = tuple(trs)

        if len(input_ops) != len(trs):
            raise ValueError(
                f"number of input operators ({len(input_ops)}) must equal the number "
                f"of transforms ({len(trs)})"
            )

        if input_ops.ndim != 1:
            raise ValueError(
                f"expected a Fold with only a single axis, but input_ops has shape: "
                f"{input_ops.shape}"
            )

        input_ops, trs = self._combine_transforms(input_ops, trs)

        # Infer input and output dimensionality of each transform.
        size_in = input_ops.size_out
        assert size_in.ndim == 1  # since input_ops.ndim == 1
        size_out = np.asarray(
            [
                get_transform_size_out(tr, size_in)
                for tr, size_in in zip(trs, input_ops.size_out)
            ]
        )
        if len(size_out) == 0 or np.any(size_out[0] != size_out[1:]):
            raise ValueError(f"input_ops must have all the same size, got: {size_out}")

        self._size_out = size_out[0]
        self._trs = trs
        super().__init__(input_ops)

    @classmethod
    def _combine_transforms(cls, input_ops, trs):
        """Converts (a*x + b*(y + z + ...)) into (a*x + b*y + b*z + ...)."""
        new_input_ops = []
        new_trs = []

        def _is_number_or_scalar(x):
            return is_number(x) or (is_array(x) and x.ndim == 0)

        for input_op, tr in zip(input_ops, trs):
            if (
                isinstance(input_op, Transforms)
                and _is_number_or_scalar(tr)
                and all(
                    _is_number_or_scalar(input_op_tr) for input_op_tr in input_op.trs
                )
            ):
                # For simplicity we skip this optimization if the weights are not
                # scalars. We could include this if we had a way to combine two Nengo
                # transforms into one. However, it is not always optimal to do so, for
                # instance, if the two reduce_transform are low rank, as in
                # p.outer(encoders, decoders). Also see np.multi_dot and MCOP:
                # https://en.wikipedia.org/wiki/Matrix_chain_multiplication
                # We view this more as the job of a general Nengo network optimizer.
                new_input_ops.extend(input_op.input_ops)
                new_trs.extend(tr * np.asarray(input_op.trs))
            else:
                new_input_ops.append(input_op)
                new_trs.append(tr)
        return fold(new_input_ops), tuple(new_trs)

    @property
    def trs(self):
        return self._trs

    @property
    def size_out(self):
        return self._size_out

    def generate(self, *nodes):
        y = nengo.Node(size_in=self.size_out)
        for node, tr in zip(nodes, self.trs):
            nengo.Connection(node, y, transform=tr, synapse=None)
        return y


@Operator.register_method("transform")
def transform(input_op, tr):
    """Operator that applies a single transform to every output."""
    return asoperator(
        np.vectorize(
            pyfunc=lambda elem: Transforms([elem], trs=[tr]), otypes=[Transforms]
        )(input_op)
    )


@Fold.register_method("reduce_transform")
def reduce_transform(input_ops, trs, axis=-1):
    """Operator that sums together transforms applied to each output along an axis.

    This reduces the dimensionality of the Fold by one by applying ``Transforms``
    along the chosen axis.
    """
    return asoperator(
        np.apply_along_axis(
            func1d=lambda arr: Transforms(arr, trs=trs), axis=axis, arr=input_ops
        )
    )


# This is a non-exhaustive whitelist of NumPy array functions that can be handled
# automatically through Fold.__array_function__; these array functions simply defer to
# the underlying NumPy version to do all the heavy lifting. That is, __array_function__
# only handles Fold conversion to and from the underlying implementation. The
# __array_ufunc__ operations on the other hand delegate to Gyrus operators that
# generate the actual Nengo objects.
_Fold_array_functions = (
    np.atleast_1d,
    np.atleast_2d,
    np.atleast_3d,
    np.apply_along_axis,
    np.array_split,
    np.block,
    np.broadcast_arrays,
    np.broadcast_to,
    np.column_stack,
    np.concatenate,
    np.dot,
    np.dsplit,
    np.dstack,
    np.expand_dims,
    np.flip,
    np.fliplr,
    np.flipud,
    np.hsplit,
    np.hstack,
    np.mean,
    np.moveaxis,
    np.outer,
    np.prod,
    np.ravel,
    np.repeat,
    np.reshape,
    np.roll,
    np.rollaxis,
    np.rot90,
    np.split,
    np.squeeze,
    np.stack,
    np.sum,
    np.swapaxes,
    np.tile,
    np.transpose,
    np.vsplit,
    np.vstack,
)

for array_function in _Fold_array_functions:
    Fold.register_method(array_function.__name__)(array_function)
