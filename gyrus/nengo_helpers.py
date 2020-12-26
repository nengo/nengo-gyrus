import numpy as np

import nengo
from nengo.utils.numpy import is_array_like


def explicit_passthrough(size_in):
    """A node that explicitly passes its signal through (a.k.a. identity function)."""
    # This is needed explicitly in cases where a nengo.Node with output=None has a
    # function applied to it. Nengo currently disallows this case, and the workaround is
    # to pass output=lambda t, x: x.
    def output(t, x):
        return x

    # Not using @wraps(explicit_passthrough) because the signature is different,
    # but copying the name and docstring can help with debugging.
    output.__name__ = explicit_passthrough.__name__
    output.__doc__ = explicit_passthrough.__doc__
    return nengo.Node(output=output, size_in=size_in)


def is_pre(obj):
    """Returns True iff obj can be used as a pre argument to nengo.Connection."""
    return isinstance(obj, (nengo.base.NengoObject, nengo.base.ObjView))


def is_probeable(obj):
    """Returns True iff obj can be used as an argument to nengo.Probe."""
    return isinstance(obj, nengo.base.ObjView) or (
        isinstance(obj, nengo.base.NengoObject) and hasattr(obj, "probeable")
    )


def validate_function_size(function, input_shape=()):
    """Determines the output size of the function when invoked with some input shape.

    Assumes the function accepts ``np.zeros(input_shape)`` as a single argument, has no
    side-effects, and returns an array_like with at most one axis. These are the same
    assumptions made by Nengo functions (e.g., for ``nengo.Connection`` or for
    ``nengo.Node``).
    """
    arg = np.zeros(input_shape)
    value, invoked = nengo.utils.stdlib.checked_call(function, arg)
    if not invoked:
        raise nengo.exceptions.ValidationError(
            f"function '{function}' must accept a single np.array argument with "
            f"shape: {input_shape}",
            attr="function",
        )
    result = np.asarray(value)
    if result.ndim > 1:
        raise nengo.exceptions.ValidationError(
            f"function '{function}' must return an array_like with at most one axis, "
            f"but got shape: {result.shape}",
            attr="function",
        )
    return result.size


def get_transform_size_out(tr, size_in):
    """Infer the size_out of a Nengo transform applied to a pre with given size_in."""

    def _validate_size_in(inferred_size_in):
        if inferred_size_in != size_in:
            raise ValueError(
                f"expected transform ({tr}) to have size_in={size_in}, not "
                f"size_in={inferred_size_in}"
            )

    if isinstance(tr, nengo.transforms.Transform):
        _validate_size_in(tr.size_in)
        return tr.size_out
    elif is_array_like(tr):
        arr = np.asarray(tr)
        if arr.ndim == 0:
            return size_in
        elif arr.ndim == 1:
            _validate_size_in(arr.shape[0])
            return arr.shape[0]
        elif arr.ndim == 2:
            _validate_size_in(arr.shape[1])
            return arr.shape[0]
        else:
            raise ValueError(
                f"expected transform ({tr}) to have ndim of at most 2, not {arr.ndim}"
            )
    raise NotImplementedError(
        f"missing logic to infer size_out of transform {tr} given size_in={size_in}"
    )
