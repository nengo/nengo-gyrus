import nengo
import numpy as np


class Parabola(nengo.neurons.NeuronType):
    """A non-spiking neuron model whose response curve is ``(a*x + b)**2``.

    This is the ideal ``neuron_type`` for implementing multiplication via the
    ``np.multiply`` or ``gyrus.multiply`` functions, or for implementing
    circular convolution via the ``gyrus.convolve`` function.

    Most neuron models obey the following two constraints to relate the ``gain``
    and ``bias`` to the ``max_rates`` and ``intercepts``:
      1. ``f(gain + bias) == max_rates``
      2. ``f(gain * intercepts + bias) == 0``
    where ``f`` is the response curve (in this case, ``f(J) = J ** 2``).

    This neuron model strengthens the first constraint as follows:
      ``max((gain * (+1) + bias) ** 2, (gain * (-1) + bias) ** 2) == max_rates``

    The element-wise maximum is required because the tuning curve is a parabola that
    is symmetric about its intercept. For negative intercepts, the maximum firing
    rate is achieved at (+1) as usual. For positive intercepts, the maximum firing
    rate is achieved at (-1) instead of (+1). Thus, to prevent f(x) from growing
    too large at the furthest end, this constraint limits the maximum rate to the
    larger of the two extremes.
    """

    negative = False  # nengo>=3.1.0

    def gain_bias(self, max_rates, intercepts):
        """Analytically determine gain, bias."""
        max_rates = np.array(max_rates, dtype=float, copy=False, ndmin=1)
        intercepts = np.array(intercepts, dtype=float, copy=False, ndmin=1)

        # Eq. 1: (gain * (+1) + bias)**2  = max_rates
        #        (gain * (-1) + bias)**2  = max_rates
        #     => bias +/- gain = sqrt(max_rates)
        # Eq. 2: (gain * intercepts + bias)**2 = 0
        #     => gain = -bias / intercepts
        #        bias = -gain * intercepts
        gain = np.sqrt(max_rates) / (1 + np.abs(intercepts))
        bias = -gain * intercepts
        return gain, bias

    def max_rates_intercepts(self, gain, bias):
        """Compute the inverse of gain_bias."""
        max_rates = np.maximum((gain + bias) ** 2, (bias - gain) ** 2)
        intercepts = -bias / gain
        return max_rates, intercepts

    def step_math(self, *args, **kwargs):  # pragma: no cover
        """Implement the square nonlinearity."""
        return self.step(*args, **kwargs)  # nengo<3.1.0

    def step(self, dt, J, output):
        """Implement the square nonlinearity."""
        output[...] = J ** 2
