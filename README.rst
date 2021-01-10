.. image:: https://i.imgur.com/xef9dhv.png
  :width: 100%
  :target: https://github.com/nengo-labs/nengo-gyrus
  :alt: Recursively generate large-scale Nengo models using NumPy semantics.

Recursively generate large-scale `Nengo <http://nengo.ai/>`_ models using
`NumPy <https://numpy.org/>`_ semantics.

Quick Start
===========

.. code-block:: bash

   pip install git+https://github.com/nengo-labs/nengo-gyrus

An example of computing the square of a two-dimensional vector with Gyrus:

.. code-block:: python

  import gyrus
  import matplotlib.pyplot as plt
  import numpy as np

  u = gyrus.stimuli([np.cos, np.sin])
  x = (u ** 2).filter(0.01)
  y = np.asarray(x.run(np.pi))  # shape: (fold, time, size_out)

  plt.figure()
  plt.plot(y.squeeze(axis=-1).T)
  plt.xlabel("Time-step")
  plt.show()

.. image:: https://i.imgur.com/KyDKeyc.png
  :width: 100%
  :target: https://github.com/nengo-labs/nengo-gyrus
  :alt: Computing the square of a two-dimensional vector with Gyrus.

This code is automagically converted to `Nengo <http://nengo.ai/>`_ and implemented
via two spiking LIF ensembles and a lowpass synapse.

Gyrus supports many common Numpy 'ufuncs', array functions, and numeric Python
operators. Thus, code can be written in a functional style using N-D arrays and then
realized as a Nengo neural network. This enables algorithms to be written in NumPy and
then compiled onto `Nengo's supported backends <https://www.nengo.ai/documentation/>`_
(e.g., GPUs, microcontrollers, neuromorphic hardware, and other neural network
accelerators).

Documentation
=============

Check out and render the Jupyter notebooks located in
`docs/examples <https://github.com/nengo-labs/nengo-gyrus/tree/master/docs/examples>`_.

The `gyrus_overview
<https://github.com/nengo-labs/nengo-gyrus/blob/master/docs/examples/gyrus_overview.ipynb>`_
notebook is currently the best starting point to learn the Gyrus API and see a variety
of examples.

Support
=======

Tested against ``nengo>=3.0.0`` and requires ``numpy>=1.17``.

This project is currently pre-alpha. Pull requests are welcome, as are breaking (i.e.,
reverse-incompatible) changes.

If something doesn't work quite as you thought it should, or if you have ideas for
improvements, please feel free to open up a `GitHub issue
<https://github.com/nengo-labs/nengo-gyrus/issues>`_ or post on the `Nengo Forum
<https://forum.nengo.ai/>`_.