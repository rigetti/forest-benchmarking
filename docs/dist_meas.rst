.. module:: forest.benchmarking.distance_measures

Distance Measures
=================

It is often the case that we wish to measure how close an experimentally prepared quantum state
is to the ideal, or how close an ideal quantum gate is to its experimental implementation. The
forest.benchmarking module ``distance_measures.py``
allows you to explore some quantitative measures of comparing quantum states and processes.

Distance measures between states or processes can be subtle. We recommend thinking about the *operational interpretation* of each measure before using the measure.

.. toctree::

    examples/distance_measures

Distance measures between quantum states
----------------------------------------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    fidelity
    infidelity
    trace_distance
    bures_distance
    bures_angle
    quantum_chernoff_bound
    hilbert_schmidt_ip
    smith_fidelity
    total_variation_distance
    purity
    impurity

Distance measures between quantum processes
-------------------------------------------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    entanglement_fidelity
    process_fidelity
    process_infidelity
    diamond_norm_distance
    watrous_bounds
