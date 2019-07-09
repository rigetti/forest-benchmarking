.. module:: forest.benchmarking.randomized_benchmarking

Randomized Benchmarking
=======================

Randomized benchmarking involves running long sequences of random Clifford group gates which
compose to the identity to observe how performance degrades with increasing circuit depth.

.. todo:: Talk some more about stuff.


Gates and Sequences
-------------------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    oneq_rb_gateset
    twoq_rb_gateset
    get_rb_gateset
    generate_rb_sequence
    merge_sequences
    generate_rb_experiment_sequences
    group_sequences_into_parallel_experiments


Standard and Interleaved RB
---------------------------

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    generate_rb_experiments
    z_obs_stats_to_survival_statistics
    fit_rb_results


Unitarity or Purity RB
----------------------

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    generate_unitarity_experiments
    estimate_purity
    estimate_purity_err
    fit_unitarity_results
    unitarity_to_rb_decay


Data Acquisition
----------------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    acquire_rb_data
    get_stats_by_qubit_group


Analysis Helper functions for RB
--------------------------------

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    coherence_angle
    gamma
    interleaved_gate_fidelity_bounds
    gate_error_to_irb_decay
    irb_decay_to_gate_error
    average_gate_error_to_rb_decay
    rb_decay_to_gate_error
    unitarity_to_rb_decay
    get_stats_by_qubit_group
