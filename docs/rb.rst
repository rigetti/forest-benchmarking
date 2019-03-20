.. currentmodule:: forest.benchmarking.rb

Randomized Benchmarking
=======================

Randomized benchmarking involves running long sequences of random Clifford group gates which
compose to the identity to observe how performance degrades with increasing circuit depth.

.. todo:: Talk some more about stuff.

Unitary RB
----------

.. todo:: Talk about unitarity RB.


API Reference
-------------

.. currentmodule:: forest.benchmarking.rb
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    rb_dataframe
    add_sequences_to_dataframe
    rb_seq_to_program
    run_rb_measurement
    oneq_rb_gateset
    twoq_rb_gateset
    get_rb_gateset
    generate_rb_sequence
    generate_simultaneous_rb_sequence
    merge_sequences
    survival_statistics
    survivals_from_results
    add_survivals
    survivals_by_qubits
    standard_rb
    standard_rb_guess
    fit_standard_rb
    estimate_purity
    estimate_purity_err
    shifted_purities_from_results
    add_shifted_purities
    shifted_purities_by_qubits

.. rubric:: Unitary RB

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    strip_inverse_from_sequences
    add_unitarity_sequences_to_dataframe
    run_unitarity_measurement
    unitarity_to_RB_decay
    unitarity_fn
    unitarity_guess
    fit_unitarity


.. rubric:: Interleaved RB

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    coherence_angle
    gamma
    interleaved_gate_fidelity_bounds
    gate_infidelity_to_irb_decay
    irb_decay_to_gate_infidelity
    average_gate_infidelity_to_RB_decay
    RB_decay_to_gate_fidelity
