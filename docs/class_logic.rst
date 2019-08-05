.. module:: forest.benchmarking.classical_logic

Classical Logic
===============

This module allows us to use a "simple" reversible binary adder to benchmark a quantum computer.
The code is contained in the module `classical_logic`.

The benchmark is simplistic and not very rigorous as it does not test any specific feature of the hardware. Further the whole circuit is classical in the sense that we start and end in computational basis states and all gates simply perform classical not, controlled not (`CNOT`), or doubly controlled not (`CCNOT` aka a Toffoli gate). Finally, even for the modest task of adding two one bit numbers, the `CZ` gate (our fundamental two qubit gate) count is very high for the circuit. This in turn implies a low probability of the entire circuit working.


.. toctree::

    examples/ripple_adder_benchmark

Circuit Primitives
------------------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    CNOT_X_basis
    CCNOT_X_basis
    majority_gate
    unmajority_add_gate
    unmajority_add_parallel_gate


Ripple Carry adder
------------------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    assign_registers_to_line_or_cycle
    get_qubit_registers_for_adder
    adder
    get_n_bit_adder_results
    get_success_probabilities_from_results
    get_error_hamming_distributions_from_results
