.. currentmodule:: forest.benchmarking.quantum_volume
Quantum Volume
==============

The quantum volume is holistic quantum computer performance measure (see https://arxiv
.org/abs/1811.12926).

Roughly the logarithm (base 2) of quantum volume $V_Q$, quantifies the largest random circuit of
equal width and depth that the computer successfully implements and is certifiably quantum. So if you have 64 qubits and the best log-QV you have measured is $\\log_2V_Q = 3$ the effectively you only have an 8 qubit device!


.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    collect_heavy_outputs
    generate_abstract_qv_circuit
    sample_rand_circuits_for_heavy_out
    calculate_prob_est_and_err
    measure_quantum_volume
    count_heavy_hitters_sampled
    get_prob_sample_heavy_by_depth
    extract_quantum_volume_from_results
