import numpy as np
from pyquil.numpy_simulator import NumpyWavefunctionSimulator

from forest.benchmarking.volumetrics import *
from forest.benchmarking.volumetrics.quantum_volume import (collect_heavy_outputs,
                                                            get_success_probabilities,
                                                            calculate_success_prob_est_and_err)

np.random.seed(1)


def test_ideal_sim_heavy_probs(qvm):
    qvm.qam.random_seed = 1

    qv_ckt_template = get_quantum_volume_template()
    depths = [2, 3]
    dimensions = {d: [d] for d in depths}

    num_ckt_samples = 40
    qv_progs = generate_volumetric_program_array(qvm, qv_ckt_template, dimensions,
                                                 num_circuit_samples=num_ckt_samples)
    wfn_sim = NumpyWavefunctionSimulator(len(qvm.qubits()))
    heavy_outputs = collect_heavy_outputs(wfn_sim, qv_progs)

    num_shots = 50
    results = acquire_volumetric_data(qvm, qv_progs, num_shots=num_shots)

    probs_by_width_depth = get_success_probabilities(results, heavy_outputs)
    num_successes = [sum(probs_by_width_depth[depth][depth]) * num_shots for depth in depths]

    qv_success_probs = [calculate_success_prob_est_and_err(n_success, num_ckt_samples, num_shots)[0]
                        for n_success in num_successes]
    target_probs = [0.788765, 0.852895]
    np.testing.assert_allclose(qv_success_probs, target_probs, atol=.05)
