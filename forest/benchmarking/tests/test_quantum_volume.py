import numpy as np
import warnings
from pyquil.numpy_simulator import NumpyWavefunctionSimulator
from forest.benchmarking.quantum_volume import *
from forest.benchmarking.quantum_volume import _naive_program_generator

np.random.seed(1)


def test_ideal_sim_heavy_probs(qvm):
    qvm.qam.random_seed = 1
    depths = [2, 3]

    # silence warning from too few circuits, since 100 circuits is too slow
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outcomes = measure_quantum_volume(qvm, num_circuits=40, num_shots=50, qubits=[0, 1, 2],
                                          stop_when_fail=False)

    target_probs = [0.788765, 0.852895]
    probs = [outcomes[depth][0] for depth in depths]
    np.testing.assert_allclose(probs, target_probs, atol=.05)


def test_extraction():
    outcomes = {2: (.72, .68), 3: (.7, .67), 4: (.69, .66)}
    assert extract_quantum_volume_from_results(outcomes) == 8


def test_qv_get_results_by_depth(qvm):
    depths = [2, 3]
    n_ckts = 10
    n_shots = 5

    ckt_results = []
    ckt_hhs = []
    for depth in depths:
        wfn_sim = NumpyWavefunctionSimulator(depth)
        for _ in range(n_ckts):
            permutations, gates = generate_abstract_qv_circuit(depth)
            program = _naive_program_generator(qvm, qvm.qubits(), permutations, gates)

            program.wrap_in_numshots_loop(n_shots)
            executable = qvm.compiler.native_quil_to_executable(program)
            results = qvm.run(executable)
            ckt_results.append(results)

            heavy_outputs = collect_heavy_outputs(wfn_sim, permutations, gates)
            ckt_hhs.append(heavy_outputs)

    num_hh_sampled = count_heavy_hitters_sampled(ckt_results, ckt_hhs)
    probs_by_depth = get_prob_sample_heavy_by_depth(depths, num_hh_sampled,
                                                    [n_shots for _ in depths])

    assert len(probs_by_depth.keys()) == len(depths)
    assert [0 <= probs_by_depth[d][1] <= probs_by_depth[d][0] <= 1 for d in depths]
