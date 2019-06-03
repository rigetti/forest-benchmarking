import numpy as np
import warnings
from forest.benchmarking.quantum_volume import *

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


def test_qv_df_generation():
    unique_depths = [2, 3]
    n_ckts = 100
    depths = [d for d in unique_depths for _ in range(n_ckts)]

    ckts = list(generate_quantum_volume_abstract_circuits(depths))

    assert len(ckts) == len(unique_depths)*n_ckts

    assert all([len(ckt[0]) == depth for ckt, depth in zip(ckts, depths)])
    assert all([len(ckt[0][0]) == depth for ckt, depth in zip(ckts, depths)])

    assert all([ckt[1].shape == (depth, depth//2, 4, 4) for ckt, depth in zip(ckts, depths)])


def test_qv_data_acquisition(qvm):
    unique_depths = [2, 3]
    n_ckts = 10
    depths = [d for d in unique_depths for _ in range(n_ckts)]
    n_shots = 5

    ckts = generate_quantum_volume_abstract_circuits(depths)
    progs = abstract_circuits_to_programs(qvm, ckts)
    results = [res[0] for res in acquire_quantum_volume_data(qvm, progs, n_shots)]

    assert all([res.shape == (n_shots, depth) for res, depth in zip(results, depths)])


def test_qv_count_heavy_hitters(qvm):
    unique_depths = [2, 3]
    n_ckts = 10
    depths = [d for d in unique_depths for _ in range(n_ckts)]
    n_shots = 5

    ckts = list(generate_quantum_volume_abstract_circuits(depths))
    progs = abstract_circuits_to_programs(qvm, ckts)
    results = [res[0] for res in acquire_quantum_volume_data(qvm, progs, n_shots)]
    hhs = [res[0] for res in acquire_heavy_hitters(ckts)]

    assert all([0 <= num_hh <= n_shots for num_hh in count_heavy_hitters_sampled(results, hhs)])


def test_qv_get_results_by_depth(qvm):
    unique_depths = [2, 3]
    n_ckts = 10
    depths = [d for d in unique_depths for _ in range(n_ckts)]
    n_shots = 5

    ckts = list(generate_quantum_volume_abstract_circuits(depths))
    progs = abstract_circuits_to_programs(qvm, ckts)
    results = [res[0] for res in acquire_quantum_volume_data(qvm, progs, n_shots)]
    hhs = [res[0] for res in acquire_heavy_hitters(ckts)]
    probs_by_depth = get_results_by_depth(depths, count_heavy_hitters_sampled(results, hhs),
                                          [n_shots for _ in depths])

    assert len(probs_by_depth.keys()) == len(unique_depths)
    assert [0 <= probs_by_depth[d][1] <= probs_by_depth[d][0] <= 1 for d in depths]
