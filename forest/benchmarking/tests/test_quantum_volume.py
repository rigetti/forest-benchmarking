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
        outcomes = measure_quantum_volume(qvm, num_circuits=40, num_shots=50, qubits=[0, 1, 2])

    assert extract_quantum_volume_from_results(outcomes) == 8
    target_probs = [0.788765, 0.852895]
    probs = [outcomes[depth][0] for depth in depths]
    np.testing.assert_allclose(probs, target_probs, atol=.05)


def test_qv_df_generation():
    depths = [2, 3]
    n_ckts = 100

    df = generate_quantum_volume_experiments(depths, n_ckts)
    df_depths = df["Depth"].values
    ckts = df["Abstract Ckt"].values

    assert len(df_depths) == len(depths)*n_ckts

    assert all([len(ckt[0]) == depth for ckt, depth in zip(ckts, df_depths)])
    assert all([len(ckt[0][0]) == depth for ckt, depth in zip(ckts, df_depths)])

    assert all([ckt[1].shape == (depth, depth//2, 4, 4) for ckt, depth in zip(ckts, df_depths)])


def test_qv_data_acquisition(qvm):
    depths = [2, 3]
    n_ckts = 10
    n_shots = 5

    df = generate_quantum_volume_experiments(depths, n_ckts)
    df = add_programs_to_dataframe(df, qvm)
    df = acquire_quantum_volume_data(df, qvm, n_shots)

    df_depths = df["Depth"].values
    results = df["Results"].values

    assert all([res.shape == (n_shots, depth) for res, depth in zip(results, df_depths)])


def test_qv_count_heavy_hitters(qvm):
    depths = [2, 3]
    n_ckts = 10
    n_shots = 5

    df = generate_quantum_volume_experiments(depths, n_ckts)
    df = add_programs_to_dataframe(df, qvm)
    df = acquire_quantum_volume_data(df, qvm, n_shots)
    df = acquire_heavy_hitters(df)

    num_hhs = df["Num HH Sampled"].values

    assert all([0 <= num_hh <= n_shots for num_hh in num_hhs])


def test_qv_get_results_by_depth(qvm):

    depths = [2, 3]
    n_ckts = 10
    n_shots = 5

    df = generate_quantum_volume_experiments(depths, n_ckts)
    df = add_programs_to_dataframe(df, qvm)
    df = acquire_heavy_hitters(df)
    df = acquire_quantum_volume_data(df, qvm, n_shots)
    results = get_results_by_depth(df)

    assert len(results.keys()) == len(depths)
    assert [0 <= results[d][1] <= results[d][0] <= 1 for d in depths]
