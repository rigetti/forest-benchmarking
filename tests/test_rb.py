import numpy as np
from numpy import random, uint8, zeros

from pyquil.gates import X, Y
from pyquil.quil import Program
from forest_benchmarking.rb import merge_sequences, fit_standard_rb, rb_dataframe, \
    add_sequences_to_dataframe, run_rb_measurement, survivals_by_qubits, add_survivals, survival_statistics, \
    fit_unitarity, add_unitarity_sequences_to_dataframe, \
    run_unitarity_measurement, add_shifted_purities, shifted_purities_by_qubits, unitarity_to_RB_decay
from typing import List, Tuple, Callable
from pandas import DataFrame
from pyquil.noise import pauli_kraus_map


def insert_noise(programs: List[Program], qubits: Tuple, noise: Callable, *noise_args):
    """
    Append noise channel to the end of each program in programs.
    This noise channel is implemented as a single noisy gate acting on the provided qubits.

    :param list|program programs: A list of programs (i.e. a Clifford gate) onto each of which will be appended noise.
    :param Tuple qubits: A tuple of the qubits on which each noisy gate should act.
    :param noise: A function which generates the kraus operators of the desired noise.
    :param noise_args: Additional parameters passed on to the noise function.
    """
    for program in programs:
        program.defgate("noise", np.eye(2 ** len(qubits)))
        program.define_noisy_gate("noise", qubits, noise(*noise_args))
        program.inst(("noise", *qubits))


def add_noise_to_sequences(df: DataFrame, qubits: Tuple, noise: Callable, *noise_args):
    """
    Append the given noise to each clifford gate (sequence)
    :param df: DataFrame with Sequence column. Noise will be added to each clifford gate program in each sequence.
    :param qubits: A tuple of the qubits on which each sequence of Cliffords acts
    :param noise: Function which takes in a gate and appends the desired Krauss operators
    :param noise_args: Additional parameters passed on to the noise function.
    """
    new_df = df.copy()
    for seq in new_df["Sequence"].values:
        insert_noise(seq, qubits, noise, *noise_args)
    return new_df


def test_1q_general_pauli_noise(qvm, benchmarker):
    qvm.qam.random_seed = 5
    expected_decay = .85
    probs = [expected_decay + .15 / 4, .06, .04, .0125]

    num_sequences_per_depth = 30
    num_trials_per_seq = 35
    depths = [2, 8, 9, 10, 11, 20]
    subgraph = [(0,)]

    num_qubits = len(subgraph[0])
    rb_type = "sim-1q" if num_qubits == 1 else "sim-2q"

    df = rb_dataframe(rb_type=rb_type,
                      subgraph=subgraph,
                      depths=depths,
                      num_sequences=num_sequences_per_depth)

    df = add_sequences_to_dataframe(df, benchmarker, random_seed=5)
    df = add_noise_to_sequences(df, subgraph[0], pauli_kraus_map, probs)

    df = run_rb_measurement(df, qvm, num_trials=num_trials_per_seq)
    df = add_survivals(df)

    depths, survivals, survival_errs = {}, {}, {}
    for qubits in subgraph:
        depths[qubits], survivals[qubits], survival_errs[qubits] = survivals_by_qubits(df, qubits)

    fit = fit_standard_rb(depths[subgraph[0]], survivals[subgraph[0]], weights=1 / survival_errs[subgraph[0]])
    observed_decay = fit.params['decay'].value
    decay_error = fit.params['decay'].stderr

    assert (np.abs(expected_decay - observed_decay) < 2 * decay_error)


def test_2q_general_pauli_noise(qvm, benchmarker):
    qvm.qam.random_seed = 5
    expected_decay = .8
    probs = [expected_decay + .2 / 4, .06] + [0] * 12 + [.04, .05]

    num_sequences_per_depth = 5
    num_trials_per_seq = 25
    depths = [2, 10, 12, 25]
    subgraph = [(0, 1)]  # for two qubit

    num_qubits = len(subgraph[0])
    rb_type = "sim-1q" if num_qubits == 1 else "sim-2q"

    df = rb_dataframe(rb_type=rb_type,
                      subgraph=subgraph,
                      depths=depths,
                      num_sequences=num_sequences_per_depth)

    df = add_sequences_to_dataframe(df, benchmarker, random_seed=5)
    df = add_noise_to_sequences(df, subgraph[0], pauli_kraus_map, probs)

    df = run_rb_measurement(df, qvm, num_trials=num_trials_per_seq)
    df = add_survivals(df)

    depths, survivals, survival_errs = {}, {}, {}
    for qubits in subgraph:
        depths[qubits], survivals[qubits], survival_errs[qubits] = survivals_by_qubits(df, qubits)

    fit = fit_standard_rb(depths[subgraph[0]], survivals[subgraph[0]], weights=1 / survival_errs[subgraph[0]])
    observed_decay = fit.params['decay'].value
    decay_error = fit.params['decay'].stderr

    assert (np.abs(expected_decay - observed_decay) < 2 * decay_error)


def test_survival_statistics():
    # setup
    p0 = 0.02
    p1 = 0.5
    n = 10000
    random.seed(0)
    bit_arrays = zeros((n, 2), dtype=uint8)
    for _ in range(n):
        bit_arrays[_, 0] = 1 if np.random.random() < p0 else 0
        bit_arrays[_, 1] = 1 if np.random.random() < p1 else 0

    mean, err = survival_statistics(bit_arrays)
    np.testing.assert_allclose(mean, .485403, rtol=1e-5)
    np.testing.assert_allclose(err, .004997, rtol=1e-4)


def test_survival_statistics_2():
    random.seed(0)
    bitstrings = np.zeros((100, 1))
    mean, err = survival_statistics(bitstrings)
    assert mean == 101 / 102

    bitstrings[0] = 1
    mean, err = survival_statistics(bitstrings)
    assert mean == 100 / 102


def test_survival_statistics_3():
    random.seed(0)
    bitstrings = np.zeros((100, 2))
    mean, err = survival_statistics(bitstrings)
    assert mean == 101 / 102

    bitstrings[0, 0] = 1
    mean, err = survival_statistics(bitstrings)
    assert mean == 100 / 102

    bitstrings[0, 1] = 1
    mean, err = survival_statistics(bitstrings)
    assert mean == 100 / 102


def test_merge_sequences():
    random.seed(0)
    seq0 = [Program(X(0)), Program(Y(0)), Program(X(0))]
    seq1 = [Program(X(1)), Program(Y(1)), Program(Y(1))]
    assert merge_sequences([seq0, seq1]) == [Program(X(0), X(1)),
                                             Program(Y(0), Y(1)),
                                             Program(X(0), Y(1))]


######
# Unitarity test
######


def depolarizing_noise(num_qubits: int, p: float = .95):
    """
    Generate the Kraus operators corresponding to a given unitary
    single qubit gate followed by a depolarizing noise channel.

    :params float num_qubits: either 1 or 2 qubit channel supported
    :params float p: parameter in depolarizing channel as defined by: p $\rho$ + (1-p)/d I
    :return: A list, eg. [k0, k1, k2, k3], of the Kraus operators that parametrize the map.
    :rtype: list
    """
    num_of_operators = 4 ** num_qubits
    probabilities = [p + (1.0 - p) / num_of_operators] + [(1.0 - p) / num_of_operators] * (num_of_operators - 1)
    return pauli_kraus_map(probabilities)


def test_unitarity(qvm, benchmarker):
    qvm.qam.random_seed = 6
    num_sequences_per_depth = 5
    num_trials_per_seq = 25
    depths = [1, 6, 7, 8]
    subgraph = [(0, )]
    num_qubits = len(subgraph[0])
    rb_type = "sim-1q"

    df = rb_dataframe(rb_type=rb_type,
                      subgraph=subgraph,
                      depths=depths,
                      num_sequences=num_sequences_per_depth)

    df = add_unitarity_sequences_to_dataframe(df, benchmarker, random_seed=1)

    expected_p = .90
    df = add_noise_to_sequences(df, subgraph[0], depolarizing_noise, num_qubits, expected_p)

    df = run_unitarity_measurement(df, qvm, num_trials=num_trials_per_seq)
    df = add_shifted_purities(df)

    depths, purities, purity_errs = {}, {}, {}
    for qubits in subgraph:
        depths[qubits], purities[qubits], purity_errs[qubits] = shifted_purities_by_qubits(df, qubits)

    fit = fit_unitarity(depths[subgraph[0]],
                        purities[subgraph[0]],
                        weights=1/purity_errs[subgraph[0]])

    observed_unitarity = fit.params['unitarity'].value
    unitarity_error = fit.params['unitarity'].stderr
    # in the case of depolarizing noise with parameter p, observed unitarity should correspond via the upper bound
    observed_p = unitarity_to_RB_decay(observed_unitarity, 2**num_qubits)
    tol = abs(observed_p - unitarity_to_RB_decay(observed_unitarity-2*unitarity_error, 2**num_qubits))
    # TODO: properly incorporate unitarity error into tol
    np.testing.assert_allclose(expected_p, observed_p, atol=0.1)
