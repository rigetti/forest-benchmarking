import numpy as np
from pyquil.noise import pauli_kraus_map
from forest.benchmarking.observable_estimation import ExperimentSetting, ExperimentResult, zeros_state
from forest.benchmarking.randomized_benchmarking import *
from forest.benchmarking.utils import all_traceless_pauli_z_terms


def add_noise_to_sequences(sequences, qubits, kraus_ops):
    """
    Append the given noise to each clifford gate (sequence)
    """
    for seq in sequences:
        for program in seq:
            program.defgate("noise", np.eye(2 ** len(qubits)))
            program.define_noisy_gate("noise", qubits, kraus_ops)
            program.inst(("noise", *qubits))


def test_1q_general_pauli_noise(qvm, benchmarker):
    qvm.qam.random_seed = 1
    expected_decay = .85
    probs = [expected_decay + .15 / 4, .06, .04, .0125]
    kraus_ops = pauli_kraus_map(probs)

    num_sequences_per_depth = 30
    num_shots = 50
    depths = [2, 8, 10, 16, 25]
    depths = [depth for depth in depths for _ in range(num_sequences_per_depth)]
    qubits = (0, )

    sequences = generate_rb_experiment_sequences(benchmarker, qubits, depths)
    add_noise_to_sequences(sequences, qubits, kraus_ops)

    expts = group_sequences_into_parallel_experiments([sequences], [qubits])

    results = acquire_rb_data(qvm, expts, num_shots)
    stats = get_stats_by_qubit_group([qubits], results)[qubits]
    fit = fit_rb_results(depths, stats['expectation'], stats['std_err'])

    observed_decay = fit.params['decay'].value
    decay_error = fit.params['decay'].stderr

    np.testing.assert_allclose(expected_decay, observed_decay, atol=2.5 * decay_error)


def test_2q_general_pauli_noise(qvm, benchmarker):
    qvm.qam.random_seed = 1

    expected_decay = .8
    probs = [expected_decay + .2 / 4, .06] + [0] * 12 + [.04, .05]
    kraus_ops = pauli_kraus_map(probs)

    num_sequences_per_depth = 5
    num_shots = 25
    depths = [2, 10, 12, 25]
    depths = [depth for depth in depths for _ in range(num_sequences_per_depth)]
    qubits = (0, 1)

    sequences = generate_rb_experiment_sequences(benchmarker, qubits, depths)
    add_noise_to_sequences(sequences, qubits, kraus_ops)

    expts = group_sequences_into_parallel_experiments([sequences], [qubits])

    results = acquire_rb_data(qvm, expts, num_shots)
    stats = get_stats_by_qubit_group([qubits], results)[qubits]
    fit = fit_rb_results(depths, stats['expectation'], stats['std_err'], num_shots)

    observed_decay = fit.params['decay'].value
    decay_error = fit.params['decay'].stderr

    np.testing.assert_allclose(expected_decay, observed_decay, atol=2.5 * decay_error)


def test_survival_statistics():
    # setup
    p0 = 0.98  # p(q0 = 0)
    p1 = 0.5   # p(q1 = 0)
    p_joint = p0*p1 + (1-p0)*(1-p1)
    expectations = [2*p0 - 1, 2*p1 - 1, 2*p_joint - 1]
    variances = np.asarray([p0*(1-p0), p1*(1-p1), p_joint*(1-p_joint)]) * 2**2
    n = 10000
    qubits = (0, 1)

    settings = [ExperimentSetting(zeros_state(qubits), op) for op in all_traceless_pauli_z_terms(qubits)]
    results = (ExperimentResult(setting, exp, n, std_err=np.sqrt(v/n)) for setting, exp, v
               in zip(settings, expectations, variances))
    stats = get_stats_by_qubit_group([qubits], [results])[qubits]
    exps = stats['expectation'][0]
    errs = stats['std_err'][0]

    np.testing.assert_allclose(exps, expectations)
    np.testing.assert_allclose(errs, np.sqrt(variances / n))

    survival_prob, survival_var = z_obs_stats_to_survival_statistics(exps, errs, n)

    np.testing.assert_allclose(survival_prob, p0*p1)
    np.testing.assert_allclose(np.sqrt(survival_var), .004999, rtol=1e-4)


def test_survival_statistics_2():
    # p0 is probability qubit 0 is 0
    for p0 in [1.0, .99]:
        exp = 2*p0 - 1
        variance = p0*(1-p0) * 2**2
        n = 100
        qubits = (0, )
        setting = [ExperimentSetting(zeros_state(qubits), op) for op in all_traceless_pauli_z_terms(qubits)][0]
        results = (ExperimentResult(setting, exp, n, std_err=np.sqrt(variance/n)), )

        stats = get_stats_by_qubit_group([qubits], [results])[qubits]
        exps = stats['expectation'][0]
        errs = stats['std_err'][0]

        np.testing.assert_allclose(exps, [exp])
        np.testing.assert_allclose(errs, [np.sqrt(variance / n)])

        survival_prob, survival_var = z_obs_stats_to_survival_statistics(exps, errs)

        assert survival_prob == p0


def test_survival_statistics_3():
    # p0 is probability qubit 0 is 0
    # p1 is probability qubit 1 is 0

    for p0, p1 in zip([1.0, .99, .99], [1.0, 1.0, .99]):
        # setup
        p_joint = p0*p1 + (1-p0)*(1-p1)
        expectations = [2*p0 - 1, 2*p1 - 1, 2*p_joint - 1]
        variances = np.asarray([p0*(1-p0), p1*(1-p1), p_joint*(1-p_joint)]) * 2**2
        n = 100
        qubits = (0, 1)
        settings = [ExperimentSetting(zeros_state(qubits), op) for op in all_traceless_pauli_z_terms(qubits)]
        results = (ExperimentResult(setting, exp, n, std_err=np.sqrt(v/n)) for setting, exp, v
                   in zip(settings, expectations, variances))

        stats = get_stats_by_qubit_group([qubits], [results])[qubits]
        exps = stats['expectation'][0]
        errs = stats['std_err'][0]

        np.testing.assert_allclose(exps, expectations)
        np.testing.assert_allclose(errs, np.sqrt(variances / n))

        survival_prob, survival_var = z_obs_stats_to_survival_statistics(exps, errs, n)
        assert survival_prob == p0 * p1


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
    probabilities = [p + (1.0 - p) / num_of_operators]
    probabilities += [(1.0 - p) / num_of_operators] * (num_of_operators - 1)
    return pauli_kraus_map(probabilities)


def test_unitarity(qvm, benchmarker):
    qvm.qam.random_seed = 6
    num_sequences_per_depth = 5
    num_shots = 50
    depths = [1, 6, 7, 8, 10]
    depths = [depth for depth in depths for _ in range(num_sequences_per_depth)]
    qubits = (0, )
    expected_p = .90
    kraus_ops = depolarizing_noise(len(qubits), expected_p)

    sequences = generate_rb_experiment_sequences(benchmarker, qubits, depths,
                                                 use_self_inv_seqs=False)
    add_noise_to_sequences(sequences, qubits, kraus_ops)

    expts = group_sequences_into_parallel_experiments([sequences], [qubits], is_unitarity_expt=True)

    results = acquire_rb_data(qvm, expts, num_shots)
    stats = get_stats_by_qubit_group([qubits], results)[qubits]
    fit = fit_unitarity_results(depths, stats['expectation'], stats['std_err'])

    observed_unitarity = fit.params['decay'].value
    unitarity_error = fit.params['decay'].stderr
    # in the case of depolarizing noise with parameter p, observed unitarity should correspond
    # via the upper bound
    observed_p = unitarity_to_rb_decay(observed_unitarity, 2**len(qubits))
    tol = abs(observed_p - unitarity_to_rb_decay(observed_unitarity - 2.5 * unitarity_error,
                                                 2 ** len(qubits)))
    # TODO: properly incorporate unitarity error into tol
    np.testing.assert_allclose(expected_p, observed_p, atol=tol)
