import numpy as np
from typing import Tuple, Callable

from pyquil import Program
from pyquil.noise import pauli_kraus_map
from pyquil.operator_estimation import ExperimentSetting, ExperimentResult, zeros_state
from forest.benchmarking.randomized_benchmarking import *
from forest.benchmarking.stratified_experiment import Layer
from forest.benchmarking.utils import all_pauli_z_terms


def insert_noise(programs: Tuple[Program], qubits: Tuple, noise: Callable, *noise_args):
    """
    Append noise channel to the end of each program in programs.
    This noise channel is implemented as a single noisy gate acting on the provided qubits.

    :param programs: A list of programs (i.e. a Clifford gate) onto each of which will be appended noise.
    :param qubits: A tuple of the qubits on which each noisy gate should act.
    :param noise: A function which generates the kraus operators of the desired noise.
    :param noise_args: Additional parameters passed on to the noise function.
    """
    for program in programs:
        program.defgate("noise", np.eye(2 ** len(qubits)))
        program.define_noisy_gate("noise", qubits, noise(*noise_args))
        program.inst(("noise", *qubits))


def add_noise_to_sequences(expt: StratifiedExperiment, qubits, noise: Callable, *noise_args):
    """
    Append the given noise to each clifford gate (sequence)
    :param qubits: A tuple of the qubits on which each sequence of Cliffords acts
    :param noise: Function which takes in a gate and appends the desired Krauss operators
    :param noise_args: Additional parameters passed on to the noise function.
    """
    for layer in expt.layers:
        insert_noise(layer.sequence, qubits, noise, *noise_args)


def test_1q_general_pauli_noise(qvm, benchmarker):
    qvm.qam.random_seed = 5
    expected_decay = .85
    probs = [expected_decay + .15 / 4, .06, .04, .0125]

    num_sequences_per_depth = 30
    num_shots = 35
    depths = [2, 8, 9, 10, 11, 20]
    qubits = [0]

    expt = generate_rb_experiment(benchmarker, qubits, depths, num_sequences_per_depth)
    add_noise_to_sequences(expt, qubits, pauli_kraus_map, probs)

    results = acquire_rb_data(qvm, [expt], num_shots)
    fit = fit_rb_results(results[0])

    observed_decay = fit.params['decay'].value
    decay_error = fit.params['decay'].stderr

    assert (np.abs(expected_decay - observed_decay) < 2 * decay_error)


def test_2q_general_pauli_noise(qvm, benchmarker):
    qvm.qam.random_seed = 5
    expected_decay = .8
    probs = [expected_decay + .2 / 4, .06] + [0] * 12 + [.04, .05]

    num_sequences_per_depth = 5
    num_shots = 25
    depths = [2, 10, 12, 25]
    qubits = (0, 1)

    expt = generate_rb_experiment(benchmarker, qubits, depths, num_sequences_per_depth)
    add_noise_to_sequences(expt, qubits, pauli_kraus_map, probs)

    results = acquire_rb_data(qvm, [expt], num_shots)
    fit = fit_rb_results(results[0])

    observed_decay = fit.params['decay'].value
    decay_error = fit.params['decay'].stderr

    assert (np.abs(expected_decay - observed_decay) < 2 * decay_error)


def test_survival_statistics():
    # setup
    p0 = 0.98  # p(q0 = 0)
    p1 = 0.5   # p(q1 = 0)
    p_joint = p0*p1 + (1-p0)*(1-p1)
    expectations = [1, 2*p0 - 1, 2*p1 - 1, 2*p_joint - 1]
    variances = [0, p0*(1-p0), p1*(1-p1), p_joint*(1-p_joint)]
    n = 10000
    qubits = (0, 1)
    settings = [ExperimentSetting(zeros_state(qubits), op) for op in all_pauli_z_terms(qubits)]
    results = (ExperimentResult(setting, exp, np.sqrt(var), n) for setting, exp, var
               in zip(settings, expectations, variances))
    layer = Layer(1, None, None, qubits, results=results, num_shots=n)
    expt = StratifiedExperiment([layer], qubits)

    populate_rb_survival_statistics(expt)
    mean, err = expt.layers[0].estimates['Survival']
    np.testing.assert_allclose(mean, .49, rtol=1e-5)
    np.testing.assert_allclose(err, .004998, rtol=1e-4)


def test_survival_statistics_2():
    p0 = 1  # p(q0 = 0)
    expectations = [1, 2*p0 - 1]
    variances = [0, p0*(1-p0)]
    n = 100
    qubits = (0, )
    settings = [ExperimentSetting(zeros_state(qubits), op) for op in all_pauli_z_terms(qubits)]
    results = (ExperimentResult(setting, exp, np.sqrt(var), n) for setting, exp, var
               in zip(settings, expectations, variances))
    layer = Layer(1, None, None, qubits, results=results, num_shots=n)
    expt = StratifiedExperiment([layer], qubits)

    populate_rb_survival_statistics(expt)
    mean, err = expt.layers[0].estimates['Survival']
    assert mean == 101 / 102

    p0 = .99  # p(q0 = 0)
    expectations = [1, 2*p0 - 1]
    variances = [0, p0*(1-p0)]
    n = 100
    qubits = (0, )
    settings = [ExperimentSetting(zeros_state(qubits), op) for op in all_pauli_z_terms(qubits)]
    results = (ExperimentResult(setting, exp, np.sqrt(var), n) for setting, exp, var
               in zip(settings, expectations, variances))
    layer = Layer(1, None, None, qubits, results=results, num_shots=n)
    expt = StratifiedExperiment([layer], qubits)

    populate_rb_survival_statistics(expt)
    mean, err = expt.layers[0].estimates['Survival']
    assert mean == 100 / 102


def test_survival_statistics_3():
    # setup
    p0 = 1.0  # p(q0 = 0)
    p1 = 1.0   # p(q1 = 0)
    p_joint = p0*p1 + (1-p0)*(1-p1)
    expectations = [1, 2*p0 - 1, 2*p1 - 1, 2*p_joint - 1]
    variances = [0, p0*(1-p0), p1*(1-p1), p_joint*(1-p_joint)]
    n = 100
    qubits = (0, 1)
    settings = [ExperimentSetting(zeros_state(qubits), op) for op in all_pauli_z_terms(qubits)]
    results = (ExperimentResult(setting, exp, np.sqrt(var), n) for setting, exp, var
               in zip(settings, expectations, variances))
    layer = Layer(1, None, None, qubits, results=results, num_shots=n)
    expt = StratifiedExperiment([layer], qubits)

    populate_rb_survival_statistics(expt)
    mean, err = expt.layers[0].estimates['Survival']
    assert mean == 101 / 102

    p0 = .99  # p(q0 = 0)
    p1 = 1.0   # p(q1 = 0)
    expectations = [1, 2*p0 - 1, 2*p1 - 1, (99 - 1)/100.]
    variances = [0, p0*(1-p0), p1*(1-p1), p_joint*(1-p_joint)]
    n = 100
    qubits = (0, 1)
    settings = [ExperimentSetting(zeros_state(qubits), op) for op in all_pauli_z_terms(qubits)]
    results = (ExperimentResult(setting, exp, np.sqrt(var), n) for setting, exp, var
               in zip(settings, expectations, variances))
    layer = Layer(1, None, None, qubits, results=results, num_shots=n)
    expt = StratifiedExperiment([layer], qubits)

    populate_rb_survival_statistics(expt)
    mean, err = expt.layers[0].estimates['Survival']
    assert mean == 100 / 102

    p0 = .99  # p(q0 = 0)
    p1 = .99   # p(q1 = 0)
    expectations = [1, 2*p0 - 1, 2*p1 - 1, (99 + 1 - 0)/100.]
    variances = [0, p0*(1-p0), p1*(1-p1), p_joint*(1-p_joint)]
    n = 100
    qubits = (0, 1)
    settings = [ExperimentSetting(zeros_state(qubits), op) for op in all_pauli_z_terms(qubits)]
    results = (ExperimentResult(setting, exp, np.sqrt(var), n) for setting, exp, var
               in zip(settings, expectations, variances))
    layer = Layer(1, None, None, qubits, results=results, num_shots=n)
    expt = StratifiedExperiment([layer], qubits)

    populate_rb_survival_statistics(expt)
    mean, err = expt.layers[0].estimates['Survival']
    assert mean == 100 / 102


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
    num_shots = 50
    depths = [1, 6, 7, 8, 10]
    qubits = (0, )
    expected_p = .90

    expt = generate_unitarity_experiment(benchmarker, qubits, depths, num_sequences_per_depth)
    add_noise_to_sequences(expt, qubits, depolarizing_noise, len(qubits), expected_p)

    results = acquire_unitarity_data(qvm, [expt], num_shots)
    fit = fit_unitarity_results(results[0])

    observed_unitarity = fit.params['decay'].value
    unitarity_error = fit.params['decay'].stderr
    # in the case of depolarizing noise with parameter p, observed unitarity should correspond
    # via the upper bound
    observed_p = unitarity_to_rb_decay(observed_unitarity, 2**len(qubits))
    tol = abs(observed_p - unitarity_to_rb_decay(observed_unitarity - 2 * unitarity_error,
                                                 2 ** len(qubits)))
    # TODO: properly incorporate unitarity error into tol
    np.testing.assert_allclose(expected_p, observed_p, atol=tol)
