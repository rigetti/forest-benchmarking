from forest.benchmarking.direct_fidelity_estimation import generate_exhaustive_state_dfe_experiment, generate_exhaustive_process_dfe_experiment, \
    generate_monte_carlo_state_dfe_experiment, generate_monte_carlo_process_dfe_experiment, acquire_dfe_data, estimate_dfe

from pyquil import Program
from pyquil.api import BenchmarkConnection
from pyquil.gates import *
from pyquil.numpy_simulator import NumpyWavefunctionSimulator
from forest.benchmarking.observable_estimation import _one_q_state_prep

from numpy.testing import assert_almost_equal, assert_allclose
from test_process_tomography import test_qc


def test_exhaustive_state_dfe(benchmarker: BenchmarkConnection):
    texpt = generate_exhaustive_state_dfe_experiment(program=Program(X(0), X(1)), qubits=[0, 1],
                                                     benchmarker=benchmarker)
    assert len(texpt) == 2 ** 2 - 1


def test_mc_state_dfe(benchmarker: BenchmarkConnection):
    texpt = generate_monte_carlo_state_dfe_experiment(program=Program(X(0), X(1), X(2), X(3), X(4), X(5), X(6)),
                                                      qubits=[0, 1, 2, 3, 4, 5, 6],
                                                      n_terms=50, benchmarker=benchmarker)
    assert len(texpt) == 50


def test_exhaustive_dfe(benchmarker: BenchmarkConnection):
    texpt = generate_exhaustive_process_dfe_experiment(program=Program(Z(0)), qubits=[0], benchmarker=benchmarker)
    assert len(texpt) == 7 ** 1 - 1


def test_exhaustive_process_dfe_run(benchmarker: BenchmarkConnection):
    wfnsim = NumpyWavefunctionSimulator(n_qubits=1)
    process = Program(Z(0))
    texpt = generate_exhaustive_process_dfe_experiment(program=process, qubits=[0], benchmarker=benchmarker)
    for setting in texpt:
        setting = setting[0]
        prog = Program()
        for oneq_state in setting.in_state.states:
            prog += _one_q_state_prep(oneq_state)
        prog += process

        expectation = wfnsim.reset().do_program(prog).expectation(setting.observable)
        assert expectation == 1.


def test_exhaustive_state_dfe_run(benchmarker: BenchmarkConnection):
    wfnsim = NumpyWavefunctionSimulator(n_qubits=1)
    process = Program(X(0))
    texpt = generate_exhaustive_state_dfe_experiment(program=process, qubits=[0], benchmarker=benchmarker)
    for setting in texpt:
        setting = setting[0]
        prog = Program()
        for oneq_state in setting.in_state.states:
            prog += _one_q_state_prep(oneq_state)
        prog += process

        expectation = wfnsim.reset().do_program(prog).expectation(setting.observable)
        assert expectation == 1.


def test_monte_carlo_process_dfe(benchmarker: BenchmarkConnection):
    process = Program(CNOT(0, 1))
    texpt = generate_monte_carlo_process_dfe_experiment(program=process, qubits=[0, 1], n_terms=10,
                                                        benchmarker=benchmarker)
    assert len(texpt) == 10

    wfnsim = NumpyWavefunctionSimulator(n_qubits=2)
    for setting in texpt:
        setting = setting[0]
        prog = Program()
        for oneq_state in setting.in_state.states:
            prog += _one_q_state_prep(oneq_state)
        prog += process
        print(setting.observable)
        expectation = wfnsim.reset().do_program(prog).expectation(setting.observable)
        assert_almost_equal(expectation, 1., decimal=7)


def test_monte_carlo_state_dfe(benchmarker: BenchmarkConnection):
    process = Program(H(0), CNOT(0, 1))
    texpt = generate_monte_carlo_state_dfe_experiment(program=process, qubits=[0, 1], n_terms=10,
                                                      benchmarker=benchmarker)
    assert len(texpt) == 10

    wfnsim = NumpyWavefunctionSimulator(n_qubits=2)
    for setting in texpt:
        setting = setting[0]
        prog = Program()
        for oneq_state in setting.in_state.states:
            prog += _one_q_state_prep(oneq_state)
        prog += process
        print(setting.observable)
        expectation = wfnsim.reset().do_program(prog).expectation(setting.observable)
        assert_almost_equal(expectation, 1., decimal=7)


def test_acquire_dfe_data(benchmarker: BenchmarkConnection, test_qc):
    # pick (Clifford) process that acts as identity on qubits 0 and 1
    process = Program(X(2), X(3))
    texpt = generate_exhaustive_state_dfe_experiment(program=process,
                                                     qubits=[0, 1], benchmarker=benchmarker)
    dfe_data = acquire_dfe_data(qc=test_qc, expr=texpt)
    fid_est, fid_std_err = estimate_dfe(dfe_data, 'state')

    assert_allclose(fid_est, 1.0)
    assert_allclose(fid_std_err, 0.0)
