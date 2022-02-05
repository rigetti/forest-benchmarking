import numpy as np
import pytest

from forest.benchmarking.operator_tools.random_operators import haar_rand_unitary
from forest.benchmarking.operator_tools.superoperator_transformations import kraus2choi
from forest.benchmarking.tomography import generate_process_tomography_experiment, \
    pgdb_process_estimate, linear_inv_process_estimate, do_tomography
from forest.benchmarking.observable_estimation import estimate_observables, ExperimentResult, \
    ObservablesExperiment, \
    _one_q_state_prep
from pyquil import Program
from pyquil.simulation import matrices as mat
from pyquil.gates import CNOT, X, H
from pyquil.simulation import NumpyWavefunctionSimulator


def wfn_estimate_observables(n_qubits, tomo_expt: ObservablesExperiment):
    if len(tomo_expt.program.defined_gates) > 0:
        raise pytest.skip("Can't do wfn on defined gates yet")
    wfn = NumpyWavefunctionSimulator(n_qubits)
    for settings in tomo_expt:
        for setting in settings:
            prog = Program()
            for oneq_state in setting.in_state.states:
                prog += _one_q_state_prep(oneq_state)
            prog += tomo_expt.program

            yield ExperimentResult(
                setting=setting,
                expectation=wfn.reset().do_program(prog).expectation(setting.observable),
                std_err=0.,
                total_counts=1,  # don't set to zero unless you want nans
            )


@pytest.fixture(params=['pauli', 'sic'])
def basis(request):
    return request.param


@pytest.fixture(params=['sampling', 'wfn'])
def measurement_func(request, test_qc):
    if request.param == 'wfn':
        return lambda expt: list(wfn_estimate_observables(n_qubits=2, tomo_expt=expt))
    elif request.param == 'sampling':
        return lambda expt: list(estimate_observables(qc=test_qc, obs_expt=expt, num_shots=500))
    else:
        raise ValueError()


@pytest.fixture(params=['X', 'haar'])
def single_q_process(request):
    if request.param == 'X':
        return Program(X(0)), mat.X
    elif request.param == 'haar':
        u_rand = haar_rand_unitary(2 ** 1, rs=np.random.RandomState(52))
        process = Program().defgate("RandUnitary", u_rand)
        process += ("RandUnitary", 0)
        return process, u_rand


@pytest.fixture()
def single_q_tomo_fixture(basis, single_q_process, measurement_func):
    qubits = [0]
    process, u_rand = single_q_process
    tomo_expt = generate_process_tomography_experiment(process, qubits, in_basis=basis)
    results = measurement_func(tomo_expt)

    return qubits, results, u_rand


def test_single_q_pgdb(single_q_tomo_fixture):
    qubits, results, u_rand = single_q_tomo_fixture

    process_choi_lin_inv_est = linear_inv_process_estimate(results, qubits)
    process_choi_est = pgdb_process_estimate(results, qubits)
    process_choi_true = kraus2choi(u_rand)
    np.testing.assert_allclose(process_choi_true, process_choi_lin_inv_est, atol=.05)
    np.testing.assert_allclose(process_choi_true, process_choi_est, atol=.05)


@pytest.fixture(params=['haar'])
def two_q_process(request):
    if request.param == 'CNOT':
        return Program(CNOT(0, 1)), mat.CNOT
    elif request.param == 'haar':
        u_rand = haar_rand_unitary(2 ** 2, rs=np.random.RandomState(52))
        process = Program().defgate("RandUnitary", u_rand)
        process += ("RandUnitary", 0, 1)
        return process, u_rand
    else:
        raise ValueError()


@pytest.fixture()
def two_q_tomo_fixture(basis, two_q_process, measurement_func):
    qubits = [0, 1]
    process, u_rand = two_q_process
    if basis == 'pauli':
        raise pytest.skip("This test is currently too slow.")
    tomo_expt = generate_process_tomography_experiment(process, qubits, in_basis=basis)
    results = measurement_func(tomo_expt)
    return qubits, results, u_rand


def test_two_q(two_q_tomo_fixture):
    qubits, results, u_rand = two_q_tomo_fixture
    process_choi_lin_inv_est = linear_inv_process_estimate(results, qubits)
    process_choi_est = pgdb_process_estimate(results, qubits)
    process_choi_true = kraus2choi(u_rand)
    np.testing.assert_allclose(process_choi_true, process_choi_lin_inv_est, atol=.1)
    np.testing.assert_allclose(process_choi_true, process_choi_est, atol=0.05)


def test_do_tomography(qvm):
    qubit = 1
    process = Program(H(qubit))
    est, _, _ = do_tomography(qvm, process, qubits=[qubit], kind='process')

    np.testing.assert_allclose(est, kraus2choi(mat.H), atol=.1)
