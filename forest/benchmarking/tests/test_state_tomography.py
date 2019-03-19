import networkx as nx
import numpy as np
import pytest
from requests.exceptions import RequestException
from forest.benchmarking.compilation import basic_compile
from forest.benchmarking.random_operators import haar_rand_unitary
from forest.benchmarking.tomography import generate_state_tomography_experiment, _R, \
    iterative_mle_state_estimate, project_density_matrix, estimate_variance, \
    linear_inv_state_estimate
from pyquil.api import ForestConnection, QuantumComputer, QVM
from pyquil.api._compiler import _extract_attribute_dictionary_from_program
from pyquil.api._qac import AbstractCompiler
from pyquil.device import NxDevice
from pyquil.gates import I, H, CZ
from pyquil.numpy_simulator import NumpyWavefunctionSimulator
from pyquil.operator_estimation import measure_observables
from pyquil.quil import Program
from rpcq.messages import PyQuilExecutableResponse

from forest.benchmarking import distance_measures as dm

# Single qubit defs
PROJ_ZERO = np.array([[1, 0], [0, 0]])
PROJ_ONE = np.array([[0, 0], [0, 1]])
ID = PROJ_ZERO + PROJ_ONE
PLUS = np.array([[1], [1]]) / np.sqrt(2)
PROJ_PLUS = PLUS @ PLUS.T.conj()
PROJ_MINUS = ID - PROJ_PLUS
Z_EFFECTS = [PROJ_ZERO, PROJ_ONE]
X_EFFECTS = [PROJ_PLUS, PROJ_MINUS]
X = PROJ_PLUS - PROJ_MINUS
Z = PROJ_ZERO - PROJ_ONE

# Two qubit defs
P00 = np.kron(PROJ_ZERO, PROJ_ZERO)
P01 = np.kron(PROJ_ZERO, PROJ_ONE)
P10 = np.kron(PROJ_ONE, PROJ_ZERO)
P11 = np.kron(PROJ_ONE, PROJ_ONE)
ID_2Q = P00 + P01 + P10 + P11
ZZ_EFFECTS = [P00, P01, P10, P11]


def test_generate_1q_state_tomography_experiment():
    qubits = [0]
    prog = Program(I(qubits[0]))
    one_q_exp = generate_state_tomography_experiment(prog, qubits=qubits)
    dimension = 2 ** len(qubits)

    assert [one_q_exp[idx][0].out_operator[qubits[0]] for idx in range(0, dimension ** 2)] == \
           ['I', 'X', 'Y', 'Z']


def test_generate_2q_state_tomography_experiment():
    p = Program()
    p += H(0)
    p += CZ(0, 1)
    two_q_exp = generate_state_tomography_experiment(p, qubits=[0, 1])
    dimension = 2 ** 2

    assert [str(two_q_exp[idx][0].out_operator) for idx in list(range(0, dimension ** 2))] == \
           ['(1+0j)*I', '(1+0j)*X1', '(1+0j)*Y1', '(1+0j)*Z1',
            '(1+0j)*X0', '(1+0j)*X0*X1', '(1+0j)*X0*Y1', '(1+0j)*X0*Z1',
            '(1+0j)*Y0', '(1+0j)*Y0*X1', '(1+0j)*Y0*Y1', '(1+0j)*Y0*Z1',
            '(1+0j)*Z0', '(1+0j)*Z0*X1', '(1+0j)*Z0*Y1', '(1+0j)*Z0*Z1']


def test_R_operator_fixed_point_1_qubit():
    # Check fixed point of operator. See Eq. 5 in Řeháček et al., PRA 75, 042108 (2007).
    obs_freqs = [1, 0]

    def test_trace(rho, effects):
        return np.trace(_R(rho, effects, obs_freqs) @ rho @ _R(rho, effects, obs_freqs) - rho)

    np.testing.assert_allclose(test_trace(PROJ_ZERO, Z_EFFECTS), 0.0, atol=1e-12)
    np.testing.assert_allclose(test_trace(PROJ_PLUS, X_EFFECTS), 0.0, atol=1e-12)


def test_R_operator_with_hand_calc_example_1_qubit():
    # This example was worked out by hand
    rho = ID / 2
    obs_freqs = [3, 7]
    my_by_hand_calc_ans_Z = ((3 / 0.5) * PROJ_ZERO + (7 / 0.5) * PROJ_ONE) / np.sum(obs_freqs)
    my_by_hand_calc_ans_X = ((3 / 0.5) * PROJ_PLUS + (7 / 0.5) * PROJ_MINUS) / np.sum(obs_freqs)

    # Z basis test
    assert np.trace(_R(rho, Z_EFFECTS, obs_freqs / np.sum(obs_freqs)) - my_by_hand_calc_ans_Z) == 0
    # X basis test
    assert np.trace(_R(rho, X_EFFECTS, obs_freqs / np.sum(obs_freqs)) - my_by_hand_calc_ans_X) == 0


def test_R_operator_fixed_point_2_qubit():
    # Check fixed point of operator. See Eq. 5 in Řeháček et al., PRA 75, 042108 (2007).
    obs_freqs = [1, 0, 0, 0]
    # Z basis test
    actual = np.trace(_R(P00, ZZ_EFFECTS, obs_freqs) @ P00 @ _R(P00, ZZ_EFFECTS, obs_freqs) - P00)
    np.testing.assert_allclose(actual, 0.0, atol=1e-12)


def get_test_qc(n_qubits):
    class BasicQVMCompiler(AbstractCompiler):
        def quil_to_native_quil(self, program: Program):
            return basic_compile(program)

        def native_quil_to_executable(self, nq_program: Program):
            return PyQuilExecutableResponse(
                program=nq_program.out(),
                attributes=_extract_attribute_dictionary_from_program(nq_program))

    try:
        qc = QuantumComputer(
            name='testing-qc',
            qam=QVM(connection=ForestConnection(), random_seed=52),
            device=NxDevice(nx.complete_graph(n_qubits)),
            compiler=BasicQVMCompiler(),
        )
        qc.run_and_measure(Program(I(0)), trials=1)
        return qc
    except (RequestException, TimeoutError) as e:
        return pytest.skip("This test requires a running local QVM: {}".format(e))


@pytest.fixture(scope='module')
def single_q_tomo_fixture():
    qubits = [0]
    qc = get_test_qc(n_qubits=len(qubits))

    # Generate random unitary
    u_rand = haar_rand_unitary(2 ** 1, rs=np.random.RandomState(52))
    state_prep = Program().defgate("RandUnitary", u_rand)
    state_prep.inst([("RandUnitary", qubits[0])])

    # True state
    wfn = NumpyWavefunctionSimulator(n_qubits=1)
    psi = wfn.do_gate_matrix(u_rand, qubits=[0]).wf.reshape(-1)
    rho_true = np.outer(psi, psi.T.conj())

    # Get data from QVM
    tomo_expt = generate_state_tomography_experiment(state_prep, qubits)
    results = list(measure_observables(qc=qc, tomo_experiment=tomo_expt, n_shots=4000))

    return results, rho_true


def test_single_qubit_linear_inv(single_q_tomo_fixture):
    qubits = [0]
    results, rho_true = single_q_tomo_fixture
    rho_est = linear_inv_state_estimate(results, qubits)
    np.testing.assert_allclose(rho_true, rho_est, atol=1e-2)


@pytest.fixture(scope='module')
def two_q_tomo_fixture():
    qubits = [0, 1]
    qc = get_test_qc(n_qubits=len(qubits))

    # Generate random unitary
    u_rand = haar_rand_unitary(2 ** 1, rs=np.random.RandomState(52))
    state_prep = Program().defgate("RandUnitary", u_rand)
    state_prep.inst([("RandUnitary", q) for q in qubits])

    # True state
    wfn = NumpyWavefunctionSimulator(n_qubits=2)
    psi = wfn \
        .do_gate_matrix(u_rand, qubits=[0]) \
        .do_gate_matrix(u_rand, qubits=[1]) \
        .wf.reshape(-1)
    rho_true = np.outer(psi, psi.T.conj())

    # Get data from QVM
    tomo_expt = generate_state_tomography_experiment(state_prep, qubits)
    results = list(measure_observables(qc=qc, tomo_experiment=tomo_expt, n_shots=4000))

    return results, rho_true


def test_two_qubit_linear_inv(two_q_tomo_fixture):
    qubits = [0, 1]
    results, rho_true = two_q_tomo_fixture
    rho_est = linear_inv_state_estimate(results, qubits)
    np.testing.assert_allclose(rho_true, rho_est, atol=0.01)


def test_single_qubit_mle(single_q_tomo_fixture):
    qubits = [0]
    results, rho_true = single_q_tomo_fixture
    estimate, status = iterative_mle_state_estimate(results=results, qubits=qubits, dilution=0.5)
    rho_est = estimate.estimate.state_point_est
    np.testing.assert_allclose(rho_true, rho_est, atol=0.01)


def test_two_qubit_mle(two_q_tomo_fixture):
    qubits = [0, 1]
    results, rho_true = two_q_tomo_fixture
    estimate, status = iterative_mle_state_estimate(results=results, qubits=qubits, dilution=0.5)
    rho_est = estimate.estimate.state_point_est
    np.testing.assert_allclose(rho_true, rho_est, atol=0.01)


def test_maxent_single_qubit(single_q_tomo_fixture):
    qubits = [0]
    results, rho_true = single_q_tomo_fixture
    estimate, status = iterative_mle_state_estimate(results=results, qubits=qubits,
                                                    dilution=0.5, entropy_penalty=1.0)
    rho_est = estimate.estimate.state_point_est
    np.testing.assert_allclose(rho_true, rho_est, atol=0.01)


def test_maxent_two_qubit(two_q_tomo_fixture):
    qubits = [0, 1]
    results, rho_true = two_q_tomo_fixture
    estimate, status = iterative_mle_state_estimate(results=results, qubits=qubits,
                                                    dilution=0.5, entropy_penalty=1.0, tol=1e-5)
    rho_est = estimate.estimate.state_point_est
    np.testing.assert_allclose(rho_true, rho_est, atol=0.01)


def test_hedged_single_qubit(single_q_tomo_fixture):
    qubits = [0]
    results, rho_true = single_q_tomo_fixture
    estimate, status = iterative_mle_state_estimate(results=results, qubits=qubits,
                                                    dilution=0.5, beta=0.5)
    rho_est = estimate.estimate.state_point_est
    np.testing.assert_allclose(rho_true, rho_est, atol=0.01)


def test_hedged_two_qubit(two_q_tomo_fixture):
    qubits = [0, 1]
    results, rho_true = two_q_tomo_fixture
    estimate, status = iterative_mle_state_estimate(results=results, qubits=qubits,
                                                    dilution=0.5, beta=0.5)
    rho_est = estimate.estimate.state_point_est
    np.testing.assert_allclose(rho_true, rho_est, atol=0.01)


def test_project_density_matrix():
    """
    Test the wizard method. Example from fig 1 of maximum likelihood minimum effort
    https://doi.org/10.1103/PhysRevLett.108.070502

    :return:
    """
    eigs = np.diag(np.array(list(reversed([3.0 / 5, 1.0 / 2, 7.0 / 20, 1.0 / 10, -11.0 / 20]))))
    phys = project_density_matrix(eigs)
    assert np.allclose(phys, np.diag([0, 0, 1.0 / 5, 7.0 / 20, 9.0 / 20]))


def test_variance_bootstrap():
    qubits = [0, 1]
    qc = get_test_qc(n_qubits=len(qubits))
    state_prep = Program([H(q) for q in qubits])
    state_prep.inst(CZ(qubits[0], qubits[1]))
    tomo_expt = generate_state_tomography_experiment(state_prep, qubits)
    results = list(measure_observables(qc=qc, tomo_experiment=tomo_expt, n_shots=4000))
    estimate, status = iterative_mle_state_estimate(results=results, qubits=qubits,
                                                    dilution=0.5)
    rho_est = estimate.estimate.state_point_est
    purity = np.trace(rho_est @ rho_est)
    purity = np.real_if_close(purity)
    assert purity.imag == 0.0

    def my_mle_estimator(_r, _q):
        return iterative_mle_state_estimate(results=_r, qubits=_q,
                                            dilution=0.5, entropy_penalty=0.0, beta=0.0)[0]

    boot_purity, boot_var = estimate_variance(results=results, qubits=qubits,
                                              tomo_estimator=my_mle_estimator, functional=dm.purity,
                                              n_resamples=5, project_to_physical=False)

    np.testing.assert_allclose(purity, boot_purity, atol=2 * np.sqrt(boot_var), rtol=0.01)
