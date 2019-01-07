import numpy as np
from forest_qcvv.tomography import generate_state_tomography_experiment, _R, \
    acquire_tomography_data, iterative_mle_state_estimate, project_density_matrix, \
    estimate_variance, linear_inv_state_estimate
from forest_qcvv import distance_measures as dm

import forest_qcvv.random_operators as rand_ops

from pyquil.quil import Program
from pyquil.gates import I, H, CZ

np.random.seed(7)  # seed random number generation for all calls to rand_ops
# Single qubit defs
P0 = np.array([[1, 0], [0, 0]])
P1 = np.array([[0, 0], [0, 1]])
Id = P0 + P1
plus = np.array([[1], [1]]) / np.sqrt(2)
Pp = np.matmul(plus, np.matrix.getH(plus))
Pm = Id - Pp
effectsZ = [P0, P1]
effectsX = [Pp, Pm]
X = Pp - Pm
Y = 1j * Pp - 1j * Pm
Z = P0 - P1
# Two qubit defs
P00 = np.kron(P0, P0)
P01 = np.kron(P0, P1)
P10 = np.kron(P1, P0)
P11 = np.kron(P1, P1)
Id2 = P00 + P01 + P10 + P11
effectsZZ = [P00, P01, P10, P11]

tol = .08
var = .005
Urand = rand_ops.haar_rand_unitary(2)


def test_generate_1q_state_tomography_experiment():
    prog = Program(I(0))
    one_q_exp = generate_state_tomography_experiment(prog)
    qubits = prog.get_qubits()
    n_qubits = len(qubits)
    dimension = 2 ** n_qubits
    assert [one_q_exp.out_ops[idx][0] for idx in list(range(0, dimension ** 2 - 1))] == \
           ['X', 'Y', 'Z']


def test_generate_2q_state_tomography_experiment():
    p = Program()
    p.inst(H(0))
    prep_prog = p.inst(CZ(0, 1))
    two_q_exp = generate_state_tomography_experiment(prep_prog)
    qubits = prep_prog.get_qubits()
    n_qubits = len(qubits)
    dimension = 2 ** n_qubits
    assert [str(two_q_exp.out_ops[idx]) for idx in list(range(0, dimension ** 2 - 1))] == \
           ['(1+0j)*X0', '(1+0j)*Y0',
            '(1+0j)*Z0', '(1+0j)*X1',
            '(1+0j)*X1*X0',
            '(1+0j)*X1*Y0',
            '(1+0j)*X1*Z0',
            '(1+0j)*Y1',
            '(1+0j)*Y1*X0',
            '(1+0j)*Y1*Y0',
            '(1+0j)*Y1*Z0',
            '(1+0j)*Z1',
            '(1+0j)*Z1*X0',
            '(1+0j)*Z1*Y0',
            '(1+0j)*Z1*Z0']


def test_R_operator_fixed_point_1_qubit():
    # Check fixed point of operator. See Eq. 5 in Řeháček et al., PRA 75, 042108 (2007).
    rho0 = P0
    rhop = Pp
    freqs = [1, 0]
    # Z basis test
    assert np.trace(_R(rho0, effectsZ, freqs).dot(rho0).dot(_R(rho0, effectsZ, freqs)) - rho0) == 0
    # X basis test (now we run into the problem of adding "machine_eps" in _R() def)
    assert np.trace(
        _R(rhop, effectsX, freqs).dot(rhop).dot(_R(rhop, effectsX, freqs)) - rhop) < 1e-12
    # TODO: talk to Anthony and Matt about this^^
    return


def test_R_operator_with_hand_calc_example_1_qubit():
    # This example was worked out by hand
    rho0 = Id / 2
    freqs = [3, 7]
    my_by_hand_calc_ans_Z = ((3 / 0.5) * P0 + (7 / 0.5) * P1) / np.sum(freqs)
    my_by_hand_calc_ans_X = ((3 / 0.5) * Pp + (7 / 0.5) * Pm) / np.sum(freqs)
    # Z basis test
    assert np.trace(_R(rho0, effectsZ, freqs / np.sum(freqs)) - my_by_hand_calc_ans_Z) == 0
    # X basis test
    assert np.trace(_R(rho0, effectsX, freqs / np.sum(freqs)) - my_by_hand_calc_ans_X) == 0
    return


def test_R_operator_fixed_point_2_qubit():
    # Check fixed point of operator. See Eq. 5 in Řeháček et al., PRA 75, 042108 (2007).
    rho00 = P00
    freqs = [1, 0, 0, 0]
    # Z basis test
    assert np.trace(_R(rho00, effectsZZ, freqs / np.sum(freqs)).dot(rho00).dot(_R(rho00, effectsZZ, freqs)) - rho00) == 0
    return


def test_single_qubit_linear_inv(qvm, wfn):
    qvm.qam.random_seed = 1
    # Single qubit test
    qubits = [0]

    # Generate random unitary
    state_prep = Program().defgate("RandUnitary", Urand)
    state_prep.inst([("RandUnitary", q) for q in qubits])

    # True state
    psi = wfn.wavefunction(state_prep)
    rho_true = np.outer(psi.amplitudes, np.transpose(np.conj(psi.amplitudes)))
    tomo_progs = generate_state_tomography_experiment(state_prep)

    # Get data from QVM then estimate state
    exp_data = acquire_tomography_data(tomo_progs, qvm, var)

    # check that input program is not mutated
    assert state_prep == exp_data.program

    estimate = linear_inv_state_estimate(exp_data)

    # Compute the Frobeius norm of the different between the estimated operator and the answer
    assert np.real(np.linalg.norm((rho_true - estimate.estimate.state_point_est), 'fro')) <= tol
    assert np.real(np.linalg.norm((rho_true - estimate.estimate.state_point_est), 'fro')) >= 0.00


def test_two_qubit_linear_inv(qvm, wfn):
    qvm.qam.random_seed = 1
    # Two qubit test
    qubits = [0, 1]

    # Generate random unitary
    state_prep = Program().defgate("RandUnitary", Urand)
    state_prep.inst([("RandUnitary", q) for q in qubits])

    # True state
    psi = wfn.wavefunction(state_prep)
    rho_true = np.outer(psi.amplitudes, np.transpose(np.conj(psi.amplitudes)))
    tomo_progs = generate_state_tomography_experiment(state_prep)

    # Get data from QVM then estimate state
    exp_data = acquire_tomography_data(tomo_progs, qvm, .0005)
    estimate = linear_inv_state_estimate(exp_data)

    # Compute the Frobeius norm of the different between the estimated operator and the answer
    assert np.real(np.linalg.norm((rho_true - estimate.estimate.state_point_est), 'fro')) <= tol
    assert np.real(np.linalg.norm((rho_true - estimate.estimate.state_point_est), 'fro')) >= 0.00


def test_single_qubit_mle(qvm, wfn):
    qvm.qam.random_seed = 1
    # Single qubit test
    qubits = [0]

    # Generate random unitary
    state_prep = Program().defgate("RandUnitary", Urand)
    state_prep.inst([("RandUnitary", q) for q in qubits])

    # True state
    psi = wfn.wavefunction(state_prep)
    rho_true = np.outer(psi.amplitudes, np.transpose(np.conj(psi.amplitudes)))
    tomo_progs = generate_state_tomography_experiment(state_prep)

    # Get data from QVM then estimate state
    exp_data = acquire_tomography_data(tomo_progs, qvm, var)

    estimate, status = iterative_mle_state_estimate(exp_data, dilution=0.5)

    # Compute the Frobeius norm of the different between the estimated operator and the answer
    assert np.real(np.linalg.norm((rho_true - estimate.estimate.state_point_est), 'fro')) <= tol
    assert np.real(np.linalg.norm((rho_true - estimate.estimate.state_point_est), 'fro')) >= 0.00


def test_two_qubit_mle(qvm, wfn):
    qvm.qam.random_seed = 1
    # Two qubit test
    qubits = [0, 1]

    # Generate random unitary
    state_prep = Program().defgate("RandUnitary", Urand)
    state_prep.inst([("RandUnitary", q) for q in qubits])

    # True state
    psi = wfn.wavefunction(state_prep)
    rho_true = np.outer(psi.amplitudes, np.transpose(np.conj(psi.amplitudes)))
    tomo_progs = generate_state_tomography_experiment(state_prep)

    # Get data from QVM then estimate state
    exp_data = acquire_tomography_data(tomo_progs, qvm, var)
    estimate, status = iterative_mle_state_estimate(exp_data, dilution=0.5)

    # Compute the Frobeius norm of the different between the estimated operator and the answer
    assert np.real(np.linalg.norm((rho_true - estimate.estimate.state_point_est), 'fro')) <= tol
    assert np.real(np.linalg.norm((rho_true - estimate.estimate.state_point_est), 'fro')) >= 0.00


def test_MaxEnt_single_qubit(qvm, wfn):
    qvm.qam.random_seed = 1
    # Single qubit test
    qubits = [0]

    # Generate random unitary
    state_prep = Program().defgate("RandUnitary", Urand)
    state_prep.inst([("RandUnitary", q) for q in qubits])

    # True state
    psi = wfn.wavefunction(state_prep)
    rho_true = np.outer(psi.amplitudes, np.transpose(np.conj(psi.amplitudes)))
    tomo_progs = generate_state_tomography_experiment(state_prep)

    # Get data from QVM then estimate state
    exp_data = acquire_tomography_data(tomo_progs, qvm, var)

    estimate, status = iterative_mle_state_estimate(exp_data, dilution=0.5, entropy_penalty=1.0)

    # Compute the Frobeius norm of the different between the estimated operator and the answer
    assert np.real(np.linalg.norm((rho_true - estimate.estimate.state_point_est), 'fro')) <= tol
    assert np.real(np.linalg.norm((rho_true - estimate.estimate.state_point_est), 'fro')) >= 0.00


def test_MaxEnt_two_qubit(qvm, wfn):
    qvm.qam.random_seed = 1
    # Two qubit test
    qubits = [0, 1]

    # Generate random unitary
    state_prep = Program().defgate("RandUnitary", Urand)
    state_prep.inst([("RandUnitary", q) for q in qubits])

    # True state
    psi = wfn.wavefunction(state_prep)
    rho_true = np.outer(psi.amplitudes, np.transpose(np.conj(psi.amplitudes)))
    tomo_progs = generate_state_tomography_experiment(state_prep)

    # Get data from QVM then estimate state
    exp_data = acquire_tomography_data(tomo_progs, qvm, var)
    estimate, status = iterative_mle_state_estimate(exp_data, dilution=0.5, entropy_penalty=1.0,
                                                    tol=.001)

    # Compute the Frobeius norm of the different between the estimated operator and the answer
    assert np.real(np.linalg.norm((rho_true - estimate.estimate.state_point_est), 'fro')) <= tol
    assert np.real(np.linalg.norm((rho_true - estimate.estimate.state_point_est), 'fro')) >= 0.00


def test_hedged_single_qubit(qvm, wfn):
    qvm.qam.random_seed = 1
    # Single qubit test
    qubits = [0]

    # Generate random unitary
    state_prep = Program().defgate("RandUnitary", Urand)
    state_prep.inst([("RandUnitary", q) for q in qubits])

    # True state
    psi = wfn.wavefunction(state_prep)
    rho_true = np.outer(psi.amplitudes, np.transpose(np.conj(psi.amplitudes)))
    tomo_progs = generate_state_tomography_experiment(state_prep)

    # Get data from QVM then estimate state
    exp_data = acquire_tomography_data(tomo_progs, qvm, var)

    estimate, status = iterative_mle_state_estimate(exp_data, dilution=0.5, beta=0.5)

    # Compute the Frobeius norm of the different between the estimated operator and the answer
    assert np.real(np.linalg.norm((rho_true - estimate.estimate.state_point_est), 'fro')) <= tol
    assert np.real(np.linalg.norm((rho_true - estimate.estimate.state_point_est), 'fro')) >= 0.00


def test_hedged_two_qubit(qvm, wfn):
    qvm.qam.random_seed = 1
    # Two qubit test
    qubits = [0, 1]

    # Generate random unitary
    state_prep = Program().defgate("RandUnitary", Urand)
    state_prep.inst([("RandUnitary", q) for q in qubits])

    # True state
    psi = wfn.wavefunction(state_prep)
    rho_true = np.outer(psi.amplitudes, np.transpose(np.conj(psi.amplitudes)))
    tomo_progs = generate_state_tomography_experiment(state_prep)

    # Get data from QVM then estimate state
    exp_data = acquire_tomography_data(tomo_progs, qvm, var)
    estimate, status = iterative_mle_state_estimate(exp_data, dilution=0.5, beta=0.5)

    # Compute the Frobeius norm of the different between the estimated operator and the answer
    assert np.real(np.linalg.norm((rho_true - estimate.estimate.state_point_est), 'fro')) <= tol
    assert np.real(np.linalg.norm((rho_true - estimate.estimate.state_point_est), 'fro')) >= 0.00


def test_project_density_matrix():
    """
    Test the wizard method. Example from fig 1 of maximum likelihood minimum effort
    https://doi.org/10.1103/PhysRevLett.108.070502

    :return:
    """
    eigs = np.diag(np.array(list(reversed([3.0 / 5, 1.0 / 2, 7.0 / 20, 1.0 / 10, -11.0 / 20]))))
    phys = project_density_matrix(eigs)
    assert np.allclose(phys, np.diag([0, 0, 1.0 / 5, 7.0 / 20, 9.0 / 20]))


def test_variance_bootstrap(qvm):
    qvm.qam.random_seed = 1

    qubits = [0, 1]
    state_prep = Program([H(q) for q in qubits])
    state_prep.inst(CZ(0, 1))
    exp_desc = generate_state_tomography_experiment(state_prep)
    exp_data = acquire_tomography_data(exp_desc, qvm, var=.1)
    estimate_mle, status = iterative_mle_state_estimate(exp_data, dilution=0.5)
    purity = np.trace(
        np.matmul(estimate_mle.estimate.state_point_est, estimate_mle.estimate.state_point_est))

    def my_mle_estimator(data):
        return iterative_mle_state_estimate(data, dilution=0.5, entropy_penalty=0.0, beta=0.0)[0]

    boot_purity, boot_var = estimate_variance(exp_data, my_mle_estimator, dm.purity, n_resamples=5,
                                              project_to_physical=False)

    assert np.isclose(np.real_if_close(purity), boot_purity, atol=2 * np.sqrt(boot_var), rtol=.01)
