from forest_benchmarking.tomography import proj_to_cp, proj_to_tni, \
    generate_process_tomography_experiment, acquire_tomography_data, pgdb_process_estimate, \
    proj_to_tp
from forest_benchmarking.tomography import _constraint_project

from forest_benchmarking.superop_conversion import vec, unvec, kraus2choi
from forest_benchmarking.utils import sigma_x, sigma_y, sigma_z, partial_trace
from numpy import pi
import numpy as np
from scipy.linalg import expm
from pyquil import Program
from pyquil.gates import *

REVERSE_CNOT_KRAUS = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
CNOT_KRAUS = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])


def test_proj_to_cp():
    state = vec(np.array([[1., 0], [0, 1.]]))
    assert np.allclose(state, proj_to_cp(state))

    state = vec(np.array([[1.5, 0], [0, 10]]))
    assert np.allclose(state, proj_to_cp(state))

    state = vec(np.array([[-1, 0], [0, 1.]]))
    cp_state = vec(np.array([[0, 0], [0, 1.]]))
    assert np.allclose(cp_state, proj_to_cp(state))

    state = vec(np.array([[0, 1], [1, 0]]))
    cp_state = vec(np.array([[.5, .5], [.5, .5]]))
    assert np.allclose(cp_state, proj_to_cp(state))

    state = vec(np.array([[0, -1j], [1j, 0]]))
    cp_state = vec(np.array([[.5, -.5j], [.5j, .5]]))
    assert np.allclose(cp_state, proj_to_cp(state))


def test_proj_to_tp():
    # Identity process is trace preserving, so no change
    state = vec(kraus2choi(np.eye(2)))
    assert np.allclose(state, proj_to_tp(state))

    # Bit flip process is trace preserving, so no change
    state = vec(kraus2choi(sigma_x))
    assert np.allclose(state, proj_to_tp(state))


def test_cptp():
    # Identity process is cptp, so no change
    state = np.array(kraus2choi(np.eye(2)))
    assert np.allclose(state, _constraint_project(state))

    # Small perturbation shouldn't change too much
    state = np.array([[1.001, 0., 0., .99], [0., 0., 0., 0.], [0., 0., 0., 0.], [1.004, 0., 0., 1.01]])
    assert np.allclose(state, _constraint_project(state), atol=.01)

    # Bit flip process is cptp, so no change
    state = kraus2choi(sigma_x)
    assert np.allclose(state, _constraint_project(state))


def test_proj_to_tni():
    state = np.array([[0., 0., 0., 0.], [0., 1.01, 1.01, 0.], [0., 1., 1., 0.], [0., 0., 0., 0.]])
    trace_non_increasing = unvec(proj_to_tni(vec(state)))
    pt = partial_trace(trace_non_increasing, dims=[2, 2], keep=[0])
    assert np.allclose(pt, np.eye(2))


def test_single_qubit_identity(qvm):
    qvm.qam.random_seed = 1
    process = Program(I(0))

    exp_desc = generate_process_tomography_experiment(process)
    exp_data = acquire_tomography_data(exp_desc, qvm)
    estimate = pgdb_process_estimate(exp_data)
    assert np.allclose(estimate.estimate.process_choi_est, kraus2choi(np.eye(2)), atol=.01)


def test_single_qubit_x(qvm):
    qvm.qam.random_seed = 1
    process = Program(RX(pi, 0))

    exp_desc = generate_process_tomography_experiment(process)
    exp_data = acquire_tomography_data(exp_desc, qvm)
    estimate = pgdb_process_estimate(exp_data)
    assert np.allclose(estimate.estimate.process_choi_est, kraus2choi(sigma_x), atol=.01)


def test_single_qubit_y(qvm):
    qvm.qam.random_seed = 1
    process = Program(RY(pi, 0))

    exp_desc = generate_process_tomography_experiment(process)
    exp_data = acquire_tomography_data(exp_desc, qvm)
    estimate = pgdb_process_estimate(exp_data)
    assert np.allclose(estimate.estimate.process_choi_est, kraus2choi(sigma_y), atol=.01)


def test_single_qubit_z(qvm):
    qvm.qam.random_seed = 1
    process = Program(RZ(pi, 0))

    exp_desc = generate_process_tomography_experiment(process)
    exp_data = acquire_tomography_data(exp_desc, qvm)
    estimate = pgdb_process_estimate(exp_data)
    assert np.allclose(estimate.estimate.process_choi_est, kraus2choi(sigma_z), atol=.01)


def test_single_qubit_rx(qvm):
    qvm.qam.random_seed = 1
    process = Program(RX(pi / 2, 0))

    exp_desc = generate_process_tomography_experiment(process)
    exp_data = acquire_tomography_data(exp_desc, qvm, var=.005)
    estimate = pgdb_process_estimate(exp_data)
    assert np.allclose(estimate.estimate.process_choi_est, kraus2choi(expm(-1j * pi / 4 * sigma_x)), atol=.01)


def test_single_qubit_rx_rz(qvm):
    qvm.qam.random_seed = 1
    process = Program(RX(pi / 2, 0)).inst(RZ(1, 0))

    exp_desc = generate_process_tomography_experiment(process)
    exp_data = acquire_tomography_data(exp_desc, qvm, var=.005)
    estimate = pgdb_process_estimate(exp_data)
    rx = expm(-1j * pi / 4 * sigma_x)
    rz = expm(-1j / 2 * sigma_z)
    assert np.allclose(estimate.estimate.process_choi_est, kraus2choi(rz @ rx), atol=.01)


def test_two_qubit_identity(qvm):
    qvm.qam.random_seed = 2
    process = Program(I(1)).inst(I(3))

    exp_desc = generate_process_tomography_experiment(process)
    exp_data = acquire_tomography_data(exp_desc, qvm, var=.05)
    estimate = pgdb_process_estimate(exp_data)
    assert np.allclose(estimate.estimate.process_choi_est, kraus2choi(np.eye(4)), atol=.06)


def test_two_qubit_cnot(qvm):
    qvm.qam.random_seed = 2
    process = Program(CNOT(5, 3))

    exp_desc = generate_process_tomography_experiment(process)
    exp_data = acquire_tomography_data(exp_desc, qvm, var=.05)
    estimate = pgdb_process_estimate(exp_data)
    assert np.allclose(estimate.estimate.process_choi_est, kraus2choi(REVERSE_CNOT_KRAUS), atol=.05)


def test_two_qubit_cnot_rx_rz(qvm):
    qvm.qam.random_seed = 1
    process = Program(CNOT(0, 1)).inst(RX(pi / 2, 0)).inst(RZ(1, 1))

    exp_desc = generate_process_tomography_experiment(process)
    exp_data = acquire_tomography_data(exp_desc, qvm, var=.05)
    estimate = pgdb_process_estimate(exp_data)
    rx = np.kron(np.eye(2), expm(-1j * pi / 4 * sigma_x), )
    rz = np.kron(expm(-1j / 2 * sigma_z), np.eye(2))
    assert np.allclose(estimate.estimate.process_choi_est, kraus2choi(rz @ rx @ CNOT_KRAUS), atol=.05)
