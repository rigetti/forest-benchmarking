import numpy as np
from forest.benchmarking.tests.test_superoperator_transformations import (
    amplitude_damping_kraus, amplitude_damping_choi, ONE_STATE, ZERO_STATE, rho_out)
from forest.benchmarking.operator_tools.apply_superoperator import (apply_kraus_ops_2_state,
                                                                    apply_choi_matrix_2_state)


def test_apply_kraus_ops_2_state():
    AD_kraus = amplitude_damping_kraus(0.1)
    # rho_out was calculated by hand
    assert np.allclose(rho_out, apply_kraus_ops_2_state(AD_kraus, ONE_STATE))


def test_apply_non_square_kraus_ops_2_state():
    Id = np.eye(2)
    bra_zero = np.asarray([[1], [0]])
    bra_one = np.asarray([[0], [1]])
    state_one = np.kron(Id / 2, ONE_STATE)
    state_zero = np.kron(Id / 2, ZERO_STATE)
    Kraus1 = np.kron(Id, bra_one.transpose())
    Kraus0 = np.kron(Id, bra_zero.transpose())
    assert np.allclose(apply_kraus_ops_2_state(Kraus0, state_zero), Id / 2)
    assert np.allclose(apply_kraus_ops_2_state(Kraus1, state_one), Id / 2)
    assert np.allclose(apply_kraus_ops_2_state(Kraus0, state_one), 0)


def test_apply_choi_matrix_2_state():
    choi = amplitude_damping_choi(0.1)
    assert np.allclose(rho_out, apply_choi_matrix_2_state(choi, ONE_STATE))


def test_apply_kraus_ops_2_state_complex():
    Id = np.eye(2)
    H = np.asarray([[1, 1], [1, -1]]) / np.sqrt(2)
    S = np.asarray([[1, 0], [0, 1j]])
    state_plus = np.asarray([[0.5, 0.5], [0.5, 0.5]])
    state_plus_i = np.asarray([[0.5, -0.5j], [0.5j, 0.5]])

    # Real Kraus with complex state (phase coherence preserved)
    real_kraus = [0.8 * H, 0.6 * Id]
    expected_real_kraus = np.asarray([[0.5, 0.14j], [-0.14j, 0.5]])
    assert np.allclose(expected_real_kraus, apply_kraus_ops_2_state(real_kraus, state_plus_i))

    # Complex Kraus with real state
    complex_kraus = [0.8 * S, 0.6 * S.conj().T]
    expected_complex_kraus = np.asarray([[0.5, -0.14j], [0.14j, 0.5]])
    assert np.allclose(expected_complex_kraus, apply_kraus_ops_2_state(complex_kraus, state_plus))

    # Complex Kraus with complex state
    parent_kraus = [0.8 * (S @ H), 0.6 * (H @ S.conj().T)]
    expected_parent = np.asarray([[0.68, 0.32], [0.32, 0.32]])
    assert np.allclose(expected_parent, apply_kraus_ops_2_state(parent_kraus, state_plus_i))

    # Non-square complex Kraus preserves output shape
    kraus_non_square = np.kron(Id, np.asarray([[1, -1j]]) / np.sqrt(2))
    state_4x4 = np.kron(Id / 2, state_plus_i)
    res_non_square = apply_kraus_ops_2_state(kraus_non_square, state_4x4)
    assert res_non_square.shape == (2, 2)
    assert np.allclose(Id / 2, res_non_square)
