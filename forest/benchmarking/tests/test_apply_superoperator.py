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
