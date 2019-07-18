import numpy as np
from pyquil.gate_matrices import I
from forest.benchmarking.tests.test_superoperator_transformations import (amplitude_damping_choi,
                                                                          ONE_STATE,
                                                                          rho_out)
from forest.benchmarking.operator_tools.apply_superoperator import apply_choi_matrix_2_state
from forest.benchmarking.operator_tools.calculational import (partial_trace,
                                                              outer_product,
                                                              inner_product,
                                                              sqrtm_psd)


def test_partial_trace():
    rho = np.kron(I, I) / 4
    np.testing.assert_array_equal(I / 2, partial_trace(rho, [1], [2, 2]))
    np.testing.assert_array_equal(I / 2, partial_trace(rho, [0], [2, 2]))
    # test using a real world example as apply_choi_matrix_2_state calls partial trace.
    choi = amplitude_damping_choi(0.1)
    assert np.allclose(rho_out, apply_choi_matrix_2_state(choi, ONE_STATE))


def test_outer_product():
    ans = np.array([[0. + 0.j, 0. - 1.j, 0. + 0.j],
                    [0. + 0.j, 0. + 0.j, 0. + 0.j],
                    [0. + 0.j, 0. + 0.j, 0. + 0.j]])
    tst = outer_product(np.array([[1], [0], [0]]), np.array([[0], [1j], [0]]))
    np.testing.assert_allclose(ans, tst)


def test_inner_product():
    v1 = np.array([[1], [0], [0]])
    v2 = np.array([[0], [1], [0]])
    v3 = np.array([[0], [1j], [0]])
    # equal
    assert np.isclose(inner_product(v1, v1), 1)
    assert np.isclose(inner_product(v2, v2), 1)
    # orthogonal
    assert np.isclose(inner_product(v2, v1), 0)
    assert np.isclose(inner_product(v2, v1), 0)
    # complex
    assert np.isclose(inner_product(v2, v3), 1j)
    assert np.isclose(inner_product(v3, v2), -1j)


def test_sqrtm_psd():
    # test obvious case
    A = np.diag([2, 1, 0])
    Asqrt = np.diag([np.sqrt(2), 1, 0])
    np.allclose(sqrtm_psd(A), Asqrt)
    # sqrt zero = zero
    A = np.zeros((2, 2))
    np.allclose(sqrtm_psd(A), A)
    # sqrt(A) * sqrt(A) =  A
    A = np.array([[1, 2], [2, 4]])
    np.allclose(sqrtm_psd(A) @ sqrtm_psd(A), A)