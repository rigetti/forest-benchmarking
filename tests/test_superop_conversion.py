import numpy as np
from forest_benchmarking.utils import *
from forest_benchmarking.superop_conversion import *

# one and zero state as a density matrix
ONE_STATE = np.asarray([[0, 0], [0, 1]])
ZERO_STATE = np.asarray([[1, 0], [0, 0]])
ONE_STATE_VEC = vec(ONE_STATE)
ZERO_STATE_VEC = vec(ZERO_STATE)

HADAMARD = (sigma_x + sigma_z) / np.sqrt(2)
id_choi = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])

# Amplitude damping Kraus operators with p = 0.1
Ad0 = np.asarray([[1, 0], [0, np.sqrt(1 - 0.1)]])
Ad1 = np.asarray([[0, np.sqrt(0.1)], [0, 0]])
AdKrausOps = (Ad0, Ad1)

# Dephasing Channel
deph_choi = np.array([[1, 0, 0, 0.95], [0, 0, 0, 0], [0, 0, 0, 0], [0.95, 0, 0, 1]])
deph_superop = np.array([[1, 0, 0, 0], [0, 0.95, 0, 0], [0, 0, 0.95, 0], [0, 0, 0, 1]])
# kraus
K0 = np.sqrt(.95) * np.eye(2)
K1 = np.sqrt(.05) * np.array([[1., 0], [0, 0]])
K2 = np.sqrt(.05) * np.array([[0, 0], [0, 1.]])

# Use Kraus operators to find out put of channel i.e.
#    rho_out = A_0 rho A_0^\dag + A_1 rho A_1^\dag.
# Except we only transpose as these matrices have real entries.
rho_out = np.matmul(np.matmul(Ad0, ONE_STATE), Ad0.transpose()) + \
          np.matmul(np.matmul(Ad1, ONE_STATE), Ad1.transpose())


def test_vec():
    A = np.asarray([[1, 2], [3, 4]])
    B = np.asarray([[1, 2, 5], [3, 4, 6]])
    np.testing.assert_array_equal(np.array([[1], [3], [2], [4]]), vec(A))
    np.testing.assert_array_equal(np.array([[1], [3], [2], [4], [5], [6]]), vec(B))


def test_unvec():
    A = np.asarray([[1, 2], [3, 4]])
    C = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    np.testing.assert_array_equal(A, unvec(vec(A)))
    np.testing.assert_array_equal(C, unvec(vec(C)))


def test_kraus_ops_sum_to_identity():
    # Check kraus ops sum to identity
    np.testing.assert_array_almost_equal_nulp(np.matmul(Ad0.transpose(), Ad0)
                                              + np.matmul(Ad1.transpose(), Ad1), np.eye(2))


def test_kraus2superop():
    np.testing.assert_array_almost_equal_nulp(unvec(np.matmul(kraus2superop(AdKrausOps),
                                                              ONE_STATE_VEC)), rho_out)

    assert np.allclose(kraus2superop([K0, K1, K2]), deph_superop)


def test_kraus2choi():
    choi = kraus2choi(AdKrausOps)
    input_state = np.asarray([[0, 0], [0, 1]])
    prody = np.matmul(choi, np.kron(input_state.transpose(), np.eye(2)))
    np.testing.assert_array_almost_equal_nulp(partial_trace(prody, [1], [2, 2]), rho_out)

    assert np.allclose(kraus2choi([K0, K1, K2]), deph_choi)


def test_kraus2pauli_liouville():
    hadamard = (sigma_x + sigma_z) / np.sqrt(2)
    assert np.allclose(kraus2pauli_liouville(hadamard), [[1., 0, 0, 0], [0, 0, 0, 1.], [0, 0, -1., 0], [0, 1., 0, 0]])


def test_basis_transform_p_to_c():
    xz_pauli_basis = np.zeros((16, 1))
    xz_pauli_basis[7] = [1.]
    assert np.allclose(unvec(pauli2computational_basis_matrix(4) @ xz_pauli_basis), np.kron(sigma_x, sigma_z))


def test_basis_transform_c_to_p():
    xz_pauli_basis = np.zeros((16, 1))
    xz_pauli_basis[7] = [1.]
    assert np.allclose(computational2pauli_basis_matrix(4) @ vec(np.kron(sigma_x, sigma_z)), xz_pauli_basis)


def test_pl_to_choi():
    for i, pauli in enumerate(n_qubit_pauli_basis(2)):
        pl = kraus2pauli_liouville(pauli[1])
        choi = kraus2choi(pauli[1])
        assert np.allclose(choi, pauli_liouville2choi(pl))

    pl = kraus2pauli_liouville(HADAMARD)
    choi = kraus2choi(HADAMARD)
    assert np.allclose(choi, pauli_liouville2choi(pl))


def test_superop_to_choi():
    for i, pauli in enumerate(n_qubit_pauli_basis(2)):
        superop = kraus2superop(pauli[1])
        choi = kraus2choi(pauli[1])
        assert np.allclose(choi, superop2choi(superop))

    superop = kraus2superop(HADAMARD)
    choi = kraus2choi(HADAMARD)
    assert np.allclose(choi, superop2choi(superop))


def test_choi_to_kraus():
    for i, pauli in enumerate(n_qubit_pauli_basis(2)):
        choi = kraus2choi(pauli[1])
        kraus = choi2kraus(choi)
        assert np.allclose(choi, kraus2choi(kraus))
    assert np.allclose(kraus2choi(choi2kraus(id_choi)), id_choi)
    for kraus in choi2kraus(id_choi):
        assert np.allclose(abs(kraus), np.eye(2)) or np.allclose(kraus, np.zeros((2, 2)))


def test_choi_pl_bijectivity():
    assert np.allclose(choi2superop(choi2superop(np.eye(4))), np.eye(4))
    assert np.allclose(superop2choi(superop2choi(np.eye(4))), np.eye(4))
    h_choi = kraus2choi(HADAMARD)
    h_superop = kraus2superop(HADAMARD)
    assert np.allclose(choi2superop(choi2superop(h_choi)), h_choi)
    assert np.allclose(superop2choi(superop2choi(h_superop)), h_superop)
