import numpy as np
from forest.benchmarking.utils import *
from forest.benchmarking.superoperator_tools import *
import forest.benchmarking.random_operators as rand_ops


# Test philosophy:
# Using the by hand calculations found in the docs we check conversion
# between one qubit channels with one Kraus operator (Hadamard) and two
# Kraus operators (the amplitude damping channel). Additionally we check
# a few two qubit channel conversions to get additional confidence.

# The Amplitude Damping channel (one qubit)
def amplitude_damping_kraus(p):
    Ad0 = np.asarray([[1, 0], [0, np.sqrt(1 - p)]])
    Ad1 = np.asarray([[0, np.sqrt(p)], [0, 0]])
    return [Ad0, Ad1]


def amplitude_damping_chi(p):
    poly1 = (1 + np.sqrt(1 - p)) ** 2
    poly2 = (-1 + np.sqrt(1 - p)) ** 2
    ad_pro = 0.25 * np.asarray([[poly1, 0, 0, p],
                                [0, p, -1j * p, 0],
                                [0, 1j * p, p, 0],
                                [p, 0, 0, poly2]])
    return ad_pro


def amplitude_damping_pauli(p):
    poly1 = np.sqrt(1 - p)
    ad_pau = np.asarray([[1, 0, 0, 0],
                         [0, poly1, 0, 0],
                         [0, 0, poly1, 0],
                         [p, 0, 0, 1 - p]])
    return ad_pau


def amplitude_damping_super(p):
    poly1 = np.sqrt(1 - p)
    ad_sup = np.asarray([[1, 0, 0, p],
                         [0, poly1, 0, 0],
                         [0, 0, poly1, 0],
                         [0, 0, 0, 1 - p]])
    return ad_sup


def amplitude_damping_choi(p):
    poly1 = np.sqrt(1 - p)
    ad_choi = np.asarray([[1, 0, 0, poly1],
                          [0, 0, 0, 0],
                          [0, 0, p, 0],
                          [poly1, 0, 0, 1 - p]])
    return ad_choi


# The Hadamard channel or gate (one qubit)

HADAMARD = (sigma_x + sigma_z) / np.sqrt(2)
HADKraus = HADAMARD

HADChi = 0.5 * np.asarray([[0, 0, 0, 0],
                           [0, 1, 0, 1],
                           [0, 0, 0, 0],
                           [0, 1, 0, 1]])

HADPauli = 1.0 * np.asarray([[1, 0, 0, 0],
                             [0, 0, 0, 1],
                             [0, 0, -1, 0],
                             [0, 1, 0, 0]])

HADSuper = 0.5 * np.asarray([[1, 1, 1, 1],
                             [1, -1, 1, -1],
                             [1, 1, -1, -1],
                             [1, -1, -1, 1]])

HADChoi = 0.5 * np.asarray([[1, 1, 1, -1],
                            [1, 1, 1, -1],
                            [1, 1, 1, -1],
                            [-1, -1, -1, 1]])


# Single Qubit Pauli Channel
def one_q_pauli_channel_chi(px, py, pz):
    p = (px + py + pz)
    pp_chi = np.asarray([[1 - p, 0, 0, 0],
                         [0, px, 0, 0],
                         [0, 0, py, 0],
                         [0, 0, 0, pz]])
    return pp_chi


# Pauli twirled Amplitude damping channel
def analytical_pauli_twirl_of_AD_chi(p):
    # see equation 7 of  https://arxiv.org/pdf/1701.03708.pdf
    poly1 = (2 + 2 * np.sqrt(1 - p) - p) / 4
    poly2 = p / 4
    poly3 = (2 - 2 * np.sqrt(1 - p) - p) / 4
    pp_chi = np.asarray([[poly1, 0, 0, 0],
                         [0, poly2, 0, 0],
                         [0, 0, poly2, 0],
                         [0, 0, 0, poly3]])
    return pp_chi


# I \otimes Z channel or gate (two qubits)
two_qubit_paulis = n_qubit_pauli_basis(2)
IZKraus = two_qubit_paulis.ops_by_label['IZ']
IZSuper = np.diag([1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, -1, 1])

# one and zero state as a density matrix
ONE_STATE = np.asarray([[0, 0], [0, 1]])
ZERO_STATE = np.asarray([[1, 0], [0, 0]])

# Amplitude damping Kraus operators with p = 0.1
[Ad0, Ad1] = amplitude_damping_kraus(0.1)
AdKrausOps = [Ad0, Ad1]

# Use Kraus operators to find out put of channel i.e.
#    rho_out = A_0 rho A_0^\dag + A_1 rho A_1^\dag.
rho_out = np.matmul(np.matmul(Ad0, ONE_STATE), Ad0.transpose().conj()) + \
          np.matmul(np.matmul(Ad1, ONE_STATE), Ad1.transpose().conj())


# ===================================================================================================
# Test superoperator conversion tools
# ===================================================================================================

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
    p = np.random.rand()
    [Ay0, Ay1] = amplitude_damping_kraus(p)
    np.testing.assert_array_almost_equal_nulp(np.matmul(Ay0.transpose().conj(), Ay0)
                                              + np.matmul(Ay1.transpose().conj(), Ay1), np.eye(2))


def test_kraus2chi():
    assert np.allclose(HADChi, kraus2chi(HADKraus))
    p = np.random.rand()
    AdKraus = amplitude_damping_kraus(p)
    AdChi = amplitude_damping_chi(p)
    assert np.allclose(AdChi, kraus2chi(AdKraus))
    assert np.allclose(superop2chi(IZSuper), kraus2chi(IZKraus))


def test_kraus2pauli_liouville():
    p = np.random.rand()
    AdKraus = amplitude_damping_kraus(p)
    AdPauli = amplitude_damping_pauli(p)
    assert np.allclose(kraus2pauli_liouville(AdKraus), AdPauli)
    assert np.allclose(kraus2pauli_liouville(HADKraus), HADPauli)


def test_kraus2superop():
    p = np.random.rand()
    AdKraus = amplitude_damping_kraus(p)
    AdSuper = amplitude_damping_super(p)
    np.testing.assert_array_almost_equal_nulp(kraus2superop(AdKraus), AdSuper)
    # test application of super operator is the same as application of Kraus ops
    ONE_STATE_VEC = vec(ONE_STATE)
    np.testing.assert_array_almost_equal_nulp(unvec(np.matmul(kraus2superop(AdKrausOps),
                                                              ONE_STATE_VEC)), rho_out)
    assert np.allclose(kraus2superop(HADKraus), HADSuper)
    assert np.allclose(kraus2superop(IZKraus), IZSuper)


def test_kraus2choi():
    p = np.random.rand()
    AdKraus = amplitude_damping_kraus(p)
    AdChoi = amplitude_damping_choi(p)
    assert np.allclose(kraus2choi(AdKraus), AdChoi)
    assert np.allclose(kraus2choi(HADKraus), HADChoi)


def test_chi2pauli_liouville():
    p = np.random.rand()
    AdChi = amplitude_damping_chi(p)
    AdPauli = amplitude_damping_pauli(p)
    assert np.allclose(AdPauli, chi2pauli_liouville(AdChi))
    assert np.allclose(HADPauli, chi2pauli_liouville(HADChi))


def test_basis_transform_p_to_c():
    xz_pauli_basis = np.zeros((16, 1))
    xz_pauli_basis[7] = [1.]
    assert np.allclose(unvec(pauli2computational_basis_matrix(4) @ xz_pauli_basis),
                       np.kron(sigma_x, sigma_z))


def test_basis_transform_c_to_p():
    xz_pauli_basis = np.zeros((16, 1))
    xz_pauli_basis[7] = [1.]
    assert np.allclose(computational2pauli_basis_matrix(4) @ vec(np.kron(sigma_x, sigma_z)),
                       xz_pauli_basis)


def test_pl_to_choi():
    for i, pauli in enumerate(n_qubit_pauli_basis(2)):
        pl = kraus2pauli_liouville(pauli[1])
        choi = kraus2choi(pauli[1])
        assert np.allclose(choi, pauli_liouville2choi(pl))

    pl = kraus2pauli_liouville(HADAMARD)
    choi = kraus2choi(HADAMARD)
    assert np.allclose(choi, pauli_liouville2choi(pl))


def test_superop_to_kraus():
    assert np.allclose(superop2kraus(IZSuper), IZKraus)
    p = np.random.rand()
    AdSuper = amplitude_damping_super(p)
    AdKraus = amplitude_damping_kraus(p)
    kraus_ops = superop2kraus(AdSuper)
    print(kraus_ops)
    # the order of the Kraus ops matters
    # TODO: fix the sign problem in Kraus operators
    assert np.allclose([np.abs(kraus_ops[1]), np.abs(kraus_ops[0])], AdKraus)


def test_superop_to_choi():
    for i, pauli in enumerate(n_qubit_pauli_basis(2)):
        superop = kraus2superop(pauli[1])
        choi = kraus2choi(pauli[1])
        assert np.allclose(choi, superop2choi(superop))
    p = np.random.rand()
    AdSuper = amplitude_damping_super(p)
    AdChoi = amplitude_damping_choi(p)
    assert np.allclose(AdChoi, superop2choi(AdSuper))
    superop = kraus2superop(HADKraus)
    choi = kraus2choi(HADKraus)
    assert np.allclose(choi, superop2choi(superop))


def test_superop_to_pl():
    p = np.random.rand()
    AdSuper = amplitude_damping_super(p)
    AdPauli = amplitude_damping_pauli(p)
    assert np.allclose(AdPauli, superop2pauli_liouville(AdSuper))
    AdKraus = amplitude_damping_kraus(p)
    superop = kraus2superop(AdKraus)
    pauli = kraus2pauli_liouville(AdKraus)
    assert np.allclose(pauli, superop2pauli_liouville(superop))


def test_pauli_liouville_to_superop():
    p = np.random.rand()
    AdSuper = amplitude_damping_super(p)
    AdPauli = amplitude_damping_pauli(p)
    assert np.allclose(AdSuper, pauli_liouville2superop(AdPauli))
    AdKraus = amplitude_damping_kraus(p)
    superop = kraus2superop(AdKraus)
    pauli = kraus2pauli_liouville(AdKraus)
    assert np.allclose(superop, pauli_liouville2superop(pauli))


def test_choi_to_kraus():
    for i, pauli in enumerate(n_qubit_pauli_basis(2)):
        choi = kraus2choi(pauli[1])
        kraus = choi2kraus(choi)
        assert np.allclose(choi, kraus2choi(kraus))
    id_choi = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
    assert np.allclose(kraus2choi(choi2kraus(id_choi)), id_choi)
    for kraus in choi2kraus(id_choi):
        assert np.allclose(abs(kraus), np.eye(2)) or np.allclose(kraus, np.zeros((2, 2)))


def test_choi_to_super():
    p = np.random.rand()
    AdSuper = amplitude_damping_super(p)
    AdChoi = amplitude_damping_choi(p)
    assert np.allclose(AdSuper, choi2superop(AdChoi))


def test_choi_pl_bijectivity():
    assert np.allclose(choi2superop(choi2superop(np.eye(4))), np.eye(4))
    assert np.allclose(superop2choi(superop2choi(np.eye(4))), np.eye(4))
    h_choi = kraus2choi(HADAMARD)
    h_superop = kraus2superop(HADAMARD)
    assert np.allclose(choi2superop(choi2superop(h_choi)), h_choi)
    assert np.allclose(superop2choi(superop2choi(h_superop)), h_superop)


# ===================================================================================================
# Test channel and superoperator approximation tools
# ===================================================================================================


def test_pauli_twirl_of_pauli_channel():
    # diagonal channel so should not change anything
    px = np.random.rand()
    py = np.random.rand()
    pz = np.random.rand()
    pauli_chan_chi_matrix = one_q_pauli_channel_chi(px, py, pz)
    pauli_twirled_chi_matrix = pauli_twirl_chi_matrix(pauli_chan_chi_matrix)
    assert np.allclose(pauli_chan_chi_matrix, pauli_twirled_chi_matrix)


def test_pauli_twirl_of_amp_damp():
    p = np.random.rand()
    ana_chi = analytical_pauli_twirl_of_AD_chi(p)
    num_chi = pauli_twirl_chi_matrix(amplitude_damping_chi(p))
    assert np.allclose(ana_chi, num_chi)


# ==================================================================================================
# Test apply channel
# ==================================================================================================


def test_apply_kraus_ops_2_state():
    AD_kraus = amplitude_damping_kraus(0.1)
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


# ==================================================================================================
# Test physicality of Channels
# ==================================================================================================

def test_choi_is_hermitian_preserving():
    D = 2
    K = 2
    choi = rand_ops.rand_map_with_BCSZ_dist(D, K)
    assert choi_is_hermitian_preserving(choi)


def test_choi_is_trace_preserving():
    D = 2
    K = 2
    choi = rand_ops.rand_map_with_BCSZ_dist(D, K)
    assert choi_is_trace_preserving(choi)


def test_choi_is_completely_positive():
    D = 2
    K = 2
    choi = rand_ops.rand_map_with_BCSZ_dist(D, K)
    assert choi_is_completely_positive(choi)
    D = 3
    K = 2
    choi = rand_ops.rand_map_with_BCSZ_dist(D, K)
    assert choi_is_completely_positive(choi)


def test_choi_is_unital():
    px = np.random.rand()
    py = np.random.rand()
    pz = np.random.rand()
    choi = chi2choi(one_q_pauli_channel_chi(px, py, pz))
    assert choi_is_unital(choi)
    assert choi_is_unital(HADChoi)
    assert not choi_is_unital(amplitude_damping_choi(0.1))


def test_choi_is_unitary():
    px = np.random.rand()
    py = np.random.rand()
    pz = np.random.rand()
    choi = chi2choi(one_q_pauli_channel_chi(px, py, pz))
    assert not choi_is_unitary(choi)
    assert choi_is_unital(choi)
    assert choi_is_unitary(HADChoi)
    assert not choi_is_unitary(amplitude_damping_choi(0.1))
