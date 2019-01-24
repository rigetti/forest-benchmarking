import forest_benchmarking.random_operators as rand_ops
import numpy.random
from scipy.linalg import fractional_matrix_power as matpow
import forest_benchmarking.distance_measures as dm
import numpy as np

numpy.random.seed(7)  # seed random number generation for all calls to rand_ops


# =================================================================================================
# Test:  Purity
# =================================================================================================
def test_purity_standard():
    rho0 = np.diag([1, 0])
    rho1 = np.diag([0.9, 0.1])
    rho2 = np.diag([0.5, 0.5])
    assert dm.purity(rho0) == 1.0
    assert np.allclose(dm.purity(rho1), 0.82)  # by hand calc
    assert dm.purity(rho2) == 0.5
    rho_qutrit = np.diag([1.0, 1.0, 1.0]) / 3
    assert dm.purity(rho_qutrit) == 1 / 3


def test_purity_renorm():
    D = 2
    rho0 = np.diag([1, 0])
    rho1 = np.diag([0.9, 0.1])
    rho2 = np.diag([0.5, 0.5])
    assert dm.purity(rho0, dim_renorm=True) == 1.0
    assert np.allclose(dm.purity(rho1, dim_renorm=True),
                       (D / (D - 1)) * (0.82 - 1 / D))  # by hand calc
    assert dm.purity(rho2, dim_renorm=True) == 0.0
    rho_qutrit = np.diag([1.0, 1.0, 1.0]) / 3
    assert dm.purity(rho_qutrit, dim_renorm=True) == 0.0


# =================================================================================================
# Test:  Fidelity
# =================================================================================================
def test_fidelity():
    # both states are real so no need to conjugate
    zero = np.array([[1], [0]])
    rho = np.matmul(zero, zero.transpose())
    assert dm.fidelity(rho, rho) == 1.0

    theta = np.pi
    psi_theta = np.array([[np.cos(theta / 2)], [np.sin(theta / 2)]])
    sigma = np.matmul(psi_theta, psi_theta.transpose())
    assert np.allclose(dm.fidelity(rho, sigma), 0.0)


# =================================================================================================
# Test:  Trace distance
# =================================================================================================
def test_trace_distance():
    # both states are real so no need to conjugate
    zero = np.array([[1], [0]])
    rho = np.matmul(zero, zero.transpose())
    assert dm.trace_distance(rho, rho) == 0

    theta = np.pi
    psi_theta = np.array([[np.cos(theta / 2)], [np.sin(theta / 2)]])
    sigma = np.matmul(psi_theta, psi_theta.transpose())
    assert np.allclose(dm.trace_distance(rho, sigma), 0.5)


# =================================================================================================
# Test:  Bures distance
# =================================================================================================
def test_bures_distance():
    # both states are real so no need to conjugate
    zero = np.array([[1], [0]])
    rho = np.matmul(zero, zero.transpose())
    assert dm.bures_distance(rho, rho) == 0

    theta = np.pi
    psi_theta = np.array([[np.cos(theta / 2)], [np.sin(theta / 2)]])
    sigma = np.matmul(psi_theta, psi_theta.transpose())
    assert np.allclose(dm.bures_distance(rho, sigma), np.sqrt(2.0))


# =================================================================================================
# Test:  Bures angle
# =================================================================================================
def test_bures_angle():
    # both states are real so no need to conjugate
    zero = np.array([[1], [0]])
    rho = np.matmul(zero, zero.transpose())
    assert dm.bures_angle(rho, rho) == 0

    theta = np.pi
    psi_theta = np.array([[np.cos(theta / 2)], [np.sin(theta / 2)]])
    sigma = np.matmul(psi_theta, psi_theta.transpose())
    assert dm.bures_angle(rho, sigma) == np.pi / 2


# =================================================================================================
# Test:  Quantum Chernoff bound
# =================================================================================================

# TODO: need more serious testing

def test_qcb_for_pure_states():
    # both states are real so no need to conjugate
    zero = np.array([[1], [0]])
    rho = np.matmul(zero, zero.transpose())
    theta = np.pi / 2
    psi_theta = np.array([[np.cos(theta / 2)], [np.sin(theta / 2)]])
    sigma = np.matmul(psi_theta, psi_theta.transpose())
    assert np.allclose(dm.quantum_chernoff_bound(rho, sigma)[0], 0.5000000000)


def test_qcb_for_mixed_states():
    # both states are real so no need to conjugate
    theta = np.pi / 2
    G = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    r = 0.9
    lam_p = r
    lam_m = 1 - lam_p
    rho = np.diag([lam_p, lam_m])
    rho_theta = np.matmul(np.matmul(G, rho), G.transpose())
    qcb, s = dm.quantum_chernoff_bound(rho, rho_theta)
    assert np.allclose(dm.quantum_chernoff_bound(rho, rho_theta)[0], 0.59999999999)


# =================================================================================================
# Test:  Hilbert-Schmidt inner product
# =================================================================================================
def test_HS_obeys_cauchy_schwarz():
    D = 2
    Ur = rand_ops.haar_rand_unitary(D)
    U = Ur + np.conj(Ur.T)
    A = np.eye(D)
    B = A / 3
    assert dm.hilbert_schmidt_ip(U, U) * dm.hilbert_schmidt_ip(A, A) >= abs(dm.hilbert_schmidt_ip(A, U))**2
    assert dm.hilbert_schmidt_ip(U, U) * dm.hilbert_schmidt_ip(B, B) >= abs(dm.hilbert_schmidt_ip(B, U))**2


def test_HS_obeys_linearity():
    D = 2
    Ur = rand_ops.haar_rand_unitary(D)
    U = Ur + np.conj(Ur.T)
    A = np.eye(D)
    B = A / 3
    alpha = 0.17
    beta = 0.6713
    LHS = alpha * dm.hilbert_schmidt_ip(U, A) + beta * dm.hilbert_schmidt_ip(U, B)
    RHS = dm.hilbert_schmidt_ip(U, alpha * A + beta * B)
    assert np.allclose(LHS, RHS)


def test_diamon_norm():
    # Test cases borrowed from qutip,
    # https://github.com/qutip/qutip/blob/master/qutip/tests/test_metrics.py
    # which were in turn generated using QuantumUtils for MATLAB
    # (https://goo.gl/oWXhO9) by Christopher Granade

    _I = np.asarray([[1, 0], [0, 1]])
    _X = np.asarray([[0, 1], [1, 0]])
    _Y = np.asarray([[0, -1.0j], [1.0j, 0]])
    _H = np.asarray([[1, 1], [1, -1]]) / np.sqrt(2)

    def _gate_to_superop(gate):
        dim = gate.shape[0]
        superop = np.outer(gate, gate.conj().T)
        superop = np.reshape(superop, [dim]*4)
        superop = np.transpose(superop, [0, 3, 1, 2])
        return superop

    def _superop_to_choi(superop):
        dim = superop.shape[0]
        superop = np.transpose(superop, (0, 2, 1, 3))
        choi = np.reshape(superop, [dim**2] * 2)
        return choi

    def _gate_to_choi(gate):
        return _superop_to_choi(_gate_to_superop(gate))

    choi0 = _gate_to_choi(_I)
    choi1 = _gate_to_choi(_X)
    dnorm = dm.diamond_norm(choi0, choi1)
    assert np.isclose(2.0, dnorm, rtol=0.01)

    turns_dnorm = [[1.000000e-03, 3.141591e-03],
                   [3.100000e-03, 9.738899e-03],
                   [1.000000e-02, 3.141463e-02],
                   [3.100000e-02, 9.735089e-02],
                   [1.000000e-01, 3.128689e-01],
                   [3.100000e-01, 9.358596e-01]]

    for turns, target in turns_dnorm:
        choi0 = _gate_to_choi(_X)
        choi1 = _gate_to_choi(matpow(_X, 1 + turns))
        dnorm = dm.diamond_norm(choi0, choi1)
        assert np.isclose(target, dnorm, rtol=0.01)

    hadamard_mixtures = [[1.000000e-03, 2.000000e-03],
                         [3.100000e-03, 6.200000e-03],
                         [1.000000e-02, 2.000000e-02],
                         [3.100000e-02, 6.200000e-02],
                         [1.000000e-01, 2.000000e-01],
                         [3.100000e-01, 6.200000e-01]]

    for p, target in hadamard_mixtures:
        chan0 = _gate_to_superop(_I) * (1 - p) + _gate_to_superop(_H) * p
        chan1 = _gate_to_superop(_I)

        choi0 = _superop_to_choi(chan0)
        choi1 = _superop_to_choi(chan1)
        dnorm = dm.diamond_norm(choi0, choi1)
        assert np.isclose(dnorm, target, rtol=0.01)

    choi0 = _gate_to_choi(_I)
    choi1 = _gate_to_choi(matpow(_Y, 0.5))
    dnorm = dm.diamond_norm(choi0, choi1)
    assert np.isclose(dnorm, np.sqrt(2), rtol=0.01)
