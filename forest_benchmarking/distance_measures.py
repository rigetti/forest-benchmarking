"""A module for computing distances (and other properites) between quantum states or
processes"""
import numpy as np
from scipy.linalg import sqrtm
from scipy.linalg import fractional_matrix_power
from scipy.optimize import minimize_scalar


# ===================================================================================================
# Functions for quantum states
# ===================================================================================================

def purity(rho, dim_renorm=False):
    """
    Calculates the purity P of a quantum state.

    If the dimensional renormalization flag is FALSE (default) then  1/D ≤ P ≤ 1.
    If the dimensional renormalization flag is TRUE then 0 ≤ P ≤ 1.

    :param rho: Is a D x D positive matrix.
    :param dim_renorm: Boolean, default False.
    :return: P the purity of the state.
    """
    if dim_renorm:
        D = np.shape(rho)[0]
        Ptemp = np.trace(np.matmul(rho, rho))
        P = (D / (D - 1.0)) * (Ptemp - 1.0 / D)
    else:
        P = np.trace(np.matmul(rho, rho))
    return P


def fidelity(rho, sigma):
    """
    Computes the fidelity F(rho,sigma) between two quantum states rho and sigma.

    If the states are pure the expression reduces to F(|psi>,|phi>) = |<psi|phi>|^2.

    The fidelity obeys 0 ≤ F(rho,sigma) ≤ 1, where F(rho,sigma)=1 iff rho = sigma and
    F(rho,sigma)= 0 iff
    :param rho: Is a D x D positive matrix.
    :param sigma: Is a D x D positive matrix.
    :return: Fidelity which is a scalar.
    """
    return (np.trace(sqrtm(np.matmul(np.matmul(sqrtm(rho), sigma), sqrtm(rho))))) ** 2


def trace_distance(rho, sigma):
    """
    Computes the trace distance between two states rho and sigma i.e.
    T(rho,sigma) = (1/2)||rho-sigma||_1 , where ||X||_1 denotes the 1 norm of X.

    :param rho: Is a D x D positive matrix.
    :param sigma: Is a D x D positive matrix.
    :return: Trace distance which is a scalar.
    """
    return (0.5) * np.linalg.norm(rho - sigma, 1)


def bures_distance(rho, sigma):
    """
    Computes the Bures distance between two states rho and sigma i.e.
    D_B(rho,sigma)^2 = 2(1- sqrt[F(rho,sigma)]) , where F(rho,sigma) is the fidelity.

    :param rho: Is a D x D positive matrix with unit trace.
    :param sigma: Is a D x D positive matrix with unit trace.
    :return: Bures distance which is a scalar.
    """
    return np.sqrt(2 * (1 - np.sqrt(fidelity(rho, sigma))))


def bures_angle(rho, sigma):
    """
    Computes the Bures angle (AKA Bures arc or Bures length) between two states rho and sigma i.e.
    D_A(rho,sigma) = arccos(sqrt[F(rho,sigma)]) , where F(rho,sigma) is the fidelity.
    The Bures angle is a measure of statistical distance between quantum states.

    :param rho: Is a D x D positive matrix.
    :param sigma: Is a D x D positive matrix.
    :return: Bures angle which is a scalar.
    """
    return np.arccos(np.sqrt(fidelity(rho, sigma)))


def quantum_chernoff_bound(rho, sigma):
    """
    Computes the exponent of the quantum Chernoff bound between rho and sigma.

    :param rho: Is a D x D positive matrix.
    :param sigma: Is a D x D positive matrix.
    :return: the exponent of the quantum Chernoff bound angle which is a scalar.
    """

    def f(s):
        s = np.real_if_close(s)
        return np.trace(
            np.matmul(fractional_matrix_power(rho, s), fractional_matrix_power(sigma, 1 - s)))

    f_min = minimize_scalar(f, bounds=(0, 1), method='bounded')
    s_opt = f_min.x
    qcb = f_min.fun
    return qcb, s_opt


def hilbert_schmidt_ip(A, B):
    """
    Computes the Hilbert-Schmidt (HS) inner product between two operators A and B as
        HS = (A|B) = Tr[A^\dagger B]
    where |B) = vec(B) and (A| is the dual vector to |A).

    :param A: Is a D x D matrix.
    :param B: Is a D x D matrix.
    :return: HS inner product which is a scalar.
    """
    return np.trace(np.matmul(np.transpose(np.conj(A)), B))


# ============================================================================
# Functions for quantum processes
# ============================================================================

def diamond_norm(choi0: np.ndarray, choi1: np.ndarray) -> float:
    """Return the diamond norm between two completely positive
    trace-preserving (CPTP) superoperators, represented as Choi matrices.

    The calculation uses the simplified semidefinite program of Watrous
    [arXiv:0901.4709](http://arxiv.org/abs/0901.4709). This calculation
    becomes very slow for 4 or more qubits.
    [J. Watrous, [Theory of Computing 5, 11, pp. 217-238
    (2009)](http://theoryofcomputing.org/articles/v005a011/)]

    :param choi0: A 4^N x 4^N matrix (where N is the number of qubits)
    :param choi1: A 4^N x 4^N matrix (where N is the number of qubits)
 
    """
    # Kudos: Based on MatLab code written by Marcus P. da Silva
    # (https://github.com/BBN-Q/matlab-diamond-norm/)
    import cvxpy as cvx
    assert choi0.shape == choi1.shape
    assert choi0.shape[0] == choi1.shape[1]
    dim2 = choi0.shape[0]
    dim = int(np.sqrt(dim2))

    delta_choi = choi0 - choi1
    delta_choi = (delta_choi.conj().T + delta_choi) / 2  # Enforce Hermiticity

    # Density matrix must be Hermitian, positive semidefinite, trace 1
    rho = cvx.Variable([dim, dim], complex=True)
    constraints = [rho == rho.H]
    constraints += [rho >> 0]
    constraints += [cvx.trace(rho) == 1]

    # W must be Hermitian, positive semidefinite
    W = cvx.Variable([dim2, dim2], complex=True)
    constraints += [W == W.H]
    constraints += [W >> 0]

    constraints += [(W - cvx.kron(np.eye(dim), rho)) << 0]

    J = cvx.Parameter([dim2, dim2], complex=True)
    objective = cvx.Maximize(cvx.real(cvx.trace(J.H * W)))

    prob = cvx.Problem(objective, constraints)

    J.value = delta_choi
    prob.solve()

    dnorm = prob.value * 2

    return dnorm
