"""A module for computing distances (and other properites) between quantum states or
processes"""
from typing import Tuple
import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.optimize import minimize_scalar
from forest.benchmarking.operator_tools.calculational import sqrtm_psd


# ===================================================================================================
# Functions for quantum states
# ===================================================================================================

def purity(rho: np.ndarray, dim_renorm=False, tol: float = 1000) -> float:
    """
    Calculates the purity :math:`P = tr[ρ^2]` of a quantum state ρ.

    As stated above lower value of the purity depends on the dimension of ρ's Hilbert space. For
    some applications this can be undesirable. For this reason we introduce an optional dimensional
    renormalization flag with the following behavior

    If the dimensional renormalization flag is FALSE (default) then  1/dim ≤ P ≤ 1.
    If the dimensional renormalization flag is TRUE then 0 ≤ P ≤ 1.

    where dim is the dimension of ρ's Hilbert space.

    :param rho: Is a dim by dim positive matrix with unit trace.
    :param dim_renorm: Boolean, default False.
    :param tol: Tolerance in machine epsilons for np.real_if_close.
    :return: P the purity of the state.
    """
    p = np.trace(rho @ rho)
    if dim_renorm:
        dim = rho.shape[0]
        p = (dim / (dim - 1.0)) * (p - 1.0 / dim)
    return np.ndarray.item(np.real_if_close(p, tol))


def impurity(rho: np.ndarray, dim_renorm=False, tol: float = 1000) -> float:
    """
    Calculates the impurity (or linear entropy) :math:`L = 1 - tr[ρ^2]` of a quantum state ρ.

    As stated above the lower value of the impurity depends on the dimension of ρ's Hilbert space.
    For some applications this can be undesirable. For this reason we introduce an optional
    dimensional renormalization flag with the following behavior

    If the dimensional renormalization flag is FALSE (default) then  0 ≤ L ≤ 1/dim.
    If the dimensional renormalization flag is TRUE then 0 ≤ L ≤ 1.

    where dim is the dimension of ρ's Hilbert space.

    :param rho: Is a dim by dim positive matrix with unit trace.
    :param dim_renorm: Boolean, default False.
    :param tol: Tolerance in machine epsilons for np.real_if_close.
    :return: L the impurity of the state.
    """
    imp = 1 - np.trace(rho @ rho)
    if dim_renorm:
        dim = rho.shape[0]
        imp = (dim / (dim - 1.0)) * imp
    return np.ndarray.item(np.real_if_close(imp, tol))


def fidelity(rho: np.ndarray, sigma: np.ndarray, tol: float = 1000) -> float:
    r"""
    Computes the fidelity :math:`F(\rho, \sigma)` between two quantum states rho and sigma.

    If the states are pure the expression reduces to

    .. math::

        F(|psi>,|phi>) = |<psi|phi>|^2

    The fidelity obeys :math:`0 ≤ F(\rho, \sigma) ≤ 1`, where
    :math:`F(\rho, \sigma)=1 iff \rho = \sigma`.

    :param rho: Is a dim by dim positive matrix with unit trace.
    :param sigma: Is a dim by dim positive matrix with unit trace.
    :param tol: Tolerance in machine epsilons for np.real_if_close.
    :return: Fidelity which is a scalar.
    """
    sqrt_rho = sqrtm_psd(rho)
    fid = (np.trace(sqrtm_psd(sqrt_rho @ sigma @ sqrt_rho))) ** 2
    return np.ndarray.item(np.real_if_close(fid, tol))


def infidelity(rho: np.ndarray, sigma: np.ndarray, tol: float = 1000) -> float:
    r"""
    Computes the infidelity, :math:`1 - F(\rho, \sigma)`, between two quantum states rho and sigma
    where :math:`F(\rho, \sigma)` is the fidelity.

    :param rho: Is a dim by dim positive matrix with unit trace.
    :param sigma: Is a dim by dim positive matrix with unit trace.
    :param tol: Tolerance in machine epsilons for np.real_if_close.
    :return: Infidelity which is a scalar.
    """
    return 1 - fidelity(rho, sigma, tol)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""
    Computes the trace distance between two states rho and sigma:

    .. math::

        T(\rho, \sigma) = (1/2)||\rho-\sigma||_1

    where :math:`||X||_1` denotes the 1 norm of X.

    :param rho: Is a dim by dim positive matrix with unit trace.
    :param sigma: Is a dim by dim positive matrix with unit trace.
    :return: Trace distance which is a scalar.
    """
    return (0.5) * np.linalg.norm(rho - sigma, 1)


def bures_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""
    Computes the Bures distance between two states rho and sigma:

    .. math::

        D_B(\rho, \sigma)^2 = 2(1- \sqrt{F(\rho, \sigma)})

    where :math:`F(\rho, \sigma)` is the fidelity.

    :param rho: Is a dim by dim positive matrix with unit trace.
    :param sigma: Is a dim by dim positive matrix with unit trace.
    :return: Bures distance which is a scalar.
    """
    return np.sqrt(2 * (1 - np.sqrt(fidelity(rho, sigma))))


def bures_angle(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""
    Computes the Bures angle (AKA Bures arc or Bures length) between two states rho and sigma:

    .. math::

        D_A(\rho, \sigma) = \arccos(\sqrt{F(\rho, \sigma)})

    where :math:`F(\rho, \sigma)` is the fidelity.

    The Bures angle is a measure of statistical distance between quantum states.

    :param rho: Is a dim by dim positive matrix with unit trace.
    :param sigma: Is a dim by dim positive matrix with unit trace.
    :return: Bures angle which is a scalar.
    """
    return np.arccos(np.sqrt(fidelity(rho, sigma)))


def quantum_chernoff_bound(rho: np.ndarray,
                           sigma: np.ndarray,
                           tol: float = 1000) -> Tuple[float, float]:
    r"""
    Computes the quantum Chernoff bound between rho and sigma.

    It is defined as

    .. math::

        ξ_{QCB}(\rho, \sigma) = - \log[ \min_{0\le s\le 1} tr(\rho^s \sigma^{1-s}) ]

    It is also common to study the non-logarithmic variety of the quantum Chernoff bound

    .. math::

        Q_{QCB}(\rho, \sigma) = \min_{0\le s\le 1} tr(\rho^s \sigma^{1-s})

    The quantum Chernoff bound has many nice properties, see [QCB]_. Importantly it is
    operationally important in the following context. Given n copies of rho or sigma the minimum
    error probability for discriminating rho from sigma is :math:`P_{e,min,n} ~ exp[-n ξ_{QCB}]`.
    
    .. [QCB] The Quantum Chernoff Bound.
          Audenaert et al.
          Phys. Rev. Lett. 98, 160501 (2007).
          https://dx.doi.org/10.1103/PhysRevLett.98.160501
          https://arxiv.org/abs/quant-ph/0610027

    :param rho: Is a dim by dim positive matrix with unit trace.
    :param sigma: Is a dim by dim positive matrix with unit trace.
    :param tol: Tolerance in machine epsilons for np.real_if_close.
    :return: The non-logarithmic quantum Chernoff bound and the s achieving the minimum.
    """

    def f(s):
        s = np.real_if_close(s)
        return np.trace(
            np.matmul(fractional_matrix_power(rho, s), fractional_matrix_power(sigma, 1 - s)))

    f_min = minimize_scalar(f, bounds=(0, 1), method='bounded')
    s_opt = np.real_if_close(f_min.x, tol)
    qcb = np.real_if_close(f_min.fun, tol)
    return qcb, s_opt


def hilbert_schmidt_ip(A: np.ndarray, B: np.ndarray, tol: float = 1000) -> float:
    r"""
    Computes the Hilbert-Schmidt (HS) inner product between two operators A and B.

    This inner product is defined as

    .. math::

        HS = (A|B) = Tr[A^\dagger B]

    where :math:`|B) = vec(B)` and :math:`(A|` is the dual vector to :math:`|A)`.

    :param A: Is a dim by dim positive matrix with unit trace.
    :param B: Is a dim by dim positive matrix with unit trace.
    :param tol: Tolerance in machine epsilons for np.real_if_close.
    :return: HS inner product which is a scalar.
    """
    hs_ip = np.trace(np.matmul(np.transpose(np.conj(A)), B))
    return np.ndarray.item(np.real_if_close(hs_ip, tol))


def smith_fidelity(rho: np.ndarray, sigma: np.ndarray, power) -> float:
    r"""
    Computes the Smith fidelity :math:`F_S(\rho, \sigma, power)` between two quantum states rho and
    sigma.

    The Smith fidelity is  defined as :math:`F_S = \sqrt{F^{power}}`, where F is  standard fidelity
    :math:`F = fidelity(\rho, \sigma)`. Since the power is only defined for values less than 2,
    it is always true that :math:`F_S > F`.

    At present there is no known operational interpretation of the Smith fidelity for an arbitrary
    power.

    :param rho: Is a dim by dim positive matrix with unit trace.
    :param sigma: Is a dim by dim positive matrix with unit trace.
    :param power: Is a positive scalar less than 2.
    :return: Smith Fidelity which is a scalar.
    """
    if power < 0:
        raise ValueError("Power must be positive")
    if power >= 2:
        raise ValueError("Power must be less than 2")
    return np.sqrt(fidelity(rho, sigma)) ** power


def total_variation_distance(P: np.ndarray, Q: np.ndarray) -> float:
    r"""
    Computes the total variation distance between two (classical) probability
    measures P(x) and Q(x).

    When x is a finite alphabet then the definition is

    .. math::

        tvd(P,Q) = (1/2) \sum_x |P(x) - Q(x)|

    where tvd(P,Q) is in [0, 1]. There is an alternate definition for non-finite alphabet measures
    involving a supremum.

    :param P: Is a dim by 1 np.ndarray.
    :param Q: Is a dim by 1 np.ndarray.
    :return: total variation distance which is a scalar.
    """
    rowsp, colsp = P.shape
    rowsq, colsq = Q.shape
    if not (colsp == colsq == 1 and rowsp > 1 and rowsq > 1):
        raise ValueError("Arrays must be the same length")
    return 0.5 * np.sum(np.abs(P - Q))


# ============================================================================
# Functions for quantum processes
# ============================================================================
def entanglement_fidelity(pauli_lio0: np.ndarray,
                          pauli_lio1: np.ndarray,
                          tol: float = 1000) -> float:
    r"""
    Returns the entanglement fidelity (F_e) between two channels, E and F, represented as Pauli
    Liouville matrix.

    The expression is

    .. math::

            F_e(E,F) = Tr[E^\dagger F] / (dim^2),

    where dim is the dimension of the Hilbert space associated with E and F.

    See the following references for more information:

    [GRAPTN]_ referenced in the superoperator_tools module. In particular section V subsection G.

    .. [H**3] General teleportation channel, singlet fraction and quasi-distillation.
           Horodecki et al.
           PRA 60, 1888 (1999).
           https://doi.org/10.1103/PhysRevA.60.1888
           https://arxiv.org/abs/quant-ph/9807091

    .. [GFID] A simple formula for the average gate fidelity of a quantum dynamical operation.
           M. Nielsen.
           Physics Letters A 303, 249 (2002).
           https://doi.org/10.1016/S0375-9601(02)01272-0
           https://arxiv.org/abs/quant-ph/0205035

    :param pauli_lio0: A dim**2 by dim**2 Pauli-Liouville matrix
    :param pauli_lio1: A dim**2 by dim**2 Pauli-Liouville matrix
    :param tol: Tolerance in machine epsilons for np.real_if_close.
    :return: Returns the entanglement fidelity between pauli_lio0 and pauli_lio1 which is a scalar.
    """
    assert pauli_lio0.shape == pauli_lio1.shape
    assert pauli_lio0.shape[0] == pauli_lio1.shape[1]
    dim_squared = pauli_lio0.shape[0]
    dim = int(np.sqrt(dim_squared))
    Fe = np.trace(np.matmul(np.transpose(np.conj(pauli_lio0)), pauli_lio1)) / (dim ** 2)
    return np.ndarray.item(np.real_if_close(Fe, tol))


def process_fidelity(pauli_lio0: np.ndarray, pauli_lio1: np.ndarray) -> float:
    r"""Returns the fidelity between two channels, E and F, represented as Pauli Liouville matrix.

    The expression is

    .. math::

             F_{process}(E,F) = ( Tr[E^\dagger F] + dim ) / (dim^2 + dim),

    which is sometimes written as

    .. math::

            F_{process}(E,F) = ( dim F_e + 1 ) / (dim + 1)

    where dim is the dimension of the Hilbert space asociated with E and F, and F_e is the
    entanglement fidelity see https://arxiv.org/abs/quant-ph/9807091 .

    NOTE: F_process is sometimes "gate fidelity" and F_e is sometimes called "process fidelity".

    If E is the ideal process, e.g. a perfect gate, and F is an experimental estimate of the
    actual process then the corresponding infidelity 1−F_process(E,F) can be seen as a
    measure of gate error, but it is not a proper metric.

    For more information see [GFID]_ and [C]_

    .. [C] Universal Quantum Gate Set Approaching Fault-Tolerant Thresholds with Superconducting
        Qubits.
        Jerry M. Chow, et al.
        Phys. Rev. Lett. 109, 060501 (2012).
        https://doi.org/10.1103/PhysRevLett.109.060501
        https://arxiv.org/abs/1202.5344

    :param pauli_lio0: A dim**2 by dim**2 Pauli-Liouville matrix
    :param pauli_lio1: A dim**2 by dim**2 Pauli-Liouville matrix
    :return: The process fidelity between pauli_lio0 and pauli_lio1 which is a scalar.
    """
    assert pauli_lio0.shape == pauli_lio1.shape
    assert pauli_lio0.shape[0] == pauli_lio1.shape[1]
    dim_squared = pauli_lio0.shape[0]
    dim = int(np.sqrt(dim_squared))

    Fe = entanglement_fidelity(pauli_lio0, pauli_lio1)

    return (dim * Fe + 1) / (dim + 1)


def process_infidelity(pauli_lio0: np.ndarray, pauli_lio1: np.ndarray) -> float:
    """
    Returns the infidelity between two channels, E and F, represented as a Pauli-Liouville
    matrix. That is::

        process_infidelity(E,F) = 1- F_process(E,F).

    See the docstrings for process_fidelity for more information.

    :param pauli_lio0: A dim**2 by dim**2 Pauli-Liouville matrix
    :param pauli_lio1: A dim**2 by dim**2 Pauli-Liouville matrix
    :return: The process fidelity between pauli_lio0 and pauli_lio1 which is a scalar.
    """
    return 1 - process_fidelity(pauli_lio0, pauli_lio1)


def diamond_norm_distance(choi0: np.ndarray, choi1: np.ndarray) -> float:
    """
    Return the diamond norm distance between two completely positive
    trace-preserving (CPTP) superoperators, represented as Choi matrices.

    The calculation uses the simplified semidefinite program of Watrous in [CBN]_

    .. note::

        This calculation becomes very slow for 4 or more qubits.

    .. [CBN] Semidefinite programs for completely bounded norms.
          J. Watrous.
          Theory of Computing 5, 11, pp. 217-238 (2009).
          http://theoryofcomputing.org/articles/v005a011
          http://arxiv.org/abs/0901.4709

    :param choi0: A 4**N by 4**N matrix (where N is the number of qubits)
    :param choi1: A 4**N by 4**N matrix (where N is the number of qubits)

    """
    # Kudos: Based on MatLab code written by Marcus P. da Silva
    # (https://github.com/BBN-Q/matlab-diamond-norm/)
    import cvxpy as cvx
    assert choi0.shape == choi1.shape
    assert choi0.shape[0] == choi1.shape[1]
    dim_squared = choi0.shape[0]
    dim = int(np.sqrt(dim_squared))

    delta_choi = choi0 - choi1
    delta_choi = (delta_choi.conj().T + delta_choi) / 2  # Enforce Hermiticity

    # Density matrix must be Hermitian, positive semidefinite, trace 1
    rho = cvx.Variable([dim, dim], complex=True)
    constraints = [rho == rho.H]
    constraints += [rho >> 0]
    constraints += [cvx.trace(rho) == 1]

    # W must be Hermitian, positive semidefinite
    W = cvx.Variable([dim_squared, dim_squared], complex=True)
    constraints += [W == W.H]
    constraints += [W >> 0]

    constraints += [(W - cvx.kron(np.eye(dim), rho)) << 0]

    J = cvx.Parameter([dim_squared, dim_squared], complex=True)
    objective = cvx.Maximize(cvx.real(cvx.trace(J.H * W)))

    prob = cvx.Problem(objective, constraints)

    J.value = delta_choi
    prob.solve()

    dnorm = prob.value * 2

    return dnorm


def _is_square(n):
    return n == np.round(np.sqrt(n)) ** 2


def watrous_bounds(choi: np.ndarray) -> Tuple[float, float]:
    """
    Return the Watrous bounds for the diamond norm of a superoperator in
    the Choi representation.

    If this is applied to the difference of two Choi
    representations it yields bounds on the diamond norm distance.

    The bound can be found in `this <https://cstheory.stackexchange.com/a/4920>`_
    StackOverflow answer, although the results can also be found scattered in 
    the literature.

    :param choi: dim1 by dim2 matrix (for qubits, dim = 4**Ni, where Ni is a number of qubits)
    """
    if len(choi.shape) != 2:
        raise ValueError("Watrous bounds only defined for matrices")

    if not (_is_square(choi.shape[0]) and _is_square(choi.shape[1])):
        raise ValueError("Choi matrix must have dimensions that are perfect squares")

    _, s, _ = np.linalg.svd(choi)
    nuclear_norm = np.sum(s)

    return nuclear_norm, choi.shape[0] * nuclear_norm
