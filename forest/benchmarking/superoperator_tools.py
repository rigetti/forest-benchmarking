"""A module containing tools for working with superoperators. Eg. converting between different
representations of superoperators.

We have arbitrarily decided to use a column stacking convention.

For more information about the conventions used, look at the file in
/docs/Superoperator representations.md

Further references include:

[GRAPTN] Tensor networks and graphical calculus for open quantum systems
         Wood et al.
         Quant. Inf. Comp. 15, 0579-0811 (2015)
         (no DOI)
         https://arxiv.org/abs/1111.6950

[MATQO] On the Matrix Representation of Quantum Operations
        Nambu et al.,
        arXiv: 0504091 (2005)
        https://arxiv.org/abs/quant-ph/0504091

[DUAL] On duality between quantum maps and quantum states
       Zyczkowski et al.,
       Open Syst. Inf. Dyn. 11, 3 (2004)
       https://dx.doi.org/10.1023/B:OPSY.0000024753.05661.c2
       https://arxiv.org/abs/quant-ph/0401119

"""
from typing import Sequence, Tuple, List
import numpy as np
from forest.benchmarking.utils import n_qubit_pauli_basis, partial_trace


# ==================================================================================================
# Superoperator conversion tools
# ==================================================================================================


def vec(matrix: np.ndarray) -> np.ndarray:
    """
    Vectorize, i.e. "vec", a matrix by column stacking.

    For example the 2 by 2 matrix A

    A = [[a, b]
         [c, d]]      becomes  |A>> := vec(A) = (a, c, b, d)^T ,

    where |A>> denotes the vec'ed version of A and T denotes transpose.

    :param matrix: A N (rows) by M (columns) numpy array.
    :return: Returns a column vector with  N by M rows.
    """
    return np.asarray(matrix).T.reshape((-1, 1))


def unvec(vector: np.ndarray, shape: Tuple[int, int] = None) -> np.ndarray:
    """
    Take a column vector and turn it into a matrix.

    By default, the unvec'ed matrix is assumed to be square. Specifying shape = [N, M] will
    produce a N by M matrix where N is the number of rows and M is the number of columns.

    Consider |A>> := vec(A) = (a, c, b, d)^T. `unvec(|A>>)` should return

    A = [[a, b]
         [c, d]].

    :param vector: A (N*M) by 1 numpy array.
    :param shape: The shape of the output matrix; by default, the matrix is assumed to be square.
    :return: Returns a N by M matrix.
    """
    vector = np.asarray(vector)
    if shape is None:
        dim = int(np.sqrt(vector.size))
        shape = dim, dim
    matrix = vector.reshape(*shape).T
    return matrix


def kraus2chi(kraus_ops: Sequence[np.ndarray]) -> np.ndarray:
    r"""
    Convert a set of Kraus operators (representing a channel) to
    a chi matrix which is also known as a process matrix.

    :param kraus_ops: A list or tuple of N Kraus operators
    :return: Returns a dim**2 by dim**2 matrix.
    """
    if isinstance(kraus_ops, np.ndarray):  # handle input of single kraus op
        if len(kraus_ops[0].shape) < 2:  # first elem is not a matrix
            kraus_ops = [kraus_ops]

    dim = np.asarray(kraus_ops[0]).shape[0]  # kraus op is dim by dim matrix
    c_vecs = [computational2pauli_basis_matrix(dim) @ vec(kraus) for kraus in kraus_ops]
    chi_mat = sum([c_vec @ c_vec.conj().T for c_vec in c_vecs])
    return chi_mat


def kraus2superop(kraus_ops: Sequence[np.ndarray]) -> np.ndarray:
    r"""
    Convert a set of Kraus operators (representing a channel) to
    a superoperator using the column stacking convention.

    Suppose the N Kraus operators M_i are dim by dim matrices. Then the
    the superoperator is a (dim**2) \otimes (dim**2) matrix. Using the relation
    column stacking relation,
    vec(ABC) = (C^T\otimes A) vec(B), we can show

    super_operator = \sum_i^N ( M_i^\dagger )^T \otimes M_i
                   = \sum_i^N M_i^* \otimes M_i

    where A^* is the complex conjugate of a matrix A, A^T is the transpose,
    and A^\dagger is the complex conjugate and transpose.

    Note: This function can also convert non-square Kraus operators to a superoperator,
    these frequently arise in quantum measurement theory and quantum error correction. In that
    situation consider a single Kraus operator that is M by N then the superoperator will be a
    M**2 by N**2 matrix.

    :param kraus_ops: A tuple of N Kraus operators
    :return: Returns a dim**2 by dim**2 matrix.
    """
    if isinstance(kraus_ops, np.ndarray):  # handle input of single kraus op
        if len(kraus_ops[0].shape) < 2:  # first elem is not a matrix
            kraus_ops = [kraus_ops]

    rows, cols = np.asarray(kraus_ops[0]).shape

    # Standard case of square Kraus operators is if rows==cols.
    # When representing a partial projection, e.g. a single measurement operator
    # M_i = Id \otimes <i| for i \in {0,1}, rows!=cols.
    # However the following will work in both cases:

    superop = np.zeros((rows**2, cols**2), dtype=complex)

    for op in kraus_ops:
        superop += np.kron(np.asarray(op).conj(), op)
    return superop


def kraus2pauli_liouville(kraus_ops: Sequence[np.ndarray]) -> np.ndarray:
    """
    Convert a set of Kraus operators (representing a channel) to
    a Pauli-Liouville matrix (aka Pauli Transfer matrix).

    :param kraus_ops: A list of Kraus operators
    :return: Returns dim**2 by dim**2 Pauli-Liouville matrix
    """
    return superop2pauli_liouville(kraus2superop(kraus_ops))


def kraus2choi(kraus_ops: Sequence[np.ndarray]) -> np.ndarray:
    r"""
    Convert a set of Kraus operators (representing a channel) to
    a Choi matrix using the column stacking convention.

    Suppose the N Kraus operators M_i are dim by dim matrices. Then the
    the Choi matrix is a dim**2 by dim**2 matrix

    choi_matrix = \sum_i^N |M_i>> (|M_i>>)^\dagger
                = \sum_i^N |M_i>> <<M_i|

    where |M_i>> = vec(M_i)

    :param kraus_ops: A list of N Kraus operators
    :return: Returns a dim**2 by dim**2 matrix.
    """
    if isinstance(kraus_ops, np.ndarray):  # handle input of single kraus op
        if len(kraus_ops[0].shape) < 2:  # first elem is not a matrix
            kraus_ops = [kraus_ops]

    return sum([vec(op) @ vec(op).conj().T for op in kraus_ops])


def chi2pauli_liouville(chi_matrix: np.ndarray) -> np.ndarray:
    r"""
    Converts a chi matrix (aka a process matrix) to the Pauli Liouville representation.

    :param chi_matrix:  a dim**2 by dim**2 process matrix
    :return: dim**2 by dim**2 Pauli-Liouville matrix
    """
    return choi2pauli_liouville(chi2choi(chi_matrix))


def chi2kraus(chi_matrix: np.ndarray) -> List[np.ndarray]:
    """
    Converts a chi matrix into a list of Kraus operators. (operators with small norm may be
    excluded)

    :param chi_matrix:  a dim**2 by dim**2 process matrix
    :return: list of Kraus operators
    """
    return pauli_liouville2kraus(chi2pauli_liouville(chi_matrix))


def chi2superop(chi_matrix: np.ndarray) -> np.ndarray:
    """
    Converts a chi matrix into a superoperator.

    :param chi_matrix:  a dim**2 by dim**2 process matrix
    :return: a dim**2 by dim**2 superoperator matrix
    """
    return pauli_liouville2superop(chi2pauli_liouville(chi_matrix))


def chi2choi(chi_matrix: np.ndarray) -> np.ndarray:
    """
    Converts a chi matrix into a Choi matrix.

    :param chi_matrix:  a dim**2 by dim**2 process matrix
    :return: a dim**2 by dim**2 Choi matrix
    """
    dim = int(np.sqrt(np.asarray(chi_matrix).shape[0]))
    p2c = pauli2computational_basis_matrix(dim)
    return p2c @ chi_matrix @ p2c.conj().T


def superop2kraus(superop: np.ndarray) -> List[np.ndarray]:
    """
    Converts a superoperator into a list of Kraus operators. (operators with small norm may be excluded)

    :param superop: a dim**2 by dim**2 superoperator
    :return: list of Kraus operators
    """
    return choi2kraus(superop2choi(superop))


def superop2chi(superop: np.ndarray) -> np.ndarray:
    """
    Converts a superoperator into a list of Kraus operators. (operators with small norm may be excluded)

    :param superop: a dim**2 by dim**2 superoperator
    :return: a dim**2 by dim**2 process matrix
    """
    return kraus2chi(superop2kraus(superop))


def superop2pauli_liouville(superop: np.ndarray) -> np.ndarray:
    """
    Converts a superoperator into a pauli_liouville matrix. This is achieved by a linear change of basis.

    :param superop: a dim**2 by dim**2 superoperator
    :return: dim**2 by dim**2 Pauli-Liouville matrix
    """
    dim = int(np.sqrt(np.asarray(superop).shape[0]))
    c2p_basis_transform = computational2pauli_basis_matrix(dim)
    return c2p_basis_transform @ superop @ c2p_basis_transform.conj().T * dim


def superop2choi(superop: np.ndarray) -> np.ndarray:
    """
    Convert a superoperator into a choi matrix. The operation acts equivalently to choi2superop, as it is a bijection.

    :param superop: a dim**2 by dim**2 superoperator
    :return: dim**2 by dim**2 choi matrix
    """
    dim = int(np.sqrt(np.asarray(superop).shape[0]))
    return np.reshape(superop, [dim] * 4).swapaxes(0, 3).reshape([dim ** 2, dim ** 2])


def pauli_liouville2kraus(pl_matrix: np.ndarray) -> List[np.ndarray]:
    """
    Converts a pauli_liouville matrix into a list of Kraus operators. (operators with small norm may be excluded)

    :param pl_matrix: a dim**2 by dim**2 pauli_liouville matrix
    :return: list of Kraus operators
    """
    return choi2kraus(pauli_liouville2choi(pl_matrix))


def pauli_liouville2chi(pl_matrix: np.ndarray) -> np.ndarray:
    """
    Converts a pauli_liouville matrix into a chi matrix. (operators with small norm may be excluded)

    :param pl_matrix: a dim**2 by dim**2 pauli_liouville matrix
    :return: a dim**2 by dim**2 process matrix
    """
    return kraus2chi(pauli_liouville2kraus(pl_matrix))


def pauli_liouville2superop(pl_matrix: np.ndarray) -> np.ndarray:
    """
    Converts a pauli_liouville matrix into a superoperator. This is achieved by a linear change of basis.

    :param pl_matrix: a dim**2 by dim**2 Pauli-Liouville matrix
    :return: dim**2 by dim**2 superoperator
    """
    dim = int(np.sqrt(np.asarray(pl_matrix).shape[0]))
    p2c_basis_transform = pauli2computational_basis_matrix(dim)
    return p2c_basis_transform @ pl_matrix @ p2c_basis_transform.conj().T / dim


def pauli_liouville2choi(pl_matrix: np.ndarray) -> np.ndarray:
    """
    Convert a Pauli-Liouville matrix into a choi matrix.

    :param pl_matrix: a dim**2 by dim**2 Pauli-Liouville matrix
    :return: dim**2 by dim**2 choi matrix
    """
    return superop2choi(pauli_liouville2superop(pl_matrix))


def choi2kraus(choi: np.ndarray, tol: float = 1e-9) -> List[np.ndarray]:
    """
    Converts a Choi matrix into a list of Kraus operators. (operators with small norm may be
    excluded)

    :param choi: a dim**2 by dim**2 choi matrix
    :param tol: optional threshold parameter for eigenvalues/kraus ops to be discarded
    :return: list of Kraus operators
    """
    eigvals, v = np.linalg.eigh(choi)
    return [np.lib.scimath.sqrt(eigval) * unvec(np.array([evec]).T) for eigval, evec in
            zip(eigvals, v.T) if abs(eigval) > tol]


def choi2chi(choi: np.ndarray) -> np.ndarray:
    """
    Converts a Choi matrix into a chi matrix. (operators with small norm may be excluded)
    :param choi: a dim**2 by dim**2 choi matrix
    :return: a dim**2 by dim**2 process matrix
    """
    return kraus2chi(choi2kraus(choi))


def choi2superop(choi: np.ndarray) -> np.ndarray:
    """
    Convert a choi matrix into a superoperator. The operation acts equivalently to superop2choi, as it is a bijection.

    :param choi: a dim**2 by dim**2 choi matrix
    :return: dim**2 by dim**2 superoperator
    """
    dim = int(np.sqrt(np.asarray(choi).shape[0]))
    return np.reshape(choi, [dim] * 4).swapaxes(0, 3).reshape([dim ** 2, dim ** 2])


def choi2pauli_liouville(choi: np.ndarray) -> np.ndarray:
    """
    Convert a choi matrix into a Pauli-Liouville matrix.

    :param choi: a dim**2 by dim**2 choi matrix
    :return: dim**2 by dim**2 Pauli-Liouville matrix
    """
    return superop2pauli_liouville(choi2superop(choi))


def pauli2computational_basis_matrix(dim) -> np.ndarray:
    """
    Produces a basis transform matrix that converts from a pauli basis to the computational basis.
        p2c_transform = sum_{k=1}^{dim**2}  | sigma_k >> <k|
    For example,
        sigma_x = [0, 1, 0, 0].T in the 'pauli basis'
        p2c * sigma_x = vec(sigma_x) = | sigma_x >>

    :param dim: dimension of the hilbert space on which the operators act.
    :return: A dim**2 by dim**2 basis transform matrix
    """
    n_qubits = int(np.log2(dim))

    conversion_mat = np.zeros((dim ** 2, dim ** 2), dtype=complex)

    for i, pauli in enumerate(n_qubit_pauli_basis(n_qubits)):
        pauli_label = np.zeros((dim ** 2, 1))
        pauli_label[i] = 1.
        pauli_mat = pauli[1]
        conversion_mat += np.kron(vec(pauli_mat), pauli_label.T)

    return conversion_mat


def computational2pauli_basis_matrix(dim) -> np.ndarray:
    """
    Produces a basis transform matrix that converts from a computational basis to a pauli basis.
    Conjugate transpose of pauli2computational_basis_matrix with an extra dimensional factor.
        c2p_transform = sum_{k=1}^{dim**2}  | k > << sigma_k |
    For example,
        vec(sigma_z) = | sigma_z >> = [1, 0, 0, -1].T in the computational basis
        c2p * | sigma_z >> = [0, 0, 0, 1].T

    :param dim: dimension of the hilbert space on which the operators act.
    :return: A dim**2 by dim**2 basis transform matrix
    """
    return pauli2computational_basis_matrix(dim).conj().T / dim


# ==================================================================================================
# Channel and Superoperator approximation tools
# ==================================================================================================
def pauli_twirl_chi_matrix(chi_matrix: np.ndarray) -> np.ndarray:
    r"""
    Implements a Pauli twirl of a chi matrix (aka a process matrix).

    See the folloiwng reference for more details

    [SPICC] Scalable protocol for identification of correctable codes
            Silva et al.,
            PRA 78, 012347 2008
            http://dx.doi.org/10.1103/PhysRevA.78.012347
            https://arxiv.org/abs/0710.1900

    Note: Pauli twirling a quantum channel can give rise to a channel that is less noisy; use with
    care.

    :param chi_matrix:  a dim**2 by dim**2 chi or process matrix
    :return: dim**2 by dim**2 chi or process matrix
    """
    return np.diag(chi_matrix.diagonal())


# TODO: Honest approximations for Channels that act on one or MORE qubits.

# ==================================================================================================
# Apply channel
# ==================================================================================================
def apply_kraus_ops_2_state(kraus_ops: Sequence[np.ndarray], state: np.ndarray) -> np.ndarray:
    r"""
    Apply a quantum channel, specified by Kraus operators, to state.

    The Kraus operators need not be square.

    :param kraus_ops: A list or tuple of N Kraus operators, each operator is M by dim ndarray
    :param state: A dim by dim ndarray which is the density matrix for the state
    :return: M by M ndarray which is the density matrix for the state after the action of kraus_ops
    """
    if isinstance(kraus_ops, np.ndarray):  # handle input of single kraus op
        if len(kraus_ops[0].shape) < 2:  # first elem is not a matrix
            kraus_ops = [kraus_ops]

    dim, _ = state.shape
    rows, cols = kraus_ops[0].shape

    if dim != cols:
        raise ValueError("Dimensions of state and Kraus operator are incompatible")

    new_state = np.zeros((rows, rows))
    for M in kraus_ops:
        new_state += M @ state @ np.transpose(M.conj())

    return new_state


def apply_choi_matrix_2_state(choi: np.ndarray, state: np.ndarray) -> np.ndarray:
    r"""
    Apply a quantum channel, specified by a Choi matrix (using the column stacking convention),
    to a state.

    The Choi matrix is a dim**2 by dim**2 matrix and the state rho is a dim by dim matrix. The
    output state is

    rho_{out} = Tr_{A_{in}}[(rho^T \otimes Id) Choi_matrix ],

    where T denotes transposition and Tr_{A_{in}} is the partial trace over input Hilbert space H_{
    A_{in}}; the Choi matrix representing a process mapping rho in H_{A_{in}} to rho_{out}
    in H_{B_{out}} is regarded as an operator on the space H_{A_{in}} \otimes H_{B_{out}}.


    :param choi: a dim**2 by dim**2 matrix
    :param state: A dim by dim ndarray which is the density matrix for the state
    :return: a dim by dim matrix.
    """
    dim = int(np.sqrt(np.asarray(choi).shape[0]))
    dims = [dim, dim]
    tot_matrix = np.kron(state.transpose(), np.identity(dim)) @ choi
    return partial_trace(tot_matrix, [1], dims)


# ==================================================================================================
# Check physicality of Channels
# ==================================================================================================
def kraus_operators_are_valid(kraus_ops: Sequence[np.ndarray],
                              rtol: float = 1e-05,
                              atol: float = 1e-08)-> bool:
    """
    Checks if a set of Kraus operators are valid.

    :param kraus_ops: A list of Kraus operators
    :param rtol: The relative tolerance parameter in np.allclose
    :param atol: The absolute tolerance parameter in np.allclose
    :return: True if the Kraus operators are valid with the given tolerance; False otherwise.
    """
    if isinstance(kraus_ops, np.ndarray):  # handle input of single kraus op
        if len(kraus_ops[0].shape) < 2:  # first elem is not a matrix
            kraus_ops = [kraus_ops]
    rows, _ = np.asarray(kraus_ops[0]).shape
    # Standard case of square Kraus operators is if rows==cols. For non-square Kraus ops it is
    # required that sum_i M_i^\dagger M_i  = np.eye(rows,rows).
    id_iff_valid = sum(np.transpose(op).conjugate().dot(op) for op in kraus_ops)
    # TODO: check if each POVM element (i.e. M_i^\dagger M_i) is PSD
    return np.allclose(id_iff_valid, np.eye(rows), rtol=rtol, atol=atol)


def choi_is_hermitian_preserving(choi: np.ndarray, rtol: float = 1e-05,
                                 atol: float = 1e-08) -> bool:
    """
    Checks if  a quantum process, specified by a Choi matrix, is hermitian-preserving.

    :param choi: a dim**2 by dim**2 Choi matrix
    :param rtol: The relative tolerance parameter in np.allclose
    :param atol: The absolute tolerance parameter in np.allclose
    :return: Returns True if the quantum channel is hermitian preserving with the given tolerance;
        False otherwise.
    """
    # Equation 3.31 of [GRAPTN]
    return np.allclose(choi, choi.conj().T, rtol=rtol, atol=atol)


def choi_is_trace_preserving(choi: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """
    Checks if  a quantum process, specified by a Choi matrix, is trace-preserving.

    :param choi: A dim**2 by dim**2 Choi matrix
    :param rtol: The relative tolerance parameter in np.allclose
    :param atol: The absolute tolerance parameter in np.allclose
    :return: Returns True if the quantum channel is trace-preserving with the given tolerance;
        False otherwise.
    """
    rows, cols = choi.shape
    dim = int(np.sqrt(rows))
    # the choi matrix acts on the Hilbert space H_{in} \otimes H_{out}.
    # We want to trace out H_{out} and so keep the H_{in} space at index 0.
    keep = [0]
    id_iff_tp = partial_trace(choi, keep, [dim, dim])
    # Equation 3.33 of [GRAPTN]
    return np.allclose(id_iff_tp, np.identity(dim), rtol=rtol, atol=atol)


def choi_is_completely_positive(choi: np.ndarray, limit: float = 1e-09) -> bool:
    """
    Checks if  a quantum process, specified by a Choi matrix, is completely positive.

    :param choi: A dim**2 by dim**2 Choi matrix
    :param limit: A tolerance parameter, all eigenvalues must be greater than -|limit|.
    :return: Returns True if the quantum channel is completely positive with the given tolerance;
        False otherwise.
    """
    evals, evecs = np.linalg.eig(choi)
    # Equation 3.35 of [GRAPTN]
    return all(x >= -abs(limit) for x in evals)


def choi_is_unital(choi: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """
    Checks if  a quantum process, specified by a Choi matrix, is unital.

    A process is unital iff it maps the identity to itself.

    :param choi: A dim**2 by dim**2 Choi matrix
    :param rtol: The relative tolerance parameter in np.allclose
    :param atol: The absolute tolerance parameter in np.allclose
    :return: Returns True if the quantum channel is unital with the given tolerance; False
        otherwise.
    """
    rows, cols = choi.shape
    dim = int(np.sqrt(rows))
    id_iff_unital = apply_choi_matrix_2_state(choi, np.identity(dim))
    return np.allclose(id_iff_unital, np.identity(dim), rtol=rtol, atol=atol)


def choi_is_unitary(choi: np.ndarray, limit: float = 1e-09) -> bool:
    """
    Checks if  a quantum process, specified by a Choi matrix, is unitary.

    :param choi: A dim**2 by dim**2 Choi matrix
    :param limit: A tolerance parameter to discard Kraus operators with small norm.
    :return: Returns True if the quantum channel is unitary with the given tolerance; False
        otherwise.
    """
    kraus_ops = choi2kraus(choi, tol=limit)
    return len(kraus_ops) == 1

# ==================================================================================================
# Project Channels to CP, TNI, TP, and physical
# ==================================================================================================
def proj_choi_to_completely_positive(choi: np.ndarray) -> np.ndarray:
    """
    Projects the Choi representation of a process into the nearest Choi matrix in the space of
    completely positive maps.

    Equation 8 of [PGD]

    [PGD] Maximum-likelihood quantum process tomography via projected gradient descent
          Knee et al.,
          Phys. Rev. A 98, 062336 (2018)
          https://dx.doi.org/10.1103/PhysRevA.98.062336
          https://arxiv.org/abs/1803.10062

    :param choi: Choi representation of a process
    :return: closest Choi matrix in the space of completely positive maps
    """
    hermitian = (choi + choi.conj().T) / 2  # enforce Hermiticity
    evals, v = np.linalg.eigh(hermitian)
    evals[evals < 0] = 0  # enforce completely positive by removing negative eigenvalues
    diag = np.diag(evals)
    return v @ diag @ v.conj().T


def proj_choi_to_trace_non_increasing(choi: np.ndarray) -> np.ndarray:
    """
    Projects the Choi matrix of a process into the space of trace non-increasing maps.

    Equation 33 of [PGD]

    :param choi: Choi representation of a process
    :return: Choi representation of the projected trace non-increasing process
    """
    dim = int(np.sqrt(choi.shape[0]))

    # trace out the output Hilbert space
    pt = partial_trace(choi, dims=[dim, dim], keep=[0])

    hermitian = (pt + pt.conj().T) / 2  # enforce Hermiticity
    d, v = np.linalg.eigh(hermitian)
    d[d > 1] = 1  # enforce trace preserving
    D = np.diag(d)
    projection = v @ D @ v.conj().T

    trace_increasing_part = np.kron((pt - projection) / dim, np.eye(dim))

    return choi - trace_increasing_part


def proj_choi_to_trace_preserving(choi: np.ndarray) -> np.ndarray:
    """
    Projects the Choi representation of a process to the closest processes in the space of trace
    preserving maps.

    Equation 12 of [PGD], but without vecing the Choi matrix. See choi_is_trace_preserving for
    comparison.

    :param choi: Choi representation of a process
    :return: Choi representation of the projected trace preserving process
    """
    dim = int(np.sqrt(choi.shape[0]))

    # trace out the output Hilbert space, keep the input space at index 0
    pt = partial_trace(choi, dims=[dim, dim], keep=[0])
    # isolate the part the violates the condition we want, namely pt = Id
    diff = pt - np.eye(dim)
    # we want to subtract off the violation from the larger operator, so 'invert' the partial_trace
    subtract = np.kron(diff/dim, np.eye(dim))
    return choi - subtract


def proj_choi_to_physical(choi, make_trace_preserving=True):
    """
    Projects the given Choi matrix into the subspace of Completetly Positive and either
    Trace Perserving (TP) or Trace-Non-Increasing maps.

    Uses Dykstra's algorithm with the stopping criterion presented in:

    [DYKALG] Dykstra’s algorithm and robust stopping criteria
             Birgin et al.,
             (Springer US, Boston, MA, 2009), pp. 828–833, ISBN 978-0-387-74759-0.
             https://doi.org/10.1007/978-0-387-74759-0_143

    This method is suggested in [PGD]

    :param choi: the Choi representation estimate of a quantum process.
    :param make_trace_preserving: default true, projects the estimate to a trace-preserving
        process. If false the output process may only be trace non-increasing
    :return: The Choi representation of the Completely Positive, Trace Preserving (CPTP) or Trace
        Non-Increasing map that is closest to the given state.
    """
    old_CP_change = np.zeros_like(choi)
    old_TP_change = np.zeros_like(choi)
    last_CP_projection = np.zeros_like(choi)
    last_state = choi

    while True:
        # Dykstra's algorithm
        pre_CP = last_state - old_CP_change
        CP_projection = proj_choi_to_completely_positive(pre_CP)
        new_CP_change = CP_projection - pre_CP

        pre_TP = CP_projection - old_TP_change
        if make_trace_preserving:
            new_state = proj_choi_to_trace_preserving(pre_TP)
        else:
            new_state = proj_choi_to_trace_non_increasing(pre_TP)
        new_TP_change = new_state - pre_TP

        CP_change_change = new_CP_change - old_CP_change
        TP_change_change = new_TP_change - old_TP_change
        state_change = new_state - last_state

        # stopping criterion
        # norm(mat) is the frobenius norm
        # norm(mat)**2 is thus equivalent to the dot product vec(mat) dot vec(mat)
        if np.linalg.norm(CP_change_change) ** 2 + np.linalg.norm(TP_change_change) ** 2 \
                + 2 * abs(np.dot(vec(old_TP_change).conj().T, vec(state_change))) \
                + 2 * abs(np.dot(vec(old_CP_change).conj().T,
                                 vec(CP_projection - last_CP_projection))) < 1e-4:
            break

        # store results from this iteration
        old_CP_change = new_CP_change
        old_TP_change = new_TP_change
        last_CP_projection = CP_projection
        last_state = new_state

    return new_state
