"""A module for converting between different representations of superoperators.

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
import numpy as np
from forest.benchmarking.utils import n_qubit_pauli_basis


def vec(matrix):
    """
    Vectorize, i.e. "vec", a matrix by column stacking.

    For example the 2x2 matrix A

    A = [[a, b]
         [c, d]]      becomes  |A>> := vec(A) = (a, c, b, d)^T ,

    where |A>> denotes the vec'ed version of A and T is a transpose.

    :param matrix: A N x M numpy array.
    :return: Returns a column vector with  N x M rows.
    """
    return matrix.T.reshape((-1, 1))


def unvec(vector):
    """
    Take a column vector and turn it into a matrix assuming it was square.

    For consider |A>> := vec(A) = (a, c, b, d)^T. `unvec(|A>>)` should return

    A = [[a, b]
         [c, d]].

    :param vector: A N^2 x 1 numpy array.
    :return: Returns a column vector with  N x N rows.
    """
    n_row = vector.shape[0]
    dim = int(np.sqrt(n_row))
    matrix = vector.reshape(dim, dim).T
    return matrix


def kraus2superop(kraus_ops: list):
    r"""
    Convert a set of Kraus operators (representing a channel) to
    a superoperator using the column stacking convention.

    Suppose the N Kraus operators M_i are DxD matrices. Then the
    the superoperator is a D*D \otimes D*D matrix. Using the relation
    column stacking relation,
    ${\rm vec}(ABC) = (C^T\otimes A) {\rm vec}(B)$, we can show

    super_operator = \sum_i^N ( M_i^\dagger )^T \otimes M_i
                   = \sum_i^N M_i^* \otimes M_i

    where A^* is the complex conjugate of a matrix A, A^T is the transpose,
    and A^\dagger is the complex conjugate and transpose.

    :param kraus_ops: A tuple of N Kraus operators
    :return: Returns a D^2 x D^2 matrix.
    """
    if not isinstance(kraus_ops, (list, tuple)):
        kraus_ops = [kraus_ops]

    dim_squared = kraus_ops[0].size
    superop = np.zeros((dim_squared, dim_squared), dtype=complex)

    for op in kraus_ops:
        superop += np.kron(op.conj(), op)
    return superop


def kraus2pauli_liouville(kraus_ops: list):
    """
    Convert a set of Kraus operators (representing a channel) to
    a pauli-liouville matrix.

    :param kraus_ops: A list of Kraus operators
    :return: Returns D^2 x D^2 pauli-liouville matrix
    """
    return superop2pauli_liouville(kraus2superop(kraus_ops))


def kraus2choi(kraus_ops: list):
    r"""
    Convert a set of Kraus operators (representing a channel) to
    a Choi matrix using the column stacking convention.

    Suppose the N Kraus operators M_i are DxD matrices. Then the
    the Choi matrix is a D^2 x D^2 matrix

    choi_matrix = \sum_i^N |M_i>> (|M_i>>)^\dagger
                = \sum_i^N |M_i>> <<M_i|

    where |M_i>> = vec(M_i)

    :param kraus_ops: A list of N Kraus operators
    :return: Returns a D^2 x D^2 matrix.
    """
    if not isinstance(kraus_ops, (list, tuple)):
        kraus_ops = [kraus_ops]

    dim_squared = kraus_ops[0].size
    choi = np.zeros((dim_squared, dim_squared), dtype=complex)

    for op in kraus_ops:
        temp = vec(op)
        choi += np.kron(temp, temp.conj().T)
    return choi


def superop2kraus(superop: np.ndarray):
    """
    Converts a superoperator into a list of Kraus operators. (operators with small norm may be excluded)

    :param superop: a dim**2 by dim**2 superoperator
    :return: list of Kraus operators
    """
    return choi2kraus(superop2choi(superop))


def superop2pauli_liouville(superop: np.ndarray):
    """
    Converts a superoperator into a pauli_liouville matrix. This is achieved by a linear change of basis.

    :param superop: a dim**2 by dim**2 superoperator
    :return: dim**2 by dim**2 pauli-liouville matrix
    """
    dim = int(np.sqrt(superop.shape[0]))
    c2p_basis_transform = computational2pauli_basis_matrix(dim)
    return c2p_basis_transform @ superop @ c2p_basis_transform.conj().T * dim


def superop2choi(superop: np.ndarray):
    """
    Convert a superoperator into a choi matrix. The operation acts equivalently to choi2superop, as it is a bijection.

    :param superop: a dim**2 by dim**2 superoperator
    :return: dim**2 by dim**2 choi matrix
    """
    dim = int(np.sqrt(superop.shape[0]))
    return np.reshape(superop, [dim] * 4).swapaxes(0, 3).reshape([dim ** 2, dim ** 2])


def pauli_liouville2kraus(pl_matrix: np.ndarray):
    """
    Converts a pauli_liouville matrix into a list of Kraus operators. (operators with small norm may be excluded)

    :param pl_matrix: a dim**2 by dim**2 pauli_liouville matrix
    :return: list of Kraus operators
    """
    return choi2kraus(pauli_liouville2choi(pl_matrix))


def pauli_liouville2superop(pl_matrix: np.ndarray):
    """
    Converts a pauli_liouville matrix into a superoperator. This is achieved by a linear change of basis.

    :param pl_matrix: a dim**2 by dim**2 pauli-liouville matrix
    :return: dim**2 by dim**2 superoperator
    """
    dim = int(np.sqrt(pl_matrix.shape[0]))
    p2c_basis_transform = pauli2computational_basis_matrix(dim)
    return p2c_basis_transform @ pl_matrix @ p2c_basis_transform.conj().T / dim


def pauli_liouville2choi(pl_matrix: np.ndarray):
    """
    Convert a pauli-liouville matrix into a choi matrix.

    :param pl_matrix: a dim**2 by dim**2 pauli-liouville matrix
    :return: dim**2 by dim**2 choi matrix
    """
    return superop2choi(pauli_liouville2superop(pl_matrix))


def choi2kraus(choi: np.ndarray):
    """
    Converts a choi matrix into a list of Kraus operators. (operators with small norm may be excluded)

    :param choi: a dim**2 by dim**2 choi matrix
    :return: list of Kraus operators
    """
    eigvals, v = np.linalg.eigh(choi)
    return [np.lib.scimath.sqrt(eigval) * unvec(np.array([evec]).T) for eigval, evec in zip(eigvals, v.T) if
            abs(eigval) > 1e-16]


def choi2superop(choi: np.ndarray):
    """
    Convert a choi matrix into a superoperator. The operation acts equivalently to superop2choi, as it is a bijection.

    :param choi: a dim**2 by dim**2 choi matrix
    :return: dim**2 by dim**2 superoperator
    """
    dim = int(np.sqrt(choi.shape[0]))
    return np.reshape(choi, [dim] * 4).swapaxes(0, 3).reshape([dim ** 2, dim ** 2])


def choi2pauli_liouville(choi: np.ndarray):
    """
    Convert a choi matrix into a pauli-liouville matrix.

    :param choi: a dim**2 by dim**2 choi matrix
    :return: dim**2 by dim**2 pauli-liouville matrix
    """
    return superop2pauli_liouville(choi2superop(choi))


def pauli2computational_basis_matrix(dim):
    """
    Produces a basis transform matrix that converts from a pauli basis to the computational basis.
        p2c_transform = sum_{k=1}^{dim^2}  | sigma_k >> <k|
    For example,
        sigma_x = [0, 1, 0, 0].T in the 'pauli basis'
        p2c * sigma_x = vec(sigma_x) = | sigma_x >>

    :param dim: dimension of the hilbert space on which the operators act.
    :return: A dim^2 by dim^2 basis transform matrix
    """
    n_qubits = int(np.log2(dim))

    conversion_mat = np.zeros((dim ** 2, dim ** 2), dtype=complex)

    for i, pauli in enumerate(n_qubit_pauli_basis(n_qubits)):
        pauli_label = np.zeros((dim ** 2, 1))
        pauli_label[i] = 1.
        pauli_mat = pauli[1]
        conversion_mat += np.kron(vec(pauli_mat), pauli_label.T)

    return conversion_mat


def computational2pauli_basis_matrix(dim):
    """
    Produces a basis transform matrix that converts from a computational basis to a pauli basis. Conjugate transpose of
    pauli2computational_basis_matrix with an extra dimensional factor.
        c2p_transform = sum_{k=1}^{dim^2}  | k > << sigma_k |
    For example,
        vec(sigma_z) = | sigma_z >> = [1, 0, 0, -1].T in the computational basis
        c2p * | sigma_z >> = [0, 0, 0, 1].T

    :param dim: dimension of the hilbert space on which the operators act.
    :return: A dim^2 by dim^2 basis transform matrix
    """
    return pauli2computational_basis_matrix(dim).conj().T / dim
