"""A module containing tools for converting between different representations of superoperators.

We have arbitrarily decided to use a column stacking convention.

For more information about the conventions used, look at the file in
/docs/Superoperator representations.md

Further references include:

.. [GRAPTN] Tensor networks and graphical calculus for open quantum systems.
         Wood et al.
         Quant. Inf. Comp. 15, 0579-0811 (2015).
         (no DOI)
         https://arxiv.org/abs/1111.6950

.. [MATQO] On the Matrix Representation of Quantum Operations.
        Nambu et al.
        arXiv: 0504091 (2005).
        https://arxiv.org/abs/quant-ph/0504091

.. [DUAL] On duality between quantum maps and quantum states.
       Zyczkowski et al.
       Open Syst. Inf. Dyn. 11, 3 (2004).
       https://dx.doi.org/10.1023/B:OPSY.0000024753.05661.c2
       https://arxiv.org/abs/quant-ph/0401119

"""
from typing import Sequence, Tuple, List
import numpy as np
from forest.benchmarking.utils import n_qubit_pauli_basis


def vec(matrix: np.ndarray) -> np.ndarray:
    """
    Vectorize, or "vec", a matrix by column stacking.

    For example the 2 by 2 matrix A::

        A = [[a, b]
             [c, d]]

    becomes::

      |A>> := vec(A) = (a, c, b, d)^T

    where `|A>>` denotes the vec'ed version of A and :math:`^T` denotes transpose.

    :param matrix: A N (rows) by M (columns) numpy array.
    :return: Returns a column vector with  N by M rows.
    """
    return np.asarray(matrix).T.reshape((-1, 1))


def unvec(vector: np.ndarray, shape: Tuple[int, int] = None) -> np.ndarray:
    """
    Take a column vector and turn it into a matrix.

    By default, the unvec'ed matrix is assumed to be square. Specifying shape = [N, M] will
    produce a N by M matrix where N is the number of rows and M is the number of columns.

    Consider::

        |A>> := vec(A) = (a, c, b, d)^T

    `unvec(|A>>)` should return::

        A = [[a, b]
             [c, d]]

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
    """
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
    the superoperator is a (dim**2) by (dim**2) matrix. Using the column stacking relation

    .. math::

        vec(ABC) = (C^T \otimes A) vec(B)

    we can show

    .. math::

        \rm{super\_operator} = \sum_i^N ( M_i^\dagger )^T \otimes M_i
                       = \sum_i^N M_i^* \otimes M_i

    where :math:`A^*` is the complex conjugate of a matrix A, :math:`A^T` is the transpose,
    and :math:`A^\dagger` is the complex conjugate and transpose.

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

    .. math::

        \rm{choi\_matrix} = \sum_i^N |M_i>> (|M_i>>)^\dagger = \sum_i^N |M_i>> <<M_i|

    where::

        |M_i>> = vec(M_i)

    :param kraus_ops: A list of N Kraus operators
    :return: Returns a dim**2 by dim**2 matrix.
    """
    if isinstance(kraus_ops, np.ndarray):  # handle input of single kraus op
        if len(kraus_ops[0].shape) < 2:  # first elem is not a matrix
            kraus_ops = [kraus_ops]

    return sum([vec(op) @ vec(op).conj().T for op in kraus_ops])


def chi2pauli_liouville(chi_matrix: np.ndarray) -> np.ndarray:
    """
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
    r"""
    Produces a basis transform matrix that converts from the unnormalized pauli basis to the
    computational basis

    .. math::

        \rm{p2c\_transform(dim)} = \sum_{k=1}^{dim^2}  | \sigma_k >> <k|

    For example

    .. math::

        \sigma_x = [0, 1, 0, 0].T

    in the 'pauli basis', so

    .. math::

        p2c * \sigma_x = vec(\sigma_x) = | \sigma_x >>

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
    r"""
    Produces a basis transform matrix that converts from a computational basis to the unnormalized
    pauli basis.

    This is the conjugate transpose of pauli2computational_basis_matrix with an extra dimensional
    factor.

    .. math::

        \rm{c2p\_transform(dim)}  = \frac{1}{dim} sum_{k=1}^{dim^2}  | k > << \sigma_k |

    For example

    .. math::

        vec(\sigma_z) = | \sigma_z >> = [1, 0, 0, -1].T

    in the computational basis, so

    .. math::

        c2p * | \sigma_z >> = [0, 0, 0, 1].T

    :param dim: dimension of the hilbert space on which the operators act.
    :return: A dim**2 by dim**2 basis transform matrix
    """
    return pauli2computational_basis_matrix(dim).conj().T / dim
