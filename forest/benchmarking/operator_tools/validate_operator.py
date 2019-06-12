"""A module allowing one to check if superoperators or channels are physical.

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
from typing import Sequence
import numpy as np
from forest.benchmarking.utils import partial_trace
from forest.benchmarking.operator_tools.superoperator_transformations import choi2kraus
from forest.benchmarking.operator_tools.superoperator_tools import apply_choi_matrix_2_state


# ==================================================================================================
# Check properties of operators or matrices
# ==================================================================================================
def is_square_matrix(matrix: np.ndarray) -> bool:
    """
    Checks if a matrix is square.

    :param matrix: a M by N matrix.
    :param rtol: The relative tolerance parameter in np.allclose
    :param atol: The absolute tolerance parameter in np.allclose
    :return: True if the matrix is square; False otherwise.
    """
    if len(matrix.shape) != 2:
        raise ValueError("The object is not a matrix.")
    rows, cols = matrix.shape
    return rows == cols


def is_symmetric_matrix(matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """
    Checks if a square matrix is symmetric. That is

    A is symmetric iff $A = A ^T$,

    where $T$ denotes transpose.

    :param matrix: a M by M matrix.
    :param rtol: The relative tolerance parameter in np.allclose
    :param atol: The absolute tolerance parameter in np.allclose
    :return: True if the matrix is symmetric; False otherwise.
    """
    if not is_square_matrix(matrix):
        raise ValueError("The matrix is not square.")
    return np.allclose(matrix, matrix.T, rtol=rtol, atol=atol)


def is_identity_matrix(matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """
    Checks if a square matrix is the identity matrix.

    :param matrix: a M by M matrix.
    :param rtol: The relative tolerance parameter in np.allclose
    :param atol: The absolute tolerance parameter in np.allclose
    :return: True if the matrix is the identity matrix; False otherwise.
    """
    if not is_square_matrix(matrix):
        raise ValueError("The matrix is not square.")
    Id = np.eye(len(matrix))
    return np.allclose(matrix, Id, rtol=rtol, atol=atol)


def is_idempotent_matrix(matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    """
    Checks if a square matrix A is idempotent. That is

    A is idempotent iff $A^2 = A.$

    :param matrix: a M by M matrix.
    :param rtol: The relative tolerance parameter in np.allclose
    :param atol: The absolute tolerance parameter in np.allclose
    :return: True if the matrix is the idempotent; False otherwise.
    """
    if not is_square_matrix(matrix):
        raise ValueError("The matrix is not square.")
    return np.allclose(matrix, matrix @ matrix, rtol=rtol, atol=atol)


def is_normal_matrix(matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""
    Checks if a square matrix A is normal. That is

    A is normal iff $A^{\dagger} A = A A^{\dagger}$,

    where $\dagger$ denotes conjugate transpose.

    :param matrix: a M by M matrix.
    :param rtol: The relative tolerance parameter in np.allclose
    :param atol: The absolute tolerance parameter in np.allclose
    :return: True if the matrix is normal; False otherwise.
    """
    if not is_square_matrix(matrix):
        raise ValueError("The matrix is not square.")
    AB = matrix.T.conj() @ matrix
    BA = matrix @ matrix.T.conj()
    return np.allclose(AB, BA, rtol=rtol, atol=atol)


def is_hermitian_matrix(matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""
    Checks if a square matrix A is Hermitian. That is

    A is Hermitian iff $A = A^{\dagger}$,

    where $\dagger$ denotes conjugate transpose.

    :param matrix: a M by M matrix.
    :param rtol: The relative tolerance parameter in np.allclose
    :param atol: The absolute tolerance parameter in np.allclose
    :return: True if the matrix is Hermitian; False otherwise.
    """
    if not is_square_matrix(matrix):
        raise ValueError("The matrix is not square.")
    return np.allclose(matrix, matrix.T.conj(), rtol=rtol, atol=atol)


def is_unitary_matrix(matrix: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
    r"""
    Checks if a square matrix A is unitary. That is

    A is unitary iff $A^{\dagger} A = A A^{\dagger}$ = Id,

    where $\dagger$ denotes conjugate transpose and Id denotes the identity.

    :param matrix: a M by M matrix.
    :param rtol: The relative tolerance parameter in np.allclose
    :param atol: The absolute tolerance parameter in np.allclose
    :return: True if the matrix is normal; False otherwise.
    """
    if not is_square_matrix(matrix):
        raise ValueError("The matrix is not square.")
    AB = matrix.T.conj() @ matrix
    BA = matrix @ matrix.T.conj()
    Id = np.eye(len(matrix))
    return np.allclose(AB, Id, rtol=rtol, atol=atol) and np.allclose(BA, Id, rtol=rtol, atol=atol)


def is_positive_definite_matrix(matrix: np.ndarray,
                                rtol: float = 1e-05,
                                atol: float = 1e-08) -> bool:
    r"""
    Checks if a square Hermitian matrix A is positive definite. That is

    A is positive definite iff eig(A) > 0.

    In this numerical implementation we check if each eigenvalue obeys eig(A) > -|atol|,
    the strict condition can be recoved by setting `atol = 0`.

    :param matrix: a M by M Hermitian matrix.
    :param rtol: The relative tolerance parameter in np.allclose
    :param atol: The absolute tolerance parameter in np.allclose
    :return: True if the matrix is normal; False otherwise.
    """
    if not is_hermitian_matrix(matrix, rtol, atol):
        raise ValueError("The matrix is not Hermitian.")
    evals, _ = np.linalg.eigh(matrix)
    return all(x > -abs(atol) for x in evals)


def is_positive_semidefinite_matrix(matrix: np.ndarray,
                                    rtol: float = 1e-05,
                                    atol: float = 1e-08) -> bool:
    r"""
    Checks if a square Hermitian matrix A is positive semi-definite. That is

    A is positive semi-definite iff eig(A) >= 0.

    In this numerical implementation we check if each eigenvalue obeys

    eig(A) >= -|atol|.

    :param matrix: a M by M Hermitian matrix.
    :param rtol: The relative tolerance parameter in np.allclose
    :param atol: The absolute tolerance parameter in np.allclose
    :return: True if the matrix is normal; False otherwise.
    """
    if not is_hermitian_matrix(matrix, rtol, atol):
        raise ValueError("The matrix is not Hermitian.")
    evals, _ = np.linalg.eigh(matrix)
    return all(x >= -abs(atol) for x in evals)
