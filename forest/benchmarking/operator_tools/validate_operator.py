"""A module allowing one to check properties of operators or matrices.
"""
import numpy as np


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
    Checks if a square matrix A is symmetric, :math:`A = A ^T`, where :math:`^T` denotes transpose.

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
    Checks if a square matrix A is idempotent, :math:`A^2 = A`.

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
    Checks if a square matrix A is normal, :math:`A^\dagger A = A A^\dagger`,
    where :math:`^\dagger` denotes conjugate transpose.

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
    Checks if a square matrix A is Hermitian, :math:`A = A^\dagger`, where :math:`^\dagger`
    denotes conjugate transpose.

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
    Checks if a square matrix A is unitary, :math:`A^\dagger A = A A^\dagger = Id`,
    where :math:`^\dagger` denotes conjugate transpose and `Id` denotes the identity.

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
    Checks if a square Hermitian matrix A is positive definite, :math:`eig(A) > 0`.

    In this numerical implementation we check if each eigenvalue obeys :math:`eig(A) > -|atol|`,
    the strict condition can be recovered by setting `atol = 0`.

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
    Checks if a square Hermitian matrix A is positive semi-definite :math:`eig(A) \geq 0`.

    In this numerical implementation we check if each eigenvalue obeys :math:`eig(A) \geq -|atol|`.

    :param matrix: a M by M Hermitian matrix.
    :param rtol: The relative tolerance parameter in np.allclose
    :param atol: The absolute tolerance parameter in np.allclose
    :return: True if the matrix is normal; False otherwise.
    """
    if not is_hermitian_matrix(matrix, rtol, atol):
        raise ValueError("The matrix is not Hermitian.")
    evals, _ = np.linalg.eigh(matrix)
    return all(x >= -abs(atol) for x in evals)
