import numpy as np
from scipy.linalg import eigh


def partial_trace(rho, keep, dims, optimize=False):
    r"""
    Calculate the partial trace.

    Consider a joint state ρ on the Hilbert space :math:`H_a \otimes H_b`. We wish to trace out
    :math:`H_b`

    .. math::

        ρ_a = Tr_b(ρ)

    :param rho: 2D array, the matrix to trace.
    :param keep: An array of indices of the spaces to keep after being traced. For instance,
                 if the space is A x B x C x D and we want to trace out B and D, keep = [0, 2].
    :param dims: An array of the dimensions of each space. For example, if the space is
                 A x B x C x D, dims = [dim_A, dim_B, dim_C, dim_D].
    :param optimize: optimize argument in einsum
    :return:  ρ_a, a 2D array i.e. the traced matrix
    """
    # Code from
    # https://scicomp.stackexchange.com/questions/30052/calculate-partial-trace-of-an-outer-product-in-python
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[keep])

    idx1 = [i for i in range(Ndim)]
    idx2 = [Ndim + i if i in keep else i for i in range(Ndim)]
    rho_a = rho.reshape(np.tile(dims, 2))
    rho_a = np.einsum(rho_a, idx1 + idx2, optimize=optimize)
    return rho_a.reshape(Nkeep, Nkeep)


def outer_product(bra1: np.ndarray, bra2: np.ndarray) -> np.ndarray:
    """
    Given two possibly complex row vectors `bra1` and `bra2` construct the outer product::

        |bra1> <bra2|

    :param bra1: Is a dim by 1 np.ndarray.
    :param bra2: Is a dim by 1 np.ndarray.
    :return: the outer product.
    """
    rows1, cols1 = bra1.shape
    rows2, cols2 = bra2.shape
    if not (cols1 == cols2 == 1 and rows1 > 1 and rows2 > 1):
        raise ValueError("The vectors do not have the correct dimensions.")
    return np.outer(bra1, bra2.conj())


def inner_product(bra1: np.ndarray, bra2: np.ndarray) -> complex:
    """
    Given two possibly complex row vectors `bra1` and `bra2` construct the inner
    product::

        <bra1|bra2>

    which can be complex,

    :param bra1: Is a dim by 1 np.ndarray.
    :param bra2: Is a dim by 1 np.ndarray.
    :return: the inner product.
    """
    rows1, cols1 = bra1.shape
    rows2, cols2 = bra2.shape
    if not (cols1 == cols2 == 1 and rows1 > 1 and rows2 > 1):
        raise ValueError("The vectors do not have the correct dimensions.")
    return np.transpose(bra1.conj()) @ bra2


# code from https://github.com/scipy/scipy/pull/4775/files
# Algorithm 9.1.1. in Gene H. Golub, Charles F. van Loan, Matrix Computations 4th ed.
def sqrtm_psd(matrix: np.ndarray, check_finite: bool =True) -> np.ndarray:
    """
    Calculates the square root of a matrix that is positive semidefinite.

    :param matrix: the matrix to square root
    :param check_finite: Whether to check that the input matrices contain only finite numbers.
    :return: sqrt(matrix)
    """
    w, v = eigh(matrix, check_finite=check_finite)
    # As we further know that your matrix is positive semidefinite,
    # we can guard against precision errors by doing
    w = np.maximum(w, 0)
    w = np.sqrt(w)
    matrix_sqrt = (v * w).dot(v.conj().T)
    return matrix_sqrt
