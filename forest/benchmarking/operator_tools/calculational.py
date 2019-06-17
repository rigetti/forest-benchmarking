import numpy as np


def partial_trace(rho, keep, dims, optimize=False):
    r"""Calculate the partial trace.

    Consider a joint state ρ on the Hilbert space H_a \otimes H_b. We wish to trace over H_b e.g.

    ρ_a = Tr_b(ρ).

    :param rho: 2D array, the matrix to trace.
    :param keep: An array of indices of the spaces to keep after being traced. For instance,
                 if the space is A x B x C x D and we want to trace out B and D, keep = [0,2].
    :param dims: An array of the dimensions of each space. For example, if the space is
                 A x B x C x D, dims = [dim_A, dim_B, dim_C, dim_D].
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
    r"""
    Given two possibly complex row vectors `bra1` and `bra2` construct the outer product:

    outer_product = |bra1> <bra2|.

    :param bra1: Is a dim by 1 np.ndarray.
    :param bra2: Is a dim by 1 np.ndarray.
    :return: the outer product.
    """
    rows1, cols1 = bra1.shape
    rows2, cols2 = bra2.shape
    if cols1 == cols2 == 1 and rows1 > 1 and rows2 > 1:
        out_prod = np.outer(bra1, bra2.conj())
    return out_prod


def inner_product(bra1: np.ndarray, bra2: np.ndarray) -> complex:
    r"""
    Given two possibly complex row vectors `bra1` and `bra2` construct the inner product:

    inner_product = <bra1|bra2>,

    which can be complex,

    :param bra1: Is a dim by 1 np.ndarray.
    :param bra2: Is a dim by 1 np.ndarray.
    :return: the inner product.
    """
    rows1, cols1 = bra1.shape
    rows2, cols2 = bra2.shape
    if cols1 == cols2 == 1 and rows1 > 1 and rows2 > 1:
        in_prod = np.transpose(bra1.conj()) @ bra2
    return in_prod
