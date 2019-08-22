"""A module for generating random quantum states and processes.

Pseudocode for many of these routines can be found in the appendix of the paper:

.. [BAYES] Practical Bayesian Tomography.
    Granade et al.
    New Journal of Physics 18, 033024 (2016).
    https://dx.doi.org/10.1088/1367-2630/18/3/033024
    https://arxiv.org/abs/1509.03770
"""
from typing import Optional, List, Union

import numpy as np
from numpy import linalg as la
from scipy.linalg import sqrtm
from sympy.combinatorics import Permutation
from numpy.random import RandomState
from forest.benchmarking.operator_tools.calculational import partial_trace


def ginibre_matrix_complex(dim: int, k: int, rs: Optional[RandomState] = None) -> np.ndarray:
    r"""
    Given a scalars dim and k, returns a dim by k matrix, drawn from the complex Ginibre
    ensemble [IM]_.

    Each element is distributed

    .. math::
        \sim [N(0, 1) + i · N(0, 1)]

    Here :math:`X \sim N(0,1)` denotes a normally distributed random variable.

    .. [IM] Induced measures in the space of mixed quantum states.
         Zyczkowski et al.
         J. Phys A: Math. and Gen. 34, 7111 (2001).
         https://doi.org/10.1088/0305-4470/34/35/335
         https://arxiv.org/abs/quant-ph/0012101

    :param dim: Hilbert space dimension.
    :param k: Ultimately becomes the rank of a state.
    :param rs: Optional random state.
    :return: Returns a dim by k matrix, drawn from the Ginibre ensemble.
    """
    if rs is None:
        rs = np.random
    return rs.randn(dim, k) + 1j * rs.randn(dim, k)


def haar_rand_unitary(dim: int, rs=None) -> np.ndarray:
    """
    Given a Hilbert space dimension dim this function returns a unitary operator
    U ∈ C^(dim by dim) drawn from the Haar measure [MEZ]_.

    The error is of order 10^-16.

    .. [MEZ] How to generate random matrices from the classical compact groups.
          Mezzadri.
          Notices of the American Mathematical Society 54, 592 (2007).
          http://www.ams.org/notices/200705/fea-mezzadri-web.pdf
          https://arxiv.org/abs/math-ph/0609050

    :param dim: Hilbert space dimension (scalar).
    :param rs: Optional random state
    :return: Returns a dim by dim unitary operator U drawn from the Haar measure.
    """
    if rs is None:
        rs = np.random
    Z = ginibre_matrix_complex(dim=dim, k=dim, rs=rs)  # /np.sqrt(2)
    Q, R = np.linalg.qr(Z)
    diag = np.diagonal(R)
    lamb = np.diag(diag) / np.absolute(diag)
    return np.matmul(Q, lamb)


def haar_rand_state(dim: int) -> np.ndarray:
    """   
    Given a Hilbert space dimension dim this function returns a vector
    representing a random pure state operator drawn from the Haar measure.

    :param dim: Hilbert space dimension.
    :return: Returns a dim by 1 vector drawn from the Haar measure.

    """
    unitary = haar_rand_unitary(dim)
    fiducial_vec = np.zeros((dim, 1))
    fiducial_vec[0] = 1
    return np.matmul(unitary, fiducial_vec)


def ginibre_state_matrix(dim: int, rank: int) -> np.ndarray:
    """
    Given a Hilbert space dimension dim and a desired rank K, returns a dim by dim positive
    semidefinite matrix of rank K drawn from the Ginibre ensemble.

    For dim = K these are states drawn from the Hilbert-Schmidt measure.

    See reference [IM]_ for more details.

    :param dim: Hilbert space dimension.
    :param rank: The rank of a state.
    :return: Returns a dim by rank matrix, drawn from the Ginibre ensemble.
    """
    if rank > dim:
        raise ValueError("The rank of the state matrix cannot exceed the dimension.")
    A = ginibre_matrix_complex(dim, rank)
    M = A.dot(np.transpose(np.conjugate(A)))
    return M / np.trace(M)


def bures_measure_state_matrix(dim: int) -> np.ndarray:
    """
    Given a Hilbert space dimension dim, returns a dim by dim positive semidefinite matrix drawn
    from the Bures measure [OSZ]_.

    .. [OSZ] Random Bures mixed states and the distribution of their purity.
          Osipov et al.
          J. Phys. A: Math. Theor. 43, 055302 (2010).
          https://doi.org/10.1088/1751-8113/43/5/055302
          https://arxiv.org/abs/0909.5094

    :param dim: Hilbert space dimension.
    :return: Returns a dim by dim matrix, drawn from the Bures measure.
    """
    A = ginibre_matrix_complex(dim, dim)
    U = haar_rand_unitary(dim)
    Udag = np.transpose(np.conjugate(U))
    Id = np.eye(dim)
    M = A.dot(np.transpose(np.conjugate(A)))
    P = (Id + U).dot(M).dot(Id + Udag)
    return P / np.trace(P)


def rand_map_with_BCSZ_dist(dim: int, kraus_rank: int) -> np.ndarray:
    """
    Given a Hilbert space dimension dim and a Kraus rank K, returns a dim^2 by dim^2 Choi
    matrix J(Λ) of a channel drawn from the BCSZ distribution with Kraus rank K [RQO]_.

    .. [RQO] Random quantum operations.
          Bruzda et al.
          Physics Letters A 373, 320 (2009).
          https://doi.org/10.1016/j.physleta.2008.11.043
          https://arxiv.org/abs/0804.2361
    
    :param dim: Hilbert space dimension.
    :param kraus_rank: The number of Kraus operators in the operator sum description of the channel.
    :return: dim^2 by dim^2 Choi matrix, drawn from the BCSZ distribution with Kraus rank K.
    """
    # TODO: this ^^ is CPTP, might want a flag that allows for just CP quantum operations.
    X = ginibre_matrix_complex(dim ** 2, kraus_rank)
    rho = X @ X.conj().T
    rho_red = partial_trace(rho, [0], [dim, dim])
    # Note that Eqn. 8 of [RQO] uses a *row* stacking convention so in that case we would write
    # Q = np.kron(np.eye(D), sqrtm(la.inv(rho_red)))
    # But as we use column stacking we need:
    Q = np.kron(sqrtm(la.inv(rho_red)), np.eye(dim))
    Z = Q @ rho @ Q
    return Z


def permute_tensor_factors(dims: Union[int, List[int]], perm: List[int]) -> np.ndarray:
    r"""
    Return a permutation matrix that appropriately swaps spaces of the given dimension(s).

    Given a Hilbert space dimension dim and an list representing the permutation perm of the
    tensor product Hilbert spaces, returns a dim**len(perm) by dim**len(perm) permutation matrix.
    
        1) Suppose dims=2 and perm=[0, 1]
            Returns the identity operator on two qubits
            
        2) Suppose dims=2 and perm=[1, 0]
            Returns the SWAP operator on two qubits which maps
            :math:`A_1 \otimes A_2 \rightarrow A_2 \otimes A_1`

        3) Suppose dims=[2, 4] and perm=[1, 0]
            Returns the SWAP operator on three qubits which maps
            :math:`A_1 \otimes (A_2 \otimes A_3) \rightarrow (A_2 \otimes A_3)`

    See: Equations 5.11, 5.12, and 5.13 in [SCOTT]_

    .. [SCOTT] Optimizing quantum process tomography with unitary 2-designs.
            A. J. Scott.
            J. Phys. A 41, 055308 (2008).
            https://dx.doi.org/10.1088/1751-8113/41/5/055308
            https://arxiv.org/abs/0711.1017

    This function is used in tests for other functions. However, it can also be useful when
    thinking about higher moment (N>2) integrals over the Haar measure.

    :param dims: The dimension of each Hilbert space factor given in order of the pre-permuted
        factorization of the total space. If an int is specified then each space in the
        permutation is assumed to have this dimension.
    :param perm: A list representing the permutation of the tensor factors.
    :return: a permutation matrix that permutes the factors of the given dimension.
    """
    if isinstance(dims, int):
        dim_list = [dims for _ in range(2 * len(perm))]
        total_dim = dims ** len(perm)
    else:
        assert len(dims) == len(perm), "Please specify the dimension of each factor to be permuted."
        dim_list = [dim for _ in range(2) for dim in dims]
        total_dim = np.prod(dims)

    transpositions = Permutation(perm).transpositions()

    # start with identity
    perm_matrix = np.eye(total_dim, total_dim)

    # reshape for easy implementation of transpositions
    perm_matrix = np.reshape(perm_matrix, dim_list)

    # build up the permutation one transposition at a time
    for swap in transpositions:
        perm_matrix = np.swapaxes(perm_matrix, swap[0], swap[1])

    # reshape to act on total_dim space
    return np.reshape(perm_matrix, [total_dim, total_dim])
