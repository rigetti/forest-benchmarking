import numpy as np
import functools
from scipy.linalg import eigh


def project_state_matrix_to_physical(rho: np.ndarray) -> np.ndarray:
    """
    Project a possibly unphysical estimated density matrix to the closest (with respect to the
    2-norm) positive semi-definite matrix with trace 1, that is a valid quantum state.

    This comes from the so called "wizard" method. It is described in [MLEWIZ]_

    .. [MLEWIZ] Efficient Method for Computing the Maximum-Likelihood Quantum State from
             Measurements with Additive Gaussian Noise.
             Smolin et al.
             Phys. Rev. Lett. 108, 070502 (2012).
             https://doi.org/10.1103/PhysRevLett.108.070502
             https://arxiv.org/abs/1106.5458

    :param rho: the density (state) matrix with shape (N, N)
    :return rho_projected: The closest positive semi-definite trace 1 matrix to rho.
    """
    # Rescale to trace 1 if the matrix is not already
    rho_impure = rho / np.trace(rho)

    dimension = rho_impure.shape[0]  # the dimension of the Hilbert space
    [eigvals, eigvecs] = eigh(rho_impure)

    # If matrix is already trace one PSD, we are done
    if np.min(eigvals) >= 0:
        return rho_impure

    # Otherwise, continue finding closest trace one, PSD matrix
    eigvals = list(eigvals)
    eigvals.reverse()
    eigvals_new = [0.0] * len(eigvals)

    i = dimension
    accumulator = 0.0  # Accumulator
    while eigvals[i - 1] + accumulator / float(i) < 0:
        accumulator += eigvals[i - 1]
        i -= 1
    for j in range(i):
        eigvals_new[j] = eigvals[j] + accumulator / float(i)
    eigvals_new.reverse()

    # Reconstruct the matrix
    rho_projected = functools.reduce(np.dot, (eigvecs,
                                              np.diag(eigvals_new),
                                              np.conj(eigvecs.T)))

    return rho_projected
