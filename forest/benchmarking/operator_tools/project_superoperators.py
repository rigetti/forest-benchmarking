"""A module containing tools for projecting superoperators to CP, TNI, TP, and physical.

We have arbitrarily decided to use a column stacking convention.

A good reference for these methods is:

[PGD] Maximum-likelihood quantum process tomography via projected gradient descent
      Knee et al.,
      Phys. Rev. A 98, 062336 (2018)
      https://dx.doi.org/10.1103/PhysRevA.98.062336
      https://arxiv.org/abs/1803.10062
"""
import numpy as np
from forest.benchmarking.utils import partial_trace
from forest.benchmarking.operator_tools.superoperator_transformations import vec


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


def proj_choi_to_physical(choi: np.ndarray, make_trace_preserving: bool = True) -> np.ndarray:
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
