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
# Check physicality of Channels
# ==================================================================================================
def kraus_operators_are_valid(kraus_ops: Sequence[np.ndarray],
                              rtol: float = 1e-05,
                              atol: float = 1e-08) -> bool:
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
