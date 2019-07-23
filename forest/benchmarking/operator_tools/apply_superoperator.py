"""A module containing tools for applying superoperators to states.

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
from typing import Sequence
import numpy as np
from forest.benchmarking.operator_tools.calculational import partial_trace


def apply_kraus_ops_2_state(kraus_ops: Sequence[np.ndarray], state: np.ndarray) -> np.ndarray:
    r"""
    Apply a quantum channel, specified by Kraus operators, to state.

    The Kraus operators need not be square.

    :param kraus_ops: A list or tuple of N Kraus operators, each operator is M by dim ndarray
    :param state: A dim by dim ndarray which is the density matrix for the state
    :return: M by M ndarray which is the density matrix for the state after the action of kraus_ops
    """
    if isinstance(kraus_ops, np.ndarray):  # handle input of single kraus op
        if len(kraus_ops[0].shape) < 2:  # first elem is not a matrix
            kraus_ops = [kraus_ops]

    dim, _ = state.shape
    rows, cols = kraus_ops[0].shape

    if dim != cols:
        raise ValueError("Dimensions of state and Kraus operator are incompatible")

    new_state = np.zeros((rows, rows))
    for M in kraus_ops:
        new_state += M @ state @ np.transpose(M.conj())

    return new_state


def apply_choi_matrix_2_state(choi: np.ndarray, state: np.ndarray) -> np.ndarray:
    r"""
    Apply a quantum channel, specified by a Choi matrix (using the column stacking convention),
    to a state.

    The Choi matrix is a dim**2 by dim**2 matrix and the state rho is a dim by dim matrix. The
    output state is

    .. math::

        \rho_{out} = Tr_{A_{in}}[(\rho^T \otimes Id) \rm{Choi\_matrix} ],

    where :math:`^T` denotes transposition and :math:`Tr_{A_{in}}` is the partial trace over input
    Hilbert space :math:`H_{A_{in}}`

    The Choi matrix representing a process mapping

     .. math::

        \rho_{in} \in H_{A_{in}} \rightarrow \rho_{out} \in H_{B_{out}}

    is regarded as an operator on the space :math:`H_{A_{in}} \otimes H_{B_{out}}`.

    :param choi: a dim**2 by dim**2 matrix
    :param state: A dim by dim ndarray which is the density matrix for the state
    :return: a dim by dim matrix.
    """
    dim = int(np.sqrt(np.asarray(choi).shape[0]))
    dims = [dim, dim]
    tot_matrix = np.kron(state.transpose(), np.identity(dim)) @ choi
    return partial_trace(tot_matrix, [1], dims)
