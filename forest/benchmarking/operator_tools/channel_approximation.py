"""A module containing tools for approximating channels.

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
import numpy as np


def pauli_twirl_chi_matrix(chi_matrix: np.ndarray) -> np.ndarray:
    """
    Implements a Pauli twirl of a chi matrix (aka a process matrix).

    See [SPICC]_ for more details

    .. [SPICC] Scalable protocol for identification of correctable codes.
            Silva et al.
            PRA 78, 012347 2008.
            http://dx.doi.org/10.1103/PhysRevA.78.012347
            https://arxiv.org/abs/0710.1900

    Note: Pauli twirling a quantum channel can give rise to a channel that is less noisy; use with
    care.

    :param chi_matrix:  a dim**2 by dim**2 chi or process matrix
    :return: dim**2 by dim**2 chi or process matrix
    """
    return np.diag(chi_matrix.diagonal())


# TODO: Honest approximations for Channels that act on one or MORE qubits.
