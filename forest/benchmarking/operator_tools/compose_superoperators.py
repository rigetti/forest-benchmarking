"""A module containing tools for composing superoperators.
"""
from typing import Sequence
import numpy as np


def tensor_channel_kraus(k2: Sequence[np.ndarray],
                         k1: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
    r"""
    Given the Kraus representaion for two channels, $\mathcal E$ and $\mathcal F$, acting on
    different systems this function returns the Kraus operators representing the composition of
    these independent channels.

    Suppose $\mathcal E$ and $\mathcal F$ both have one Kraus operator K_1 = X and K_2 = H,
    that is they are unitary. Then, with respect to the tensor product structure

            $H_2 \otimes H_1$

    of the individual systems this function returns $K_{\rm tot} = H \otimes X$.

    :param k1: The list of Kraus operators on the first system.
    :param k2: The list of Kraus operators on the second system.
    :return: A list of tensored Kraus operators.
    """
    # TODO: make this function work for an arbitrary number of Kraus operators
    return [np.kron(k2l, k1j) for k1j in k1 for k2l in k2]


def compose_channel_kraus(k2: Sequence[np.ndarray],
                          k1: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
    r"""
    Given two channels, K_1 and K_2, acting on the same system in the Kraus representation this
    function return the Kraus operators representing the composition of the channels.

    It is assumed that K_1 is applied first then K_2 is applied.

    :param k2: The list of Kraus operators that are applied second.
    :param k1: The list of Kraus operators that are applied first.
    :return: A combinatorially generated list of composed Kraus operators.
    """
    # TODO: make this function work for an arbitrary number of Kraus operators
    return [np.dot(k2l, k1j) for k1j in k1 for k2l in k2]
