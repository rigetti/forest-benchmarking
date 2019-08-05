"""A module containing tools for composing superoperators.
"""
from typing import Sequence
import numpy as np


def tensor_channel_kraus(k2: Sequence[np.ndarray],
                         k1: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
    r"""
    Given the Kraus representation for two channels, :math:`\mathcal E` and :math:`\mathcal F`,
    acting on different systems this function returns the Kraus operators representing the
    composition of these independent channels.

    Suppose :math:`\mathcal E` and :math:`\mathcal F` each have one Kraus operator,
    :math:`K_1 = X` and :math:`K_2 = H`, so each channel is unitary. Then, with respect to the
    tensor product structure :math:`H_2 \otimes H_1` of the individual systems, this function
    returns

    .. math::

        K_{\rm tot} = H \otimes X

    :param k1: The list of Kraus operators on the first system.
    :param k2: The list of Kraus operators on the second system.
    :return: A list of tensored Kraus operators.
    """
    # TODO: make this function work for an arbitrary number of Kraus operators
    return [np.kron(k2l, k1j) for k1j in k1 for k2l in k2]


def compose_channel_kraus(k2: Sequence[np.ndarray],
                          k1: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
    """
    Given two channels, K_1 and K_2, acting on the same system in the Kraus representation this
    function return the Kraus operators representing the composition of the channels.

    It is assumed that K_1 is applied first then K_2 is applied.

    :param k2: The list of Kraus operators that are applied second.
    :param k1: The list of Kraus operators that are applied first.
    :return: A combinatorially generated list of composed Kraus operators.
    """
    # TODO: make this function work for an arbitrary number of Kraus operators
    return [np.dot(k2l, k1j) for k1j in k1 for k2l in k2]
