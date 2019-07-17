import numpy as np
from forest.benchmarking.operator_tools.channel_approximation import pauli_twirl_chi_matrix
from forest.benchmarking.tests.test_superoperator_transformations import (one_q_pauli_channel_chi,
                                                                          amplitude_damping_chi)


# Pauli twirled Amplitude damping channel
def analytical_pauli_twirl_of_AD_chi(p):
    # see equation 7 of  https://arxiv.org/pdf/1701.03708.pdf
    poly1 = (2 + 2 * np.sqrt(1 - p) - p) / 4
    poly2 = p / 4
    poly3 = (2 - 2 * np.sqrt(1 - p) - p) / 4
    pp_chi = np.asarray([[poly1, 0, 0, 0],
                         [0, poly2, 0, 0],
                         [0, 0, poly2, 0],
                         [0, 0, 0, poly3]])
    return pp_chi


def test_pauli_twirl_of_pauli_channel():
    # diagonal channel so should not change anything
    px = np.random.rand()
    py = np.random.rand()
    pz = np.random.rand()
    pauli_chan_chi_matrix = one_q_pauli_channel_chi(px, py, pz)
    pauli_twirled_chi_matrix = pauli_twirl_chi_matrix(pauli_chan_chi_matrix)
    assert np.allclose(pauli_chan_chi_matrix, pauli_twirled_chi_matrix)


def test_pauli_twirl_of_amp_damp():
    p = np.random.rand()
    ana_chi = analytical_pauli_twirl_of_AD_chi(p)
    num_chi = pauli_twirl_chi_matrix(amplitude_damping_chi(p))
    assert np.allclose(ana_chi, num_chi)
