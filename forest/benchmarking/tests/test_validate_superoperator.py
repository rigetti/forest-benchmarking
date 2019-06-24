from pyquil.gate_matrices import H
import forest.benchmarking.operator_tools.random_operators as rand_ops
from forest.benchmarking.operator_tools.superoperator_transformations import chi2choi
from forest.benchmarking.tests.test_superoperator_transformations import (
    amplitude_damping_kraus, AdKrausOps, HADChoi, amplitude_damping_choi, one_q_pauli_channel_chi)
from forest.benchmarking.operator_tools.validate_superoperator import *


def test_kraus_operators_are_valid():
    assert kraus_operators_are_valid(amplitude_damping_kraus(np.random.rand()))
    assert kraus_operators_are_valid(H)
    assert not kraus_operators_are_valid(AdKrausOps[0])


def test_choi_is_hermitian_preserving():
    D = 2
    K = 2
    choi = rand_ops.rand_map_with_BCSZ_dist(D, K)
    assert choi_is_hermitian_preserving(choi)


def test_choi_is_trace_preserving():
    D = 2
    K = 2
    choi = rand_ops.rand_map_with_BCSZ_dist(D, K)
    assert choi_is_trace_preserving(choi)


def test_choi_is_completely_positive():
    D = 2
    K = 2
    choi = rand_ops.rand_map_with_BCSZ_dist(D, K)
    assert choi_is_completely_positive(choi)
    D = 3
    K = 2
    choi = rand_ops.rand_map_with_BCSZ_dist(D, K)
    assert choi_is_completely_positive(choi)


def test_choi_is_unital():
    px = np.random.rand()
    py = np.random.rand()
    pz = np.random.rand()
    norm = px + py + pz
    choi = chi2choi(one_q_pauli_channel_chi(px/norm, py/norm, pz/norm))
    assert choi_is_unital(choi)
    assert choi_is_unital(HADChoi)
    assert not choi_is_unital(amplitude_damping_choi(0.1))


def test_choi_is_unitary():
    px = np.random.rand()
    py = np.random.rand()
    pz = np.random.rand()
    norm = px + py + pz
    choi = chi2choi(one_q_pauli_channel_chi(px/norm, py/norm, pz/norm))
    assert not choi_is_unitary(choi)
    assert choi_is_unital(choi)
    assert choi_is_unitary(HADChoi)
    assert not choi_is_unitary(amplitude_damping_choi(0.1))
