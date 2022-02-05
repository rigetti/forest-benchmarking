import numpy as np
from pyquil.simulation.matrices import I, X, H

from forest.benchmarking.operator_tools.compose_superoperators import (compose_channel_kraus,
                                                                       tensor_channel_kraus)
from forest.benchmarking.operator_tools.superoperator_transformations import kraus2superop
from forest.benchmarking.tests.test_superoperator_transformations import amplitude_damping_kraus


def bit_flip_kraus(p):
    M0 = np.sqrt(1 - p) * I
    M1 = np.sqrt(p) * X
    return [M0, M1]


AD_kraus = amplitude_damping_kraus(0.1)
BitFlip_kraus = bit_flip_kraus(0.2)
BitFlip_super = kraus2superop(BitFlip_kraus)
AD_super = kraus2superop(AD_kraus)


def test_compose_channel_kraus():
    function_output = kraus2superop(compose_channel_kraus(AD_kraus, BitFlip_kraus))
    independent_answer = AD_super @ BitFlip_super
    assert np.allclose(function_output, independent_answer)


def test_tensor_channel_kraus():
    function_output = tensor_channel_kraus([X], [H])
    independent_answer = np.kron(X,H)
    wrong_answer = np.kron(H,X)
    assert np.allclose(function_output, independent_answer)
    assert not np.allclose(function_output, wrong_answer)
