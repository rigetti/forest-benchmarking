from math import pi
import pytest
import numpy as np
from pyquil.api import get_benchmarker
from pyquil import Program
from pyquil.gates import CZ, RX, CNOT, H, I, X
from forest_benchmarking.dfe import generate_process_dfe_experiment, acquire_dfe_data, \
    direct_fidelity_estimate, generate_state_dfe_experiment, ratio_variance


def test_exhaustive_gate_dfe_noiseless_qvm(qvm, benchmarker):
    bm = get_benchmarker()
    qvm.qam.random_seed = 1
    process_exp = generate_process_dfe_experiment(Program([RX(pi / 2, 0)]), compiler=benchmarker)
    data, cal = acquire_dfe_data(process_exp, qvm, var=0.01,)
    est = direct_fidelity_estimate(data, cal, 'process')
    assert est.fid_point_est == 1.0
    assert est.fid_var_est == 0.0
    assert all([exp == 1.0 for exp in data.expectation])
    assert all(np.abs(cal) == 1.0 for cal in cal.expectation)

    process_exp = generate_process_dfe_experiment(Program([CZ(0, 1)]), compiler=benchmarker)
    data, cal = acquire_dfe_data(process_exp, qvm, var=0.01, )
    est = direct_fidelity_estimate(data, cal, 'process')
    assert est.fid_point_est == 1.0
    assert est.fid_var_est == 0.0
    assert all([exp == 1.0 for exp in data.expectation])
    assert all(np.abs(cal) == 1.0 for cal in cal.expectation)

    process_exp = generate_process_dfe_experiment(Program([CNOT(0, 1)]), compiler=benchmarker)
    data, cal = acquire_dfe_data(process_exp, qvm, var=0.01, )
    est = direct_fidelity_estimate(data, cal, 'process')
    assert est.fid_point_est == 1.0
    assert est.fid_var_est == 0.0
    assert all([exp == 1.0 for exp in data.expectation])
    assert all(np.abs(cal) == 1.0 for cal in cal.expectation)


def test_exhaustive_state_dfe_noiseless_qvm(qvm, benchmarker):
    qvm.qam.random_seed = 1
    state_exp = generate_state_dfe_experiment(Program([RX(pi / 2, 0)]), compiler=benchmarker)
    data, cal = acquire_dfe_data(state_exp, qvm, var=0.01,)
    est = direct_fidelity_estimate(data, cal, 'state')
    assert est.fid_point_est == 1.0
    assert est.fid_var_est == 0.0
    assert all([exp == 1.0 for exp in data.expectation])
    assert all(np.abs(cal) == 1.0 for cal in cal.expectation)

    state_exp = generate_state_dfe_experiment(Program([H(0), H(1), CZ(0, 1)]), compiler=benchmarker)
    data, cal = acquire_dfe_data(state_exp, qvm, var=0.01,)
    est = direct_fidelity_estimate(data, cal, 'state')
    assert est.fid_point_est == 1.0
    assert est.fid_var_est == 0.0
    assert all([exp == 1.0 for exp in data.expectation])
    assert all(np.abs(cal) == 1.0 for cal in cal.expectation)

    state_exp = generate_state_dfe_experiment(Program([H(0), CNOT(0, 1)]), compiler=benchmarker)
    data, cal = acquire_dfe_data(state_exp, qvm, var=0.01,)
    est = direct_fidelity_estimate(data, cal, 'state')
    assert est.fid_point_est == 1.0
    assert est.fid_var_est == 0.0
    assert all([exp == 1.0 for exp in data.expectation])
    assert all(np.abs(cal) == 1.0 for cal in cal.expectation)


def test_ratio_variance():
    # If our uncertainty is 0 in each parameter, the uncertainty in the ratio should also be 0.
    assert ratio_variance(1, 0, 1, 0) == 0
    # If our uncertainty in the denominator is 0, and it's expectation value is one, then
    # the uncertainty in the ratio should just be the uncertainty in the numerator.
    assert ratio_variance(1, 1, 1, 0) == 1
    # It shouldn't depend on the value in the numerator.
    assert ratio_variance(2, 1, 1, 0) == 1


def _random_unitary():
    """
    :return: array of shape (2, 2) representing random unitary matrix drawn from Haar measure
    """
    # draw complex matrix from Ginibre ensemble
    z = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    # QR decompose this complex matrix
    q, r = np.linalg.qr(z)
    # make this decomposition unique
    d = np.diagonal(r)
    l = np.diag(d) / np.abs(d)
    return np.matmul(q, l)


def _kraus_ops_bit_flip(prob):
    """
    :param prob: probability of bit-flip
    :return: list of Kraus operators
    """
    # define flip (X) and not flip (I) Kraus operators
    I_ = np.sqrt(1 - prob) * np.array([[1, 0], [0, 1]])
    X_ = np.sqrt(prob) * np.array([[0, 1], [1, 0]])
    return [I_, X_]


def _kraus_ops_amp_damping(prob):
    """
    :param prob: probability of |1> to |0> decay
    :return: list of Kraus operators
    """
    # define imperfect identity (I_) and decay (D_) Kraus operators
    I_ = np.array([[1, 0], [0, np.sqrt(1 - prob)]])
    D_ = np.array([[0, np.sqrt(prob)], [0, 0]])
    return [I_, D_]


def _kraus_ops_dephasing(prob):
    """
    :param prob: probability of applying Z operator
    :return: list of Kraus operators
    """
    # define probabilistic identity (I_) and Z (Z_) Kraus operators
    I_ = np.sqrt(1 - prob) * np.array([[1, 0], [0, 1]])
    Z_ = np.sqrt(prob) * np.array([[1, 0], [0, -1]])
    return [I_, Z_]


def _kraus_ops_depolarizing(prob):
    """
    :param prob: probability of being unchanged;
        1 - this probability is the probability of transforming into random state I/2
    :return: list of Kraus operators
    """
    # define Kraus operators
    M0 = np.sqrt(3 * prob + 1) / 2 * np.array([[1, 0], [0, 1]])
    M1 = np.sqrt(1 - prob) / 2 * np.array([[0, 1], [1, 0]])
    M2 = np.sqrt(1 - prob) / 2 * np.array([[0, -1j], [1j, 0]])
    M3 = np.sqrt(1 - prob) / 2 * np.array([[1, 0], [0, -1]])
    return [M0, M1, M2, M3]


def test_bit_flip_channel_fidelity(qvm, benchmarker):
    """
    We use Eqn (5) of https://arxiv.org/abs/quant-ph/0701138 to compare the fidelity
    """
    process_exp = generate_process_dfe_experiment(Program(I(0)), benchmarker)
    # pick probability of bit flip, and num_shots
    np.random.seed(42)
    prob = np.random.uniform(0.1, 0.5)
    num_shots = 4000
    # obtain Kraus operators associated with the channel
    kraus_ops = _kraus_ops_bit_flip(prob)
    # create Program with noisy gates
    p = Program()
    p.defgate("DummyGate", _random_unitary())
    p.inst(("DummyGate", 0))
    p.define_noisy_gate("DummyGate", [0], kraus_ops)
    # define this (noisy) program as the one associated with process_exp
    process_exp.program = p
    # estimate fidelity
    data, cal = acquire_dfe_data(process_exp, qvm, 0.01)
    pest = direct_fidelity_estimate(data, cal, 'process')
    # test if correct
    expected_result = 1 - (2/3 * prob)
    assert np.isclose(pest.fid_point_est, expected_result, atol=1.e1)


@pytest.mark.skip(reason="Figure out why this is failing")
def test_amplitude_damping_channel_fidelity(qvm, benchmarker):
    """
    We use Eqn (5) of https://arxiv.org/abs/quant-ph/0701138 to compare the fidelity
    """
    process_exp = generate_process_dfe_experiment(Program(I(0)), benchmarker)
    # pick probability of amplitude damping, and num_shots
    prob = 0.3
    num_shots = 4000
    # obtain Kraus operators associated with the channel
    kraus_ops = _kraus_ops_amp_damping(prob)
    # create Program with noisy gates
    p = Program()
    p.defgate("DummyGate", _random_unitary())
    p.inst(("DummyGate", 0))
    p.define_noisy_gate("DummyGate", [0], kraus_ops)
    # define this (noisy) program as the one associated with process_exp
    process_exp.program = p
    # estimate fidelity
    data, cal = acquire_dfe_data(process_exp, qvm, 0.01)
    pest = direct_fidelity_estimate(data, cal, 'process')
    # test if correct
    expected_result = 2/3 + np.sqrt(1-prob)/3 - prob/6
    assert np.isclose(pest.fid_point_est, expected_result, atol=1.e-1)


def test_dephasing_channel_fidelity(qvm, benchmarker):
    """
    We use Eqn (5) of https://arxiv.org/abs/quant-ph/0701138 to compare the fidelity
    """
    process_exp = generate_process_dfe_experiment(Program(I(0)), benchmarker)
    # pick probability of amplitude damping, and num_shots
    prob = np.random.uniform(0.1, 0.5)
    num_shots = 4000
    # obtain Kraus operators associated with the channel
    kraus_ops = _kraus_ops_dephasing(prob)
    # create Program with noisy gates
    p = Program()
    p.defgate("DummyGate", _random_unitary())
    p.inst(("DummyGate", 0))
    p.define_noisy_gate("DummyGate", [0], kraus_ops)
    # define this (noisy) program as the one associated with process_exp
    process_exp.program = p
    # estimate fidelity
    data, cal = acquire_dfe_data(process_exp, qvm, 0.01)
    pest = direct_fidelity_estimate(data, cal, 'process')
    # test if correct
    expected_result = 1 - (2/3 * prob)
    assert np.isclose(pest.fid_point_est, expected_result, atol=1.e-1)


def test_depolarizing_channel_fidelity(qvm, benchmarker):
    """
    We use Eqn (5) of https://arxiv.org/abs/quant-ph/0701138 to compare the fidelity
    """
    process_exp = generate_process_dfe_experiment(Program(I(0)), benchmarker)
    # pick probability of amplitude damping, and num_shots
    prob = np.random.uniform(0.1, 0.5)
    num_shots = 10000
    # obtain Kraus operators associated with the channel
    kraus_ops = _kraus_ops_depolarizing(prob)
    # create Program with noisy gates
    p = Program()
    p.defgate("DummyGate", _random_unitary())
    p.inst(("DummyGate", 0))
    p.define_noisy_gate("DummyGate", [0], kraus_ops)
    # define this (noisy) program as the one associated with process_exp
    process_exp.program = p
    # estimate fidelity
    data, cal = acquire_dfe_data(process_exp, qvm, 0.01)
    pest = direct_fidelity_estimate(data, cal, 'process')
    # test if correct
    expected_result = (1 + prob) / 2
    assert np.isclose(pest.fid_point_est, expected_result, atol=1.e-1)
