from math import pi

import numpy as np

from pyquil import Program
from pyquil.gates import CZ, RX, CNOT, H
from forest_benchmarking.dfe import generate_process_dfe_experiment, acquire_dfe_data, \
    direct_fidelity_estimate, generate_state_dfe_experiment, ratio_variance

def test_exhaustive_gate_dfe_noiseless_qvm(qvm, benchmarker):
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
