import numpy as np
from numpy import pi
from forest.benchmarking.qubit_spectroscopy import generate_cz_phase_ramsey_experiment, \
    acquire_cz_phase_ramsey_data, fit_cz_phase_ramsey_results


def test_cz_ramsey(qvm):
    qubits = [0, 1]
    measure_q = qubits[0]
    num_shots = 50
    qvm.qam.random_seed = 1
    angles = np.linspace(0, 2 * pi, 15)
    ramsey_expt = [generate_cz_phase_ramsey_experiment(qubits, measure_q, angles)]
    result = acquire_cz_phase_ramsey_data(qvm, ramsey_expt, num_shots)[0]
    fit = fit_cz_phase_ramsey_results(result)

    freq = fit.params['frequency'].value
    freq_err = fit.params['frequency'].stderr

    assert (np.abs(freq - 1) < 2 * freq_err)

    amplitude = fit.params['amplitude'].value
    amplitude_err = fit.params['amplitude'].stderr

    assert (np.abs(amplitude - .5) < 2 * amplitude_err)

    baseline = fit.params['baseline'].value
    baseline_err = fit.params['baseline'].stderr

    assert (np.abs(baseline - .5) < 2 * baseline_err)
