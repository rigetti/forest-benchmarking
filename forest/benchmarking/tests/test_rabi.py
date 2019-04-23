import numpy as np
from numpy import pi
from forest.benchmarking.qubit_spectroscopy import generate_rabi_experiment, acquire_rabi_data, \
    fit_rabi_results


def test_rabi_flop(qvm):
    qubits = [0]
    num_shots = 50
    qvm.qam.random_seed = 1
    angles = np.linspace(0, 2 * pi, 15)
    rabi_expts = [generate_rabi_experiment(qubit, angles) for qubit in qubits]
    result = acquire_rabi_data(qvm, rabi_expts, num_shots)[0]
    fit = fit_rabi_results(result)

    freq = fit.params['frequency'].value
    freq_err = fit.params['frequency'].stderr

    assert (np.abs(freq - 1) < 2 * freq_err)

    amplitude = fit.params['amplitude'].value
    amplitude_err = fit.params['amplitude'].stderr

    assert (np.abs(amplitude - .5) < 2 * amplitude_err)

    baseline = fit.params['baseline'].value
    baseline_err = fit.params['baseline'].stderr

    assert (np.abs(baseline - .5) < 2 * baseline_err)
