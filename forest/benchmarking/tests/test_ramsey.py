import numpy as np
from numpy import pi
from forest.benchmarking.qubit_spectroscopy import (generate_cz_phase_ramsey_experiments,
                                                    acquire_qubit_spectroscopy_data,
                                                    fit_cz_phase_ramsey_results,
                                                    get_stats_by_qubit)


def test_cz_ramsey(qvm):
    qubits = [0, 1]
    measure_q = qubits[0]
    num_shots = 100
    qvm.qam.random_seed = 1
    angles = np.linspace(0, 2 * pi, 15)
    ramsey_expt = generate_cz_phase_ramsey_experiments(qubits, measure_q, angles)
    results = acquire_qubit_spectroscopy_data(qvm, ramsey_expt, num_shots)
    stats = get_stats_by_qubit(results)[measure_q]

    fit = fit_cz_phase_ramsey_results(angles, stats['expectation'], stats['std_err'])

    freq = fit.params['frequency'].value
    freq_err = fit.params['frequency'].stderr

    assert np.isclose(freq, 1, atol=2 * freq_err)

    amplitude = fit.params['amplitude'].value
    amplitude_err = fit.params['amplitude'].stderr

    assert np.isclose(amplitude, .5, atol=2 * amplitude_err)

    baseline = fit.params['baseline'].value
    baseline_err = fit.params['baseline'].stderr

    assert np.isclose(baseline, .5, atol=2 * baseline_err)
