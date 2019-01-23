import numpy as np
import pytest

from forest_benchmarking.qubit_spectroscopy import fit_to_exponentially_decaying_sinusoidal_curve


@pytest.fixture()
def mock_t2():
    num_points_sampled = 30
    true_T2 = 15e-6
    QUBIT_DETUNING = 2.5e6

    return {'T2': true_T2,
            'qubit_detuning': QUBIT_DETUNING,
            'num_points': num_points_sampled}


def test_exponentially_decaying_sinusoidal_waveform(mock_t2):
    times = np.linspace(0, 2 * mock_t2['T2'], mock_t2['num_points'])
    data = np.asarray([0.5 * np.exp(-1 * t / mock_t2['T2']) * \
                       np.sin(mock_t2['qubit_detuning'] * t) + 0.5
                       for t in times])

    params, params_errs = fit_to_exponentially_decaying_sinusoidal_curve(times, data)

    assert np.isclose(params[1], mock_t2['T2'])
