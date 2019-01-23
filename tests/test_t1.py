import numpy as np
import pytest

from forest_benchmarking.qubit_spectroscopy import fit_to_exponential_decay_curve


@pytest.fixture()
def mock_t1():
    num_points_sampled = 30
    true_T1 = 15e-6

    return {'T1': true_T1,
            'num_points': num_points_sampled}


def test_exponential_waveform(mock_t1):
    times = np.linspace(0, 2 * mock_t1['T1'], mock_t1['num_points'])
    data = np.asarray([np.exp(-1 * t / mock_t1['T1'])
                       for t in times])

    params, params_errs = fit_to_exponential_decay_curve(times, data)

    assert np.isclose(params[1], mock_t1['T1'])
