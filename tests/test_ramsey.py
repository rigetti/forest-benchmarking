import numpy as np
import pytest

from forest_benchmarking import qubit_spectroscopy as qs


@pytest.fixture()
def mock_ramsey():
    return {'fixed_rz': 0.03,
            'tunable_rz': 0.51}


def test_sinusoidal_waveform(mock_ramsey):
    thetas = np.linspace(0, 2 * np.pi, 30)

    fixed_data = np.asarray([0.5 * np.sin(theta + mock_ramsey['fixed_rz']) + 0.5
                             for theta in thetas])
    params, params_errs = qs.fit_to_sinusoidal_waveform(thetas, fixed_data)
    assert np.isclose(params[3], mock_ramsey['fixed_rz'])

    tunable_data = np.asarray([0.5 * np.sin(theta + mock_ramsey['tunable_rz']) + 0.5
                               for theta in thetas])
    params, params_errs = qs.fit_to_sinusoidal_waveform(thetas, tunable_data)
    assert np.isclose(params[3], mock_ramsey['tunable_rz'])
