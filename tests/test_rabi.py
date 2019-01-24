import numpy as np
import pytest

from forest_benchmarking import qubit_spectroscopy as qs


@pytest.fixture()
def mock_rabi():
    actual_rabi_period = 2.15 * np.pi  # not quite 2pi

    return {'rabi_per': actual_rabi_period}


def test_sinusoidal_waveform(mock_rabi):
    thetas = np.linspace(0, 2 * np.pi, 30)
    data = np.asarray([0.5 * np.sin(theta * 2 * np.pi / mock_rabi['rabi_per']) + 0.5
                       for theta in thetas])

    params, params_errs = qs.fit_to_sinusoidal_waveform(thetas, data)

    assert np.isclose(2 * np.pi / params[2], mock_rabi['rabi_per'])
