import numpy as np

from forest.benchmarking.analysis.fitting import fit_shifted_cosine, \
    fit_decay_constant_param_decay, fit_decaying_sinusoid


def test_shifted_cosine():
    actual_rabi_period = 2.15 * np.pi  # not quite 2pi
    mock_rabi = {'rabi_per': actual_rabi_period}
    thetas = np.linspace(0, 2 * np.pi, 30)
    data = np.asarray([- 0.5 * np.cos(theta * 2 * np.pi / mock_rabi['rabi_per']) + 0.5
                       for theta in thetas])

    fit = fit_shifted_cosine(thetas, data)

    assert np.isclose(2 * np.pi / fit.params['frequency'], mock_rabi['rabi_per'])


def test_fit_decay_constant_param_decay():
    num_points_sampled = 30
    true_t1 = 15  # us
    mock_t1 = {'T1': true_t1, 'num_points': num_points_sampled}

    times = np.linspace(0, 2 * mock_t1['T1'], mock_t1['num_points'])
    data = np.asarray([np.exp(-1 * t / mock_t1['T1']) for t in times])

    fit = fit_decay_constant_param_decay(times, data)

    assert np.isclose(fit.params['decay_constant'], mock_t1['T1'])


def test_decaying_sinusoid():
    num_points_sampled = 50
    true_t2 = 15  # us
    qubit_detuning = 2.5  # MHZ

    mock_t2 = {'T2': true_t2, 'qubit_detuning': qubit_detuning, 'num_points': num_points_sampled}

    times = np.linspace(0, 2 * mock_t2['T2'], mock_t2['num_points'])
    data = np.asarray([0.5 * np.exp(-1 * t / mock_t2['T2']) *
                       np.sin(mock_t2['qubit_detuning'] * t) + 0.5 for t in times])

    fit = fit_decaying_sinusoid(times, data)

    assert np.isclose(fit.params['decay_constant'], mock_t2['T2'])
    assert np.isclose(fit.params['frequency'], mock_t2['qubit_detuning'])

