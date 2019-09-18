import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from lmfit import Model
from lmfit.model import ModelResult


def _check_data(x, y, weights):
    if not len(x) == len(y):
        raise ValueError("Lengths of x and y arrays must be equal.")
    if weights is not None and not len(x) == len(weights):
        raise ValueError("Lengths of x and weights arrays must be equal if weights is not None.")


def base_param_decay(x: np.ndarray, amplitude: float, decay: float, baseline: float):
    """
    Model an exponential decay parameterized by a base parameter raised to the power of the
    independent variable x.

    :param numpy.ndarray x: Independent variable
    :param float baseline: Offset value
    :param float amplitude: Amplitude of exponential decay
    :param float decay: Decay parameter
    :return: Exponential decay fit function
    """
    return np.asarray(baseline + amplitude * decay ** x)


def fit_base_param_decay(x: np.ndarray, y: np.ndarray, weights: np.ndarray = None,
                         param_guesses: tuple = (1., .9, 0.)) -> ModelResult:
    """
    Fit experimental data x, y to an exponential decay parameterized by a base decay parameter.

    :param x: The independent variable, e.g. depth or time
    :param y: The dependent variable, e.g. survival probability
    :param weights: Optional weightings of each point to use when fitting.
    :param param_guesses: initial guesses for the parameters
    :return: a lmfit Model
    """
    _check_data(x, y, weights)
    decay_model = Model(base_param_decay)
    params = decay_model.make_params(amplitude=param_guesses[0], decay=param_guesses[1],
                                     baseline=param_guesses[2])
    return decay_model.fit(y, x=x, params=params, weights=weights)


def decay_time_param_decay(x: np.ndarray, amplitude: float, decay_time: float,
                           offset: float = 0.0) -> np.ndarray:
    """
    Model an exponential decay parameterized by a decay constant with constant e as the base.

    :param x: The independent variable with respect to which decay is calculated.
    :param amplitude: The amplitude of the decay curve.
    :param decay_time: The inverse decay rate - e.g. T1 - of the decay curve.
    :param offset: The offset of the curve, (typically take to be 0.0).
    :return: Exponential decay fit function, parameterized with a decay constant
    """
    return np.asarray(amplitude * np.exp(-1 * (x - offset) / decay_time))


def fit_decay_time_param_decay(x: np.ndarray, y: np.ndarray, weights: np.ndarray = None,
                               param_guesses: tuple = (1., 10, 0)) -> ModelResult:
    """
    Fit experimental data x, y to an exponential decay parameterized by a decay time, or inverse
    decay rate.

    :param x: The independent variable, e.g. depth or time
    :param y: The dependent variable, e.g. survival probability
    :param weights: Optional weightings of each point to use when fitting.
    :param param_guesses: initial guesses for the parameters
    :return: a lmfit Model
    """
    _check_data(x, y, weights)
    decay_model = Model(decay_time_param_decay)
    params = decay_model.make_params(amplitude=param_guesses[0], decay_time=param_guesses[1],
                                     offset=param_guesses[2])
    return decay_model.fit(y, x=x, params=params, weights=weights)


def decaying_cosine(x: np.ndarray, amplitude: float, decay_time: float, offset: float,
                    baseline: float, frequency: float) -> np.ndarray:
    """
    Calculate exponentially decaying cosine at a series of points.

    :param x: The independent variable with respect to which decay is calculated.
    :param amplitude: The amplitude of the decaying cosine.
    :param decay_time: The inverse decay rate - e.g. T2 - of the decay curve.
    :param offset: The argument offset of the cosine, e.g. o for cos(x - o)
    :param baseline: The baseline of the cosine, e.g. b for cos(x) + b
    :param frequency: The frequency of the cosine, e.g. f for cos(2 pi f x). If decay_time is
        indeed a time then the frequency has units of inverse time.
    :return: The exponentially decaying cosine evaluated at the point(s) x
    """
    return amplitude * np.exp(-1 * x / decay_time) * np.cos(2 * pi * frequency * x + offset) + \
           baseline


def fit_decaying_cosine(x: np.ndarray, y: np.ndarray, weights: np.ndarray = None,
                        param_guesses: tuple = (.5, 10, 0.0, 0.5, 5)) -> ModelResult:
    """
    Fit experimental data x, y to an exponentially decaying cosine.

    :param x: The independent variable, e.g. depth or time
    :param y: The dependent variable, e.g. probability of measuring 1
    :param weights: Optional weightings of each point to use when fitting.
    :param param_guesses: initial guesses for the parameters
    :return: a lmfit Model
    """
    _check_data(x, y, weights)
    decay_model = Model(decaying_cosine)
    params = decay_model.make_params(amplitude=param_guesses[0], decay_time=param_guesses[1],
                                     offset=param_guesses[2], baseline=param_guesses[3],
                                     frequency=param_guesses[4])
    return decay_model.fit(y, x=x, params=params, weights=weights)


def shifted_cosine(x: np.ndarray, amplitude: float, offset: float, baseline: float,
                   frequency: float) -> np.ndarray:
    """
    Model for a cosine shifted vertically by the amount baseline.

    :param x: The independent variable;
    :param amplitude: The amplitude of the cosine.
    :param offset: The argument offset of the cosine, e.g. o for cos(x - o)
    :param baseline: The baseline of the cosine, e.g. b for cos(x) + b
    :param frequency: The angular frequency, e.g. f for cos(f x)
    :return: The sinusoidal response at the given phases(s).
    """
    return amplitude * np.cos(frequency * x + offset) + baseline


def fit_shifted_cosine(x: np.ndarray, y: np.ndarray, weights: np.ndarray = None,
                       param_guesses: tuple = (.5, 0, .5, 1.)) -> ModelResult:
    """
    Fit experimental data x, y to a cosine shifted vertically by amount baseline.

    :param x: The independent variable, e.g. depth or time
    :param y: The dependent variable, e.g. probability of measuring 1
    :param weights: Optional weightings of each point to use when fitting.
    :param param_guesses: initial guesses for the parameters
    :return: a lmfit Model
    """
    _check_data(x, y, weights)
    decay_model = Model(shifted_cosine)
    params = decay_model.make_params(amplitude=param_guesses[0], offset=param_guesses[1],
                                     baseline=param_guesses[2],
                                     frequency=param_guesses[3])
    return decay_model.fit(y, x=x, params=params, weights=weights)


def fit_result_to_json(fit_result):
    """
    Convert a fit result to a JSON-serializable dictionary.

    :param fit_result: (ModelResult) the result to serialize.
    :return: (dict)
    """
    # Parameters has a handy method to dump itself as a json string, but not as a JSON
    # dictionary.  We insert the string directly into the JSON dictionary because
    # 1) Parameters also has a loads() method that operates on this string, so we can recover the
    #    original parameters in this way
    # 2) The Python JSON parser has a bug in that it will happily convert a value of inf to a bare,
    #    unquoted value Infinity, which is not a valid JSON value and other parsers, like the one
    #    in Postgres will choke.  Keeping it as a string means it won't get parsed except by another
    #    Python instance when recreating the Parameters, which converts it back to the float inf.
    params_str = fit_result.params.dumps()
    json_dict = {
        "chisqr": fit_result.chisqr,
        "redchi": fit_result.redchi,
        "best_fit": fit_result.best_fit.tolist(),
        "best_values": fit_result.best_values,
        "covar": fit_result.covar.tolist() if fit_result.covar is not None else None,
        "params": params_str
    }
    # NOTE: aic and bic values are not included because they might be +/- infinity, which is not
    # valid JSON.  If we need them for some reason at a later date, we will need to explicitly
    # handle that case.
    return json_dict


TEAL = "#6CAFB7"
DARK_TEAL = '#48737F'
FUSCHIA = "#D6619E"
BEIGE = '#EAE8C6'
GRAY = '#494949'

FIT_PLOT_KWS = {
    'data_kws': {'color': 'black', 'marker': 'o', 'markersize': 4.0},
    'init_kws': {'color': TEAL, 'alpha': 0.4, 'linestyle': '--'},
    'fit_kws': {'color': TEAL, 'alpha': 1.0, 'linewidth': 2.0},
    'numpoints': 1000
}

DEFAULT_FIG_SIZE = (7, 10)
DEFAULT_AXIS_FONT_SIZE = 14
DEFAULT_REPORT_FONT_SIZE = 11


def plot_figure_for_fit(fit_result: ModelResult, xlabel: str = 'x', ylabel: str = 'y',
                        xscale: float = 1.0, yscale: float = 1.0, title: str = '',
                        figsize=DEFAULT_FIG_SIZE, axis_fontsize: tuple = DEFAULT_AXIS_FONT_SIZE,
                        report_fontsize: float = DEFAULT_REPORT_FONT_SIZE) -> plt.figure:
    """
    Plots fit and residuals from lmfit with residuals *below* fit.
    Also shows fit result text below.

    :param fit_result: lmfit fit result object
    :param xlabel: label for the shared x axis
    :param ylabel: ylabel for fit plot
    :param xscale: xaxis will be divided by xscale
    :param yscale: yaxis will be divided by yscale
    :param title: title of the plot
    :param figsize: size of the plot
    :param axis_fontsize: size of the font
    :param report_fontsize: size of font for the stats report
    :return: matplotlib figure
    """

    # layout subplots for fit plot and residuals
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True,
                            gridspec_kw={'height_ratios': (3, 1)},
                            figsize=figsize)

    # add space for fit result text at bottom
    plt.subplots_adjust(hspace=0, top=0.9, bottom=0.3)

    # plot the fits and residuals
    fit_result.plot_fit(ax=axs[0], **FIT_PLOT_KWS)
    fit_result.plot_residuals(ax=axs[1], data_kws=FIT_PLOT_KWS["data_kws"],
                              fit_kws=FIT_PLOT_KWS["fit_kws"])

    # title and labels
    axs[1].set_title('')
    axs[1].set_ylabel('residuals', fontsize=axis_fontsize)
    axs[1].set_xlabel(xlabel, fontsize=axis_fontsize)
    axs[0].set_ylabel(ylabel, fontsize=axis_fontsize)
    axs[0].set_title(title, fontsize=axis_fontsize)

    # residuals don't need a legend
    axs[1].legend().set_visible(False)

    # adjust tick labels for scales
    xticks = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / xscale))
    axs[1].xaxis.set_major_formatter(xticks)
    yticks = ticker.FuncFormatter(lambda y, pos: '{0:g}'.format(y / yscale))
    for ax in axs:
        ax.yaxis.set_major_formatter(yticks)

    # print fit report in space below plot, after dropping first two lines
    report = fit_result.fit_report(show_correl=False)
    report = ''.join(report.splitlines(True)[2:])
    fig.suptitle(report, fontsize=report_fontsize, family='monospace',
                 horizontalalignment='left', x=0.1, y=0.25)

    return fig, axs
