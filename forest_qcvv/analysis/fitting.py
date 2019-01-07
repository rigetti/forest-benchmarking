import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from lmfit.model import ModelResult


def fit_result_to_json(fit_result):
    """
    Convert a fit result to a JSON-serializable dictionary.

    :param fit_result: (ModelResult) the result to serialize.
    :return: (dict)
    """
    # annoyingly, Parameters has a handy method to dump itself as a json string, but not as a JSON
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


def make_figure(fit_result: ModelResult, xlabel='x', ylabel='y', xscale=1.0, yscale=1.0, title='',
                figsize=DEFAULT_FIG_SIZE, axis_fontsize=DEFAULT_AXIS_FONT_SIZE,
                report_fontsize=DEFAULT_REPORT_FONT_SIZE):
    """
    Plots fit and residuals from lmfit with residuals *below* fit.
    Also shows fit result text below.

    :param lmfit.ModelResult fit_result: lmfit fit result object
    :param string xlabel: label for the shared x axis
    :param string ylabel: ylabel for fit plot
    :param float xscale: xaxis will be divided by xscale
    :param float yscale: yaxis will be divided by yscale
    :param str title: title of the plot
    :param tuple figsize: size of the plot
    :param float fontsize: size of the font

    :return: matplotlib figure
    :rtype: matplotlib.pyplot.figure
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
