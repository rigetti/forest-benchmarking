from forest.benchmarking.volumetrics._main import get_random_hamming_wt_distr
from typing import Sequence, Dict
import numpy as np
import matplotlib.pyplot as plt


def plot_error_distributions(distr_arr: Dict[int, Dict[int, Sequence[float]]], widths=None,
                             depths=None, plot_rand_distr=False):
    """
    For each width and depth plot the distribution of errors provided in distr_arr.

    :param distr_arr:
    :param widths:
    :param depths:
    :param plot_rand_distr:
    :return:
    """
    if widths is None:
        widths = list(distr_arr.keys())

    if depths is None:
        depths = list(list(distr_arr.values())[0].keys())

    legend = ['data']
    if plot_rand_distr:
        legend.append('random')

    fig = plt.figure(figsize=(18, 6 * len(depths)))
    axs = fig.subplots(len(depths), len(widths), sharex='col', sharey=True)

    for w_idx, w in enumerate(widths):
        x_labels = np.arange(0, w + 1)
        depth_distrs = distr_arr[w]

        if plot_rand_distr:
            rand_distr = get_random_hamming_wt_distr(w)

        for d_idx, d in enumerate(depths):
            distr = depth_distrs[d]

            idx = d_idx * len(widths) + w_idx
            if len(widths) == len(depths) == 1:
                ax = axs
            else:
                ax = axs.flatten()[idx]
            ax.bar(x_labels, distr, width=0.61, align='center')

            if plot_rand_distr:
                ax.bar(x_labels, rand_distr, width=0.31, align='center')

            ax.set_xticks(x_labels)
            ax.grid(axis='y', alpha=0.75)
            ax.set_title(f'w = {w}, d = {d}', size=20)

            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(15)

            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(15)

    fig.legend(legend, loc='right', fontsize=15)
    plt.ylim(0, 1)
    fig.text(0.5, 0.05, 'Hamming Weight of Error', ha='center', va='center', fontsize=20)
    fig.text(0.06, 0.5, 'Relative Frequency of Occurrence', ha='center', va='center',
             rotation='vertical', fontsize=20)
    plt.subplots_adjust(wspace=0, hspace=.15, left=.1)

    return fig, axs


def plot_success(successes, title, widths=None, depths=None, boxsize=1500):
    """
    Plot the given successes at each width and depth.

    If a given (width, depth) is not recorded in successes then nothing is plotted for that
    point. Successes are displayed as filled boxes while failures are simply box outlines.

    :param successes:
    :param title:
    :param widths:
    :param depths:
    :param boxsize:
    :return:
    """
    if widths is None:
        widths = list(successes.keys())

    if depths is None:
        depths = list(set(d for w in successes.keys() for d in successes[w].keys()))

    fig_width = min(len(widths), 15)
    fig_depth = min(len(depths), 15)

    fig, ax = plt.subplots(figsize=(fig_width, fig_depth))

    margin = .5
    ax.set_xlim(-margin, len(widths) + margin - 1)
    ax.set_ylim(-margin, len(depths) + margin - 1)
    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels(widths)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels(depths)
    ax.set_xlabel('Width')
    ax.set_ylabel('Depth')

    colors = ['white', 'lightblue']

    for w_idx, w in enumerate(widths):
        if w not in successes.keys():
            continue
        depth_succ = successes[w]
        for d_idx, d in enumerate(depths):
            if d not in depth_succ.keys():
                continue
            color = colors[0]
            if depth_succ[d]:
                color = colors[1]
            ax.scatter(w_idx, d_idx, marker='s', s=boxsize, color=color,
                       edgecolors='black')

    # legend
    labels = ['Fail', 'Pass']
    for color, label in zip(colors, labels):
        plt.scatter([], [], marker='s', c=color, label=label, edgecolors='black')
    ax.legend()

    ax.set_title(title)

    return fig, ax


def plot_pareto_frontier(successes, title, widths=None, depths=None):
    """
    Given the successes at measured widths and depths, draw the frontier that separates success
    from failure.

    Specifically, the frontier is drawn as follows::

        For a given width, draw a line separating all low-depth successes from the minimum
        depth failure. For each depth smaller than the minimum failure depth, draw a line
        separating the neighboring (width +/- 1, depth) cell if depth is less than the
        minimum depth failure for that neighboring width.

    If a requested (width, depth) cell is not specified in successes then no lines will be drawn
    around that cell.

    :param successes:
    :param title:
    :param widths:
    :param depths:
    :return:
    """
    if widths is None:
        widths = list(successes.keys())

    if depths is None:
        depths = list(set(d for w in successes.keys() for d in successes[w].keys()))

    fig_width = min(len(widths), 15)
    fig_depth = min(len(depths), 15)

    fig, ax = plt.subplots(figsize=(fig_width, fig_depth))

    margin = .5
    ax.set_xlim(-margin, len(widths) + margin - 1)
    ax.set_ylim(-margin, len(depths) + margin - 1)
    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels(widths)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels(depths)
    ax.set_xlabel('Width')
    ax.set_ylabel('Depth')

    min_depth_idx_failure_at_width = []
    for w_idx, w in enumerate(widths):
        if w not in successes.keys():
            min_depth_idx_failure_at_width.append(None)
            continue

        depth_succ = successes[w]
        min_depth_failure = len(depths)
        for d_idx, d in enumerate(depths):
            if d not in depth_succ.keys():
                continue
            if not depth_succ[d]:
                min_depth_failure = d_idx
                break
        min_depth_idx_failure_at_width.append(min_depth_failure)

    for w_idx, failure_idx in enumerate(min_depth_idx_failure_at_width):
        if failure_idx is None:
            continue  # this width was not measured, so leave the boundary open

        # horizontal line for this width
        if failure_idx < len(depths):  # measured a failure
            ax.plot((w_idx - margin, w_idx + margin), (failure_idx - margin, failure_idx - margin),
                    color='black')

        # vertical lines
        if w_idx < len(widths) - 1:  # check not at max width
            for d_idx in range(len(depths)):
                # check that the current depth was measured for this width
                if depths[d_idx] not in [d for d in successes[widths[w_idx]].keys()]:
                    continue  # do not plot line if this depth was not measured

                # if the adjacent width is not measured leave the boundary open
                if min_depth_idx_failure_at_width[w_idx + 1] is None:
                    continue

                # check if in the interior but adjacent to exterior
                # or if in the exterior but adjacent to interior
                if failure_idx > d_idx >= min_depth_idx_failure_at_width[w_idx + 1] \
                        or failure_idx <= d_idx < min_depth_idx_failure_at_width[w_idx + 1]:
                    ax.plot((w_idx + margin, w_idx + margin), (d_idx - margin, d_idx + margin),
                            color='black')

    ax.set_title(title)
    return fig, ax
