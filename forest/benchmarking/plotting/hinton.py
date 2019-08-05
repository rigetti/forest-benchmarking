import numpy as np
from typing import List
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from matplotlib.colors import Normalize

ANGLE_MAPPER = cm.ScalarMappable(norm=Normalize(vmin=-np.pi, vmax=np.pi))


# Modified from the SciPy Cookbook.
def hinton(matrix, max_weight=1.0, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('lightgrey')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = np.arctan2(w.real, w.imag)
        color = ANGLE_MAPPER.to_rgba(color)
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.set_xlim((-max_weight / 2, matrix.shape[0] - max_weight / 2))
    ax.set_ylim((-max_weight / 2, matrix.shape[1] - max_weight / 2))
    ax.autoscale_view()
    ax.invert_yaxis()


# From QuTiP which in turn modified the code from the SciPy Cookbook.
def _blob(x, y, w, w_max, area, cmap=None):
    """
    Draws a square-shaped blob with the given area (< 1) at the given coordinates.
    """
    hs = np.sqrt(area) / 2
    xcorners = np.array([x - hs, x + hs, x + hs, x - hs])
    ycorners = np.array([y - hs, y - hs, y + hs, y + hs])

    plt.fill(xcorners, ycorners, color=cmap)  # cmap(int((w + w_max) * 256 / (2 * w_max))))


# Modified from QuTip (see https://bit.ly/2LrbayH ) which in turn modified the code from the
# SciPy Cookbook.
def hinton_real(matrix: np.ndarray,
                max_weight: float = None,
                xlabels: List[str] = None,
                ylabels: List[str] = None,
                title: str = None,
                ax=None,
                cmap=None,
                label_top: bool = True):
    """
    Draw Hinton diagram for visualizing a real valued weight matrix.

    In the traditional Hinton diagram positive and negative values are represented by white and
    black squares respectively. The size of each square represents the magnitude of each value.
    The traditional Hinton diagram can be recovered by setting cmap = cm.Greys_r.

    :param matrix: The matrix to be visualized.
    :param max_weight: normalize size to this scalar.
    :param xlabels: The labels for the operator basis.
    :param ylabels: The labels for the operator basis.
    :param title: The title for the plot.
    :param ax: The matplotlib axes.
    :param cmap: A matplotlib colormap to use when plotting.
    :param label_top: If True, x-axis labels will be placed on top, otherwise they will appear
        below the plot.
    :return: A tuple of the matplotlib figure and axes instances used to produce the figure.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Grays in increasing darkness: smokewhite, gainsboro, lightgrey, lightgray, silver
    backgnd_gray = 'gainsboro'

    if cmap is None:
        cmap = cm.RdBu
        cneg = cmap(0)
        cpos = cmap(256)
        cmap = mpl.colors.ListedColormap([cneg, backgnd_gray, cpos])
    else:
        cneg = cmap(0)
        cpos = cmap(256)
        cmap = mpl.colors.ListedColormap([cneg, backgnd_gray, cpos])

    if title and fig:
        ax.set_title(title, y=1.1, fontsize=18)

    ax.set_aspect('equal', 'box')
    ax.set_frame_on(False)

    height, width = matrix.shape
    if max_weight is None:
        max_weight = 1.25 * max(abs(np.diag(np.matrix(matrix))))
        if max_weight <= 0.0:
            max_weight = 1.0

    bounds = [-max_weight, -0.0001, 0.0001, max_weight]
    tick_loc = [-max_weight / 2, 0, max_weight / 2]

    ax.fill(np.array([0, width, width, 0]), np.array([0, 0, height, height]), color=cmap(1))
    for x in range(width):
        for y in range(height):
            _x = x + 1
            _y = y + 1
            if np.real(matrix[x, y]) > 0.0:
                _blob(_x - 0.5, height - _y + 0.5, abs(matrix[x, y]), max_weight,
                      min(1, abs(matrix[x, y]) / max_weight), cmap=cmap(2))
            else:
                _blob(_x - 0.5, height - _y + 0.5, -abs(matrix[x, y]), max_weight,
                      min(1, abs(matrix[x, y]) / max_weight), cmap=cmap(0))

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    cax, kw = mpl.colorbar.make_axes(ax, shrink=0.75, pad=.1)
    mpl.colorbar.ColorbarBase(cax,
                              norm=norm,
                              cmap=cmap,
                              boundaries=bounds,
                              ticks=tick_loc).set_ticklabels(['$-$', '$0$', '$+$'])
    cax.tick_params(labelsize=14)
    # x axis
    ax.xaxis.set_major_locator(plt.IndexLocator(1, 0.5))
    if xlabels:
        ax.set_xticklabels(xlabels)
        if label_top:
            ax.xaxis.tick_top()
    ax.tick_params(axis='x', labelsize=14)
    # y axis
    ax.yaxis.set_major_locator(plt.IndexLocator(1, 0.5))
    if ylabels:
        ax.set_yticklabels(list(reversed(ylabels)))
    ax.tick_params(axis='y', labelsize=14)

    return fig, ax