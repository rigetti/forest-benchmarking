import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

ANGLE_MAPPER = cm.ScalarMappable(norm=Normalize(vmin=-np.pi, vmax=np.pi))


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
