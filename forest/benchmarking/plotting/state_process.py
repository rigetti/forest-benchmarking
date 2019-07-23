import matplotlib.pyplot as plt
import itertools
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

THREE_COLOR_MAP = ['#48737F', '#FFFFFF', '#D6619E']
rigetti_3_color_cm = LinearSegmentedColormap.from_list("Rigetti", THREE_COLOR_MAP[::-1], N=100)


def plot_pauli_rep_of_state(state_pl_basis, ax, labels, title):
    """
    Visualize a quantum state in the Pauli-Liouville basis.

    :Examples:

    ::

        from forest.benchmarking.superoperator_tools import *
        from forest.benchmarking.utils import n_qubit_pauli_basis
        # zero state in the (Z) computational basis
        rho_std_basis = np.array([[1, 0], [0, 0]])
        # change to Pauli-Liouville basis
        n_qubits = 1
        pl_basis = n_qubit_pauli_basis(n_qubits)
        c2p = computational2pauli_basis_matrix(2*n_qubits)
        rho_pl_basis = np.real(c2p@vec(rho_std_basis))
        # plot
        fig, ax = plt.subplots(1)
        plot_pauli_rep_of_state(rho_pl_basis, ax, pl_basis.labels, 'Zero state |0>')


    :param numpy.ndarray state_pl_basis: The quantum state represented in the Pauli-Liouville basis.
    :param ax: The matplotlib axes.
    :param labels: The labels for the operator basis states.
    :param title: The title for the plot.
    """
    if len(state_pl_basis.shape) == 1:
        raise ValueError("You must pass in a (N by 1) or a (1 by N) numpy.ndarray")
    if np.iscomplexobj(state_pl_basis):
        raise ValueError("You must pass in a real vector")

    im = ax.imshow(state_pl_basis, interpolation="nearest", cmap="RdBu", vmin=-1 / 2, vmax=1 / 2)
    dim = len(labels)
    # make the colorbar ticks look pretty
    rows, cols = state_pl_basis.shape
    if rows > cols:
        cb = plt.colorbar(im, ax=ax, ticks=[-1 / 2, -1 / 4, 0, 1 / 4, 1 / 2])
        ticklabs = cb.ax.get_yticklabels()
        cb.ax.set_yticklabels(ticklabs, ha='right')
        cb.ax.yaxis.set_tick_params(pad=35)
        # axis labels etc
        ax.set_xlabel("Coefficient")
        ax.set_xticks([])
        ax.set_yticks(range(dim))
        ax.set_ylabel("Pauli Operator")
        ax.set_yticklabels(labels)
    else:
        cb = plt.colorbar(im, ax=ax, ticks=[-1 / 2, -1 / 4, 0, 1 / 4, 1 / 2],
                          orientation="horizontal",
                          pad=0.22)
        ax.set_ylabel("Coefficient")
        ax.set_yticks([])
        ax.set_xticks(range(dim))
        ax.set_xlabel("Pauli Operator")
        ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.grid(False)


def plot_pauli_bar_rep_of_state(state_pl_basis, ax, labels, title):
    """
    Visualize a quantum state in the Pauli-Liouville basis. The magnitude of the operator
    coefficients are represented by the height of a bar in the bargraph.

    :param numpy.ndarray state_pl_basis: The quantum state represented in the Pauli-Liouville basis.
    :param ax: The matplotlib axes.
    :param labels: The labels for the operator basis states.
    :param title: The title for the plot.
    """
    dim = len(labels)
    im = ax.bar(np.arange(dim) - .4, np.real(state_pl_basis), width=.8)
    ax.set_xticks(range(dim))
    ax.set_xlabel("Pauli Operator")
    ax.set_ylabel("Coefficient")
    ax.set_title(title)
    ax.set_xticklabels(labels, rotation=45)
    ax.grid(False)


def plot_pauli_transfer_matrix(ptransfermatrix: np.ndarray, ax, labels=None, title='',
                               fontsizes: int = 16):
    """
    Visualize a quantum process using the Pauli-Liouville representation (aka the Pauli Transfer
    Matrix) of the process.

    :param ptransfermatrix: The Pauli Transfer Matrix
    :param ax: The matplotlib axes.
    :param labels: The labels for the operator basis states.
    :param title: The title for the plot
    :param fontsizes: Font size for axis labels
    :return: The modified axis object.
    :rtype: AxesSubplot
    """
    ptransfermatrix = np.real_if_close(ptransfermatrix)
    im = ax.imshow(ptransfermatrix, interpolation="nearest", cmap="RdBu", vmin=-1, vmax=1)
    if labels is None:
        dim_squared = ptransfermatrix.shape[0]
        num_qubits = np.int(np.log2(np.sqrt(dim_squared)))
        labels = [''.join(x) for x in itertools.product('IXYZ', repeat=num_qubits)]
    else:
        dim_squared = len(labels)

    cb = plt.colorbar(im, ax=ax, ticks=[-1, -3 / 4, -1 / 2, -1 / 4, 0, 1 / 4, 1 / 2, 3 / 4, 1])
    ticklabs = cb.ax.get_yticklabels()
    cb.ax.set_yticklabels(ticklabs, ha='right')
    cb.ax.yaxis.set_tick_params(pad=35)
    cb.draw_all()
    ax.set_xticks(range(dim_squared))
    ax.set_xlabel("Input Pauli Operator", fontsize=fontsizes)
    ax.set_yticks(range(dim_squared))
    ax.set_ylabel("Output Pauli Operator", fontsize=fontsizes)
    ax.set_title(title, fontsize= int(np.floor(1.2*fontsizes)), pad=15)
    ax.set_xticklabels(labels, rotation=45, fontsize=int(np.floor(0.7*fontsizes)))
    ax.set_yticklabels(labels, fontsize=int(np.floor(0.7*fontsizes)))
    ax.grid(False)
    return ax
