import matplotlib.pyplot as plt
import numpy as np

def plot_pauli_rep_of_state(state_pl_basis, ax, labels, title):
    """
    Visualize a quantum state in the Pauli-Liouville basis.

    Example:

    from forest.benchmarking.superoperator_tools import *
    from forest.benchmarking.utils import n_qubit_pauli_basis

    # zero state in the (Z) computational basis
    rho_std_basis = np.array([[1, 0], [0, 0]])

    # change to Pauli-Liouville basis
    n_qubits = 1
    pl_basis = n_qubit_pauli_basis(n_qubits)
    c2p = computational2pauli_basis_matrix(2*n_qubits)
    rho_pl_basis = np.real(c2p@vec(rho_std_basis))

    #plot
    fig, ax = plt.subplots(1)
    plot_pauli_rep_of_state(rho_pl_basis, ax, pl_basis.labels, 'Zero state |0>')


    :param numpy.ndarray state_pl_basis: The quantum state represented in the Pauli-Liouville basis.
    :param ax: The matplotlib axes.
    :param labels: The labels for the operator basis states.
    :param title: The title for the plot.
    """
    if len(state_pl_basis.shape)==1:
        raise ValueError("You must pass in a (N by 1) or a (1 by N) numpy.ndarray")
    if np.iscomplexobj(rho_std_basis):
        raise ValueError("You must pass in a real vector")

    im = ax.imshow(state_pl_basis, interpolation="nearest", cmap="RdBu", vmin=-1/2, vmax=1/2)
    dim = len(labels)
    # make the colorbar ticks look pretty
    rows, cols = state_pl_basis.shape
    if rows>cols:
        cb = plt.colorbar(im, ax=ax,ticks=[-1/2,-1/4, 0, 1/4, 1/2])
        ticklabs = cb.ax.get_yticklabels()
        cb.ax.set_yticklabels(ticklabs,ha='right')
        cb.ax.yaxis.set_tick_params(pad=35)
        # axis labels etc
        ax.set_xlabel("Coefficient")
        ax.set_xticks([])
        ax.set_yticks(range(dim))
        ax.set_ylabel("Pauli Operator")
        ax.set_yticklabels(labels)
    else:
        cb = plt.colorbar(im, ax=ax,ticks=[-1/2,-1/4, 0, 1/4, 1/2],orientation="horizontal", pad=0.23)
        ax.set_ylabel("Coefficient")
        ax.set_yticks([])
        ax.set_xticks(range(dim))
        ax.set_xlabel("Pauli Operator")
        ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.grid(False)
