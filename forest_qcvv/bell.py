import networkx as nx
import numpy as np
from pyquil.gates import H, CNOT, MEASURE
from pyquil.quil import Program


def create_bell_program(tree: nx.DiGraph):
    """
    Create a Bell/GHZ state with CNOTs described by tree.

    :param tree: A tree that describes the CNOTs to perform to create a bell/GHZ state.
    :return: the program
    """
    assert nx.is_tree(tree), 'Needs to be a tree'
    nodes = list(nx.topological_sort(tree))
    n_qubits = len(nodes)
    program = Program(H(nodes[0]))

    for node in nodes:
        for child in tree.successors(node):
            program += CNOT(node, child)

    ro = program.declare('ro', 'BIT', n_qubits)
    for i, q in enumerate(nodes):
        program += MEASURE(q, ro[i])

    return program


def bell_state_statistics(bitstrings):
    """
    Compute statistics bitstrings sampled from a Bell/GHZ state

    :param bitstrings: An array of bitstrings
    :return: A dictionary where bell = number of bitstrings consistent with a bell/GHZ state;
        total = total number of bitstrings.
    """
    bitstrings = np.asarray(bitstrings)
    bell = np.sum(np.logical_or(np.all(bitstrings == 0, axis=1),
                                np.all(bitstrings == 1, axis=1)))
    total = len(bitstrings)
    return {
        'bell': int(bell),
        'total': int(total),
    }
