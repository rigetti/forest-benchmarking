import networkx as nx
import numpy as np
import pytest
from pyquil.quil import address_qubits

from forest_benchmarking.bell import create_bell_program, bell_state_statistics


def test_create_bell_program(wfn):
    tree = nx.from_edgelist([(0, 1), (0, 2)], create_using=nx.DiGraph())
    prog = create_bell_program(tree)
    for _ in tree.nodes:
        prog.pop()  # remove measurements
    prog = address_qubits(prog)
    wf = wfn.wavefunction(prog)
    should_be = [0.5] + [0] * (2 ** tree.number_of_nodes() - 2) + [0.5]
    np.testing.assert_allclose(should_be, wf.probabilities())


def test_create_bad_bell_program():
    tree = nx.from_edgelist([(0, 1), (1, 2), (2, 0)], create_using=nx.DiGraph())
    with pytest.raises(AssertionError) as e:
        prog = create_bell_program(tree)

    assert e.match(r'Needs to be a tree')


def test_bell_state_stats():
    bitstrings = [
        [0, 0],
        [1, 1],
        [0, 1],
        [1, 0],
    ]
    stats = bell_state_statistics(bitstrings)
    assert stats['bell'] == 2
    assert stats['total'] == 4
