import networkx as nx
import numpy as np
import pytest
from pyquil.quil import address_qubits

from forest.benchmarking.entangled_states import *


def test_create_ghz_program(wfn):
    tree = nx.from_edgelist([(0, 1), (0, 2)], create_using=nx.DiGraph())
    prog = create_ghz_program(tree, skip_measurements=True)
    prog = address_qubits(prog)
    wf = wfn.wavefunction(prog)
    should_be = [0.5] + [0] * (2 ** tree.number_of_nodes() - 2) + [0.5]
    np.testing.assert_allclose(should_be, wf.probabilities())


def test_create_bad_ghz_program():
    tree = nx.from_edgelist([(0, 1), (1, 2), (2, 0)], create_using=nx.DiGraph())
    with pytest.raises(AssertionError) as e:
        prog = create_ghz_program(tree)

    assert e.match(r'Needs to be a tree')


def test_ghz_state_stats():
    bitstrings = [
        [0, 0],
        [1, 1],
        [0, 1],
        [1, 0],
    ]
    stats = ghz_state_statistics(bitstrings)
    assert stats['bell'] == 2
    assert stats['total'] == 4


def test_create_graph_state():
    graph = nx.complete_graph(4)
    graph = nx.relabel_nodes(graph, {i: i * 2 for i in range(4)})
    prog = create_graph_state(graph)
    n_czs = 0
    for line in prog.out().splitlines():
        if line.startswith('CZ'):
            n_czs += 1

    assert n_czs == nx.number_of_edges(graph)


def test_measure_graph_state():
    graph = nx.complete_graph(4)
    prog, addr = measure_graph_state(graph, focal_node=0)

    assert 'RY(theta[0])' in str(prog)
    assert addr == list(range(4))
    for a in addr:
        assert f'ro[{a}]' in prog.out()
