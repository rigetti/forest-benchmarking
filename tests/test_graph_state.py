import networkx as nx

from forest_benchmarking.graph_state import create_graph_state, measure_graph_state


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
    print(prog.out())

    assert 'RY(theta)' in str(prog)
    assert addr == list(range(4))
    for a in addr:
        assert f'ro[{a}]' in prog.out()
