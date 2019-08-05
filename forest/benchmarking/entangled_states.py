import networkx as nx
import numpy as np

from pyquil.api import QPUCompiler
from pyquil.gates import H, RY, CZ, CNOT, MEASURE
from pyquil.quil import Program
from pyquil.quilbase import Pragma
from forest.benchmarking.compilation import basic_compile


def create_ghz_program(tree: nx.DiGraph):
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


def ghz_state_statistics(bitstrings):
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


def create_graph_state(graph: nx.Graph, use_pragmas=False):
    """
    Write a program to create a graph state according to the specified graph

    A graph state involves Hadamarding all your qubits and then applying a CZ for each
    edge in the graph. A graph state and the ability to measure it however you want gives
    you universal quantum computation. Some good references are [MBQC]_ and [MBCS]_.

    Similar to a Bell state / GHZ state, we can try to prepare a graph state and measure
    how well we've done according to expected parities.

    .. [MBQC] A One-Way Quantum Computer.
           Raussendorf et al.
           Phys. Rev. Lett. 86, 5188 (2001).
           https://doi.org/10.1103/PhysRevLett.86.5188
           https://arxiv.org/abs/quant-ph/0010033

    .. [MBCS] Measurement-based quantum computation with cluster states.
           Raussendorf et al.
           Phys. Rev. A 68, 022312 (2003).
           https://dx.doi.org/10.1103/PhysRevA.68.022312
           https://arxiv.org/abs/quant-ph/0301052

    :param graph: The graph. Nodes are used as arguments to gates, so they should be qubit-like.
    :param use_pragmas: Use COMMUTING_BLOCKS pragmas to hint at the compiler
    :return: A program that constructs a graph state.
    """
    program = Program()
    for q in graph.nodes:
        program += H(q)

    if use_pragmas:
        program += Pragma('COMMUTING_BLOCKS')
    for a, b in graph.edges:
        if use_pragmas:
            program += Pragma('BLOCK')
        program += CZ(a, b)
        if use_pragmas:
            program += Pragma('END_BLOCK')
    if use_pragmas:
        program += Pragma('END_COMMUTING_BLOCKS')

    return program


def measure_graph_state(graph: nx.Graph, focal_node: int):
    """
    Given a graph state, measure a focal node and its neighbors with a particular measurement
    angle.

    :param graph: The graph state graph. This is needed to figure out what the neighbors are
    :param focal_node: The node in the graph to serve as the focus. The focal node is measured
        at an angle and all its neighbors are measured in the Z basis
    :return: Program, list of classical offsets into the ``ro`` register.
    """
    program = Program()
    theta = program.declare('theta', 'REAL')
    program += RY(theta, focal_node)

    neighbors = sorted(graph[focal_node])
    ro = program.declare('ro', 'BIT', len(neighbors) + 1)

    program += MEASURE(focal_node, ro[0])
    for i, neighbor in enumerate(neighbors):
        program += MEASURE(neighbor, ro[i + 1])

    classical_addresses = list(range(len(neighbors) + 1))
    return program, classical_addresses


def compiled_parametric_graph_state(compiler: QPUCompiler, graph: nx.Graph, focal_node: int,
                                    num_shots: int = 1000):
    """
    Construct a program to create and measure a graph state, map it to qubits using ``addressing``,
    and compile to an ISA.

    Hackily implement a parameterized program by compiling a program with a particular angle,
    finding where that angle appears in the results, and replacing it with ``"{angle}"`` so
    the resulting compiled program can be run many times by using python's str.format method.

    :param graph: A networkx graph defining the graph state
    :param focal_node: The node of the graph to measure
    :param compiler: The compiler to do the compiling.
    :param num_shots: The number of shots to take when measuring the graph state.
    :return: an executable that constructs and measures a graph state.
    """
    program = create_graph_state(graph)
    measure_prog, c_addrs = measure_graph_state(graph, focal_node)
    program += measure_prog
    program.wrap_in_numshots_loop(num_shots)
    nq_program = basic_compile(program)
    executable = compiler.native_quil_to_executable(nq_program)
    return executable
