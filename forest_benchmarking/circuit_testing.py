from typing import List
import networkx as nx
import random
import itertools

from pyquil.quilbase import Pragma
from pyquil.quil import Program
from pyquil.api import QuantumComputer
from pyquil.api import BenchmarkConnection
from pyquil.gates import CNOT, CCNOT, Z, X, I, H, CZ, MEASURE, RESET
from pyquil.quil import address_qubits
from forest_benchmarking.rb import get_rb_gateset


# ==================================================================================================
# Gate Sets
# ==================================================================================================
def random_single_qubit_gates(graph: nx.Graph, gates: list):
    """Create a program comprised of single qubit gates randomly placed on the nodes of the
    specified graph. The gates are chosen uniformly from the list provided.

    :param graph: The graph. Nodes are used as arguments to gates, so they should be qubit-like.
    :param gates: A list of gates e.g. [I, X, Z] or [I, X].
    :return: A program that randomly places single qubit gates on a graph.
    """
    program = Program()
    for q in graph.nodes:
        gate = random.choice(gates)
        program += gate(q)
    return program


def random_two_qubit_gates(graph: nx.Graph, gates: list):
    """Write a program to randomly place two qubit gates on edges of the specified graph.

    :param graph: The graph. Nodes are used as arguments to gates, so they should be qubit-like.
    :param gates: A list of gates e.g. [I otimes I, CZ] or [CZ, SWAP, CNOT]
    :return: A program that has two qubit gates randomly placed on the graph edges.
    """
    program = Program()
    # do the two coloring with pragmas?
    # no point until fencing is over
    for a, b in graph.edges:
        gate = random.choice(gates)
        program += gate(a, b)
    return program


def random_single_qubit_cliffords(bm: BenchmarkConnection, graph: nx.Graph):
    """Create a program comprised of single qubit Cliffords gates randomly placed on the nodes of
    the specified graph. The gates are chosen uniformly from the list provided.

    :param bm: A benchmark connection that will do the grunt work of generating the Cliffords
    :param graph: The graph. Nodes are used as arguments to gates, so they should be qubit-like.
    :return: A program that randomly places single qubit Clifford gates on a graph.
    """
    num_qubits = len(graph.nodes)
    gateset_1q, q_placeholders1 = get_rb_gateset(rb_type='1q')

    # the +1 is because the depth includes the inverse
    clif_n_inv = bm.generate_rb_sequence(depth=(num_qubits + 1), gateset=gateset_1q, seed=None)
    rand_cliffords = clif_n_inv[0:num_qubits]

    prog = Program()
    for q, clif in zip(graph.nodes, rand_cliffords):
        gate = address_qubits(clif, qubit_mapping={clif.get_qubits().pop(): q})
        prog += gate
    return prog


def random_two_qubit_cliffords(bm: BenchmarkConnection, graph: nx.Graph):
    """Write a program to place random two qubit Cliffords gates on edges of the graph.

    :param bm: A benchmark connection that will do the grunt work of generating the Cliffords
    :param graph: The graph. Nodes are used as arguments to gates, so they should be qubit-like.
    :return: A program that has two qubit gates randomly placed on the graph edges.
    """
    num_2q_gates = len(graph.edges)
    gateset_2q, q_placeholders2 = get_rb_gateset(rb_type='2q')

    # the +1 is because the depth includes the inverse
    clif_n_inv = bm.generate_rb_sequence(depth=(num_2q_gates + 1), gateset=gateset_2q, seed=None)
    rand_cliffords = clif_n_inv[0:num_2q_gates]

    prog = Program()
    # do the two coloring with pragmas?
    # no point until fencing is over
    for edges, clif in zip(graph.edges, rand_cliffords):
        qb1, qb2 = clif.get_qubits()
        gate = address_qubits(clif, qubit_mapping={qb1: edges[0], qb2: edges[1], })
        prog += gate
    return prog


# ==================================================================================================
# Prefix // Suffix programs; pre and post
# ==================================================================================================

def pre_trival(graph: nx.Graph):
    # Install identity on all qubits so that we can find all the qubits from prog.get_qubits().
    # Otherwise if the circuit happens to be identity on a particular qubit you will get
    # not get that qubit from get_qubits. Worse, if the entire program is identity you will
    # get the empty set. Do not delete this!
    prep_gate = I
    prog = Program()
    prog += [prep_gate(qubit) for qubit in list(graph.nodes)]
    return prog

def post_trival():
    prog = Program()
    return prog


# ==================================================================================================
# Layer tools
# ==================================================================================================

def layer_1q_and_2q_rand_cliff(bm: BenchmarkConnection,
                               graph: nx.Graph,
                               layer_dagger: bool = False):
    '''
    Creates a layer of random one qubit Cliffords followed by random two qubit Cliffords.

    :param bm: A benchmark connection that will do the grunt work of generating the Cliffords
    :param graph:  The graph. Nodes are used as arguments to gates, so they should be qubit-like.
    :param layer_dagger: Bool if true will add the dagger to the layer, making the layer
    efectivley the identity
    :return: program
    '''
    prog = Program()
    prog += random_single_qubit_cliffords(bm, graph)
    prog += random_two_qubit_cliffords(bm, graph)
    if layer_dagger:
        prog += prog.dagger()
    return prog

def layer_1q_and_2q_rand_gates(graph: nx.Graph,
                               one_q_gates,
                               two_q_gates,
                               layer_dagger: bool = False):
    '''
    You pass in two lists of one and two qubit gates. This function creates a layer of random one
    qubit gates followed by random two qubit gates

    :param graph: The graph. Nodes are used as arguments to gates, so they should be qubit-like.
    :param one_q_gates: list of one qubit gates
    :param two_q_gates: list of two qubit gates e.g. [CZ, ID]
    :param layer_dagger: Bool if true will add the dagger to the layer, making the layer
    efectivley the identity
    :return: program
    '''
    prog = Program()
    prog += random_single_qubit_gates(graph, one_q_gates)
    prog += random_two_qubit_gates(graph, two_q_gates)
    if layer_dagger:
        prog += prog.dagger()
    return prog

# ==================================================================================================
# Sandwich tools
# ==================================================================================================
def circuit_sandwich_rand_gates(graph: nx.Graph,
                                depth: int,
                                one_q_gates: list,
                                two_q_gates: list,
                                layer_dagger: bool = False,
                                sandwich_dagger: bool = False):
    '''

    :param graph:
    :param depth:
    :param one_q_gates:
    :param two_q_gates:
    :param layer_dagger:
    :param sandwich_dagger:
    :return:
    '''
    total_prog = Program()
    total_prog += pre_trival(graph)

    if sandwich_dagger:
        depth = int(np.floor(depth / 2))

    layer_progs = Program()
    for ddx in range(1, depth + 1):
        layer_progs += layer_1q_and_2q_rand_gates(graph,
                                                  one_q_gates,
                                                  two_q_gates,
                                                  layer_dagger)
    if sandwich_dagger:
        layer_progs += layer_progs.dagger()

    total_prog += layer_progs
    total_prog += post_trival()
    return total_prog


def circuit_sandwich_clifford(bm: BenchmarkConnection,
                              graph: nx.Graph,
                              depth: int,
                              layer_dagger: bool = False,
                              sandwich_dagger: bool = False):
    '''

    :param bm:
    :param graph:
    :param depth:
    :param layer_dagger:
    :param sandwich_dagger:
    :return:
    '''
    total_prog = Program()

    total_prog += pre_trival(graph)

    if sandwich_dagger:
        depth = int(np.floor(depth / 2))

    layer_progs = Program()
    for ddx in range(1, depth + 1):
        layer_progs += layer_1q_and_2q_rand_cliff(bm, graph, layer_dagger)
    if sandwich_dagger:
        layer_progs += layer_progs.dagger()

    total_prog += layer_progs
    total_prog += post_trival()
    return total_prog

# ==================================================================================================
# Graph tools
# ==================================================================================================
def generate_connected_subgraphs(G: nx.Graph, n_vert: int):
    '''
    Given a lattice on the QPU or QVM, specified by a networkx graph, return a list of all
    subgraphs with n_vert connect vertices.

    :params n_vert: number of verticies of connected subgraph.
    :params G: networkx Graph
    :returns: list of subgraphs with n_vert connected vertices
    '''
    subgraph_list = []
    for sub_nodes in itertools.combinations(G.nodes(), n_vert):
        subg = G.subgraph(sub_nodes)
        if nx.is_connected(subg):
            subgraph_list.append(subg)
    return  subgraph_list