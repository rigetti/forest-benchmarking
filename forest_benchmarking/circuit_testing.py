from typing import List
import networkx as nx
import random

from pyquil.quilbase import Pragma
from pyquil.quil import Program
from pyquil.api import QuantumComputer
from pyquil.api import BenchmarkConnection
from pyquil.quil import address_qubits
from forest_benchmarking.rb import get_rb_gateset



#===================================================================================================
# Gate Sets
#===================================================================================================
def random_single_qubit_gates(graph: nx.Graph, gates: list):
    """Create a program comprised of single qubit gates randomly placed on the nodes
    according to the specified graph. The gates are chosen uniformly from the list specified.

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
    """Write a program to randomly place two qubit gates on edges of the graph.

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
    """Create a program comprised of single qubit Cliffords gates randomly placed on the nodes
    according to the specified graph. The gates are chosen uniformly from the list specified.

    :param bm: A benchmark connection that will do the grunt work of generating the Cliffords
    :param graph: The graph. Nodes are used as arguments to gates, so they should be qubit-like.
    :return: A program that randomly places single qubit Clifford gates on a graph.
    """
    gateset_1q, q_placeholders1 = get_rb_gateset(rb_type='1q')
    prog = Program()
    for q in graph.nodes:
        clif_n_inv = bm.generate_rb_sequence(depth=2,gateset=gateset_1q,seed=None)
        # two elements are return for depth two. We take the first, the second is the inverse
        gate = address_qubits(clif_n_inv[0],qubit_mapping={clif_n_inv[0].get_qubits().pop():q})
        prog += gate
    return prog

def random_two_qubit_cliffords(bm: BenchmarkConnection, graph: nx.Graph):
    """Write a program to place random two qubit Cliffords gates on edges of the graph.

    :param bm: A benchmark connection that will do the grunt work of generating the Cliffords
    :param graph: The graph. Nodes are used as arguments to gates, so they should be qubit-like.
    :return: A program that has two qubit gates randomly placed on the graph edges.
    """
    gateset_2q, q_placeholders2 = get_rb_gateset(rb_type='2q')
    prog = Program()
    # do the two coloring with pragmas?
    # no point until fencing is over
    for a, b in graph.edges:
        clif_n_inv = bm.generate_rb_sequence(depth=2,gateset=gateset_2q,seed=None)
        qb1, qb2 = clif_n_inv[0].get_qubits()
        # two elements are return for depth two. We take the first, the second is the inverse
        gate = address_qubits(clif_n_inv[0],qubit_mapping={qb1: a, qb2: b,})
        prog += gate
    return prog