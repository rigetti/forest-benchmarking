from typing import Sequence, List
import networkx as nx
import numpy as np
import random

from pyquil.quilbase import Pragma, Gate, DefGate, DefPermutationGate
from pyquil.quilatom import QubitPlaceholder
from pyquil.quil import Program, address_qubits, merge_programs
from pyquil.api import BenchmarkConnection
from pyquil.gates import *

from forest.benchmarking.randomized_benchmarking import get_rb_gateset
from forest.benchmarking.operator_tools.random_operators import haar_rand_unitary


def random_single_qubit_gates(graph: nx.Graph, gates: Sequence[Gate]) -> Program:
    """
    Create a program comprised of random single qubit gates acting on the qubits of the
    specified graph; each gate is chosen uniformly at random from the list provided.

    :param graph: The graph. Nodes are used as arguments to gates, so they should be qubit-like.
    :param gates: A list of gates e.g. [I, X, Z] or [I, X].
    :return: A program that randomly places single qubit gates on a graph.
    """
    program = Program()
    for q in graph.nodes:
        gate = random.choice(gates)
        program += gate(q)
    return program


def random_two_qubit_gates(graph: nx.Graph, gates: Sequence[Gate]) -> Program:
    """
    Create a program to randomly place two qubit gates on edges of the specified graph.

    :param graph: The graph. Nodes are used as arguments to gates, so they should be qubit-like.
    :param gates: A list of gates e.g. [I otimes I, CZ] or [CZ, SWAP, CNOT]
    :return: A program that has two qubit gates randomly placed on the graph edges.
    """
    program = Program()
    # TODO: two coloring with pragmas
    for a, b in graph.edges:
        gate = random.choice(gates)
        program += gate(a, b)
    return program


def random_single_qubit_cliffords(bm: BenchmarkConnection, graph: nx.Graph) -> Program:
    """
    Create a program comprised of single qubit Clifford gates randomly placed on the nodes of
    the specified graph. Each uniformly random choice of Clifford is implemented in the native
    gateset.

    :param bm: A benchmark connection that will do the grunt work of generating the Cliffords
    :param graph: The graph. Nodes are used as arguments to gates, so they should be qubit-like.
    :return: A program that randomly places single qubit Clifford gates on a graph.
    """
    num_qubits = len(graph.nodes)

    q_placeholder = QubitPlaceholder()
    gateset_1q = get_rb_gateset([q_placeholder])

    # the +1 is because the depth includes the inverse
    clif_n_inv = bm.generate_rb_sequence(depth=(num_qubits + 1), gateset=gateset_1q, seed=None)
    rand_cliffords = clif_n_inv[0:num_qubits]

    prog = Program()
    for q, clif in zip(graph.nodes, rand_cliffords):
        gate = address_qubits(clif, qubit_mapping={q_placeholder: q})
        prog += gate
    return prog


def random_two_qubit_cliffords(bm: BenchmarkConnection, graph: nx.Graph) -> Program:
    """
    Write a program to place random two qubit Clifford gates on edges of the graph.

    :param bm: A benchmark connection that will do the grunt work of generating the Cliffords
    :param graph: The graph. Nodes are used as arguments to gates, so they should be qubit-like.
    :return: A program that has two qubit gates randomly placed on the graph edges.
    """
    num_2q_gates = len(graph.edges)
    q_placeholders = QubitPlaceholder.register(n=2)
    gateset_2q = get_rb_gateset(q_placeholders)

    # the +1 is because the depth includes the inverse
    clif_n_inv = bm.generate_rb_sequence(depth=(num_2q_gates + 1), gateset=gateset_2q, seed=None)
    rand_cliffords = clif_n_inv[0:num_2q_gates]

    prog = Program()
    # TODO: two coloring with PRAGMAS?
    # TODO: longer term, fence to be 'simultaneous'?
    for edges, clif in zip(graph.edges, rand_cliffords):
        gate = address_qubits(clif, qubit_mapping={q_placeholders[0]: edges[0],
                                                   q_placeholders[1]: edges[1]})
        prog += gate
    return prog


def dagger_previous(sequence: List[Program], n: int = 1) -> Program:
    """
    Create a program which is the inverse (conjugate transpose; adjoint; dagger) of the last n
    layers of the provided sequence.

    :param sequence: a sequence of PyQuil programs whose elements are layers in a circuit
    :param n: the number of layers at the end of the sequence that will be inverted
    :return: a program that inverts the last n layers of the provided sequence.
    """
    return merge_programs(sequence[-n:]).dagger()


def random_su4_pairs(graph: nx.Graph, idx_label: int) -> Program:
    """
    Create a program that enacts a Haar random 2 qubit gate on random pairs of qubits in the
    graph, irrespective of graph topology.

    If the graph contains an odd number of nodes, then one random qubit will not be acted upon by
    any gate.

    The output program will need to be compiled into native gates.

    This generator is the repeated unit of the quantum volume circuits described in [QVol]_. Note
    that the qubit permutation is done implicitly--the compiler will have to figure out how to
    move potentially distant qubits onto a shared edge in order to enact the random two qubit gate.

    :param graph: a graph containing qubits that will be randomly paired together. Note that
        the graph topology (the edges) are ignored.
    :param idx_label: a label that uniquely identifies the set of gate definitions used in the
        output program. This prevents subsequent calls to this method from producing a program
        with definitions that overwrite definitions in previously generated programs.
    :return: a program with random two qubit gates between random pairs of qubits.
    """
    qubits = list(graph.nodes)
    qubits = [qubits[idx] for idx in np.random.permutation(range(len(qubits)))]
    prog = Program()
    # ignore the edges in the graph
    for q1, q2 in zip(qubits[::2], qubits[1::2]):
        matrix = haar_rand_unitary(4)
        gate_definition = DefGate(f"LYR{idx_label}_RSU4_{q1}_{q2}", matrix)
        RSU4 = gate_definition.get_constructor()
        prog += gate_definition
        prog += RSU4(q1, q2)
    return prog
