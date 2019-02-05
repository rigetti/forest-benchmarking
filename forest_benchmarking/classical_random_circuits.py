from typing import List
import itertools
import numpy as np
from scipy.spatial.distance import hamming
from scipy.special import comb
import networkx as nx
import random

from pyquil.gates import CNOT, CCNOT, X, Z, I, H, CZ, MEASURE, RESET
from pyquil.quilbase import Pragma
from pyquil.quil import Program

from forest_benchmarking.compilation import CNOT_X_basis


def generate_connected_subgraphs(G: nx.Graph, n_vert: int):
    '''
    Given a lattice on the QPU or QVM, specified by a networkx graph, return a list of all
    subgraphs with n_vert connect vertices.

    :params n_vert: number of verticies of connected subgraph.
    :params G: networkx Graph
    :returns: list of subgraphs with n_vert connected vertices
    '''
    target = nx.complete_graph(n_vert)
    subgraph_list = []
    for sub_nodes in itertools.combinations(G.nodes(), len(target.nodes())):
        subg = G.subgraph(sub_nodes)
        if nx.is_connected(subg):
            subgraph_list.append(subg)
    return subgraph_list

def random_single_bit_gates(graph: nx.Graph, in_x_basis: bool = False):
    """Write a program to randomly flip bits by randomly placing X gates or Z gates on qubits
    according to the specified graph.

    The X gate flips bits in the Z basis i.e. X|0> = |1> and X|1> = |0>.
    The Z gate flips bits in the X basis i.e. Z|+> = |-> and Z|-> = |+>.

    :param graph: The graph. Nodes are used as arguments to gates, so they should be qubit-like.
    :param in_x_basis: if true, will apply Z gate instead of X gates.
    :return: A program that randomly places single qubit X gates or Z gates if in_x_basis = True.
    """
    if in_x_basis:
        bit_flip = Z
    else:
        bit_flip = X

    program = Program()
    for q in graph.nodes:
        if random.random()>0.5:
            program += bit_flip(q)
    return program

def random_two_bit_gates(graph: nx.Graph, in_x_basis: bool = False):
    """Write a program to randomly place CNOT gates between qubits according to the specified
    graph. If the flag in_x_basis = True, then a CNOT in the X basis is applied, i.e.

    CNOTX = |+X+| * I + |-X-| * Z

    where |+> and |-> are the +/- eigenstate of the Pauli X operator and * denotes a tensor product.

    :param graph: The graph. Nodes are used as arguments to gates, so they should be qubit-like.
    :param in_x_basis: if true, will apply a CNOT in the X basis instead of the CNOT gate.
    :return: A program that has randomly placed single qubit X gates.
    """
    if in_x_basis:
        cnot_gate = CNOT_X_basis
    else:
        cnot_gate = CNOT

    program = Program()
    program += Pragma('COMMUTING_BLOCKS')
    for a, b in graph.edges:
        if random.random()>0.5:
            program += Pragma('BLOCK')
            program += cnot_gate(a, b)
            program += Pragma('END_BLOCK')
    program += Pragma('END_COMMUTING_BLOCKS')
    return program


def generate_random_classial_circuit_with_depth(graph: nx.Graph,
                                                depth: int,
                                                in_x_basis: bool = False):
    """Generate a program to randomly single and two qubit classical gates on specified graph
    with a certain "depth". The initial state is all all qubits prepared in zero.

    Depth has a special meaning here. A depth one program contains one 'layer' of random single
    qubit gates on the vertices and 'layer' of randomly placed two qubit gates. A depth two
    program would be two such layers.

    If in_x_basis = True, the program initializes all qubits the the plus state. Then all logic
    is in the X basis. That is bit flips are implemented by Z and two bit gates are implemented
    by CNOT_in_X_basis = |+X+| * I + |-X-| * Z, where |+> and |-> are the +/- eigenstate of the
    Pauli X operator and * denotes a tensor product. Finally the circuit is transformed back to
    the z basis for measurement by Hadamarding all qubits.

    :param graph: The graph. Nodes are used as arguments to gates, so they should be qubit-like.
    :param depth: The number of times a layers of single qubit gates and two qubit gates.
    :param in_x_basis: if true, will generate classical logic in the X basis.
    :return: A program that has randomly placed single qubit and two qubit gates to some depth.
    """
    prog = Program()

    if in_x_basis:
        prep_gate = H
    else:
        # Install identity on all qubits so that we can find all the qubits from prog.get_qubits().
        # Otherwise if the circuit happens to be identity on a particular qubit you will get
        # not get that qubit from get_qubits. Worse, if the entire program is identity you will
        # get the empty set. Do not delete this!!
        prep_gate = I

    prog += [prep_gate(qubit) for qubit in list(graph.nodes)]

    for ddx in range(1, depth + 1):
        # random one qubit gates
        prog += random_single_bit_gates(graph, in_x_basis)
        # random two qubit gates
        prog += random_two_bit_gates(graph, in_x_basis)

    # if in X basis change back to Z basis for measurement
    if in_x_basis:
        prog += [prep_gate(qubit) for qubit in list(graph.nodes)]

    return prog

def get_error_hamming_distance_from_results(perfect_bit_string,results):
    """Get the hamming weight of the error vector (number of bits flipped between output and
    expected answer).

    :param perfect_bit_string: a np.ndarray with shape (1,number_of_bits)
    :param results: a np.ndarray with shape (num_shots,number_of_bits)
    :return: a list of length num_shots containing the hamming weight
    """
    num_shots, n_bits = results.shape
    _, pn_bits = perfect_bit_string.shape
    if n_bits != pn_bits:
        raise ValueError("Bit strings are not equal length, check you are runing on the same graph")
    wt = []
    # loop over all results
    for shot in results:
        wt.append(n_bits * hamming(perfect_bit_string, shot))
    return wt

def get_error_hamming_distributions_from_list(wt_list, n_bits):
    """ Get the distribution of the hamming weight of the error vector.

    :param wt_list:  a list of length num_shots containing the hamming weight.
    :param n_bits:  the number of bit in the original binary strings. The hamming weight is an
    integer between 0 and n_bits.
    :return: the relative frequency of observing each hamming weight
    """
    num_shots = len(wt_list)

    if n_bits < max(wt_list):
        raise ValueError("Hamming weight can't be larger than the number of bits in a string.")

    hamming_wt_distrs = []
    hamming_wt_distr = [0. for _ in range(n_bits+1)]
    # record the fraction of shots that resulted in an error of the given weight
    for wdx in range(n_bits):
        hamming_wt_distr[int(wdx)] = wt_list.count(wdx)/num_shots
    return hamming_wt_distr


def hamming_dist_rand(num_bits: int, pad: int = 0):
    '''Return a list representing the Hamming distribution of
    a particular bit string, of length num_bits, to randomly drawn bits.

    :param num_bits: number of bits in string
    :param pad: number of zero elements to pad
    returns: list of hamming weights with zero padding
    '''
    N = 2 ** num_bits
    pr = [comb(num_bits, ndx)/(2**num_bits) for ndx in range(0, num_bits + 1)]
    padding = [0 for pdx in range(0, pad)]
    return flatten_list([pr, padding])


def flatten_list(xlist):
    '''Flattens a list of lists.

    :param xlist: list of lists
    :returns: a flattend list
    '''
    return [item for sublist in xlist for item in sublist]
