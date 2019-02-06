from typing import List
import itertools
import numpy as np
from scipy.spatial.distance import hamming
import scipy.interpolate
from scipy.special import comb
import networkx as nx
import random
import pandas as pd


from pyquil.gates import CNOT, CCNOT, X, Z, I, H, CZ, MEASURE, RESET
from pyquil.quilbase import Pragma
from pyquil.quil import Program
from pyquil.api import QuantumComputer

from forest_benchmarking.compilation import CNOT_X_basis


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
    # do the two coloring?
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
        # get the empty set. Do not delete this!
        prep_gate = I

    prog += [prep_gate(qubit) for qubit in list(graph.nodes)]

    for ddx in range(1, depth + 1):
        # preserve block ensures the compiler doesn't compile the circuit away
        prog += Pragma('PRESERVE_BLOCK')
        # random one qubit gates
        prog += random_single_bit_gates(graph, in_x_basis)
        # random two qubit gates
        prog += random_two_bit_gates(graph, in_x_basis)
        prog += Pragma('END_PRESERVE_BLOCK')

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

def get_random_classical_circuit_results(qc_perfect: QuantumComputer,
                                         qc_noisy: QuantumComputer,
                                         circuit_depth: int,
                                         circuit_width: int,
                                         num_rand_subgraphs: int = 10,
                                         num_shots_per_circuit: int = 100,
                                         in_x_basis: bool = False,
                                         use_active_reset: bool = False):
    '''
    Convenient wrapper for collecting the results of running classical random circuits on a
    particular lattice.

    It will run a series of random circuits with widths from [1, ...,circuit_width] and depths
    from [1, ..., circuit_depth].

    :param qc_perfect: the "perfect" quantum resource (QVM) to determine the true outcome.
    :param qc_noisy: the noisy quantum resource (QPU or QVM) to
    :param circuit_depth: maximum depth of quantum circuit
    :param circuit_width: maximum width of quantum circuit
    :param num_rand_subgraphs: number of random circuits of circuit_width to be sampled
    :param num_shots_per_circuit: number of shots per random circuit
    :param in_x_basis: performs the random circuit in the x basis
    :param use_active_reset: whether or not to use active reset. Doing so will speed up execution
        on a QPU.
    :return: the data as a list of dicts with keys 'depth', 'width', and 'hamming_dist'.
    '''
    if qc_perfect.name == qc_noisy.name:
        raise ValueError("The noisy and perfect device can't be the same device.")

    # get the networkx graph of the lattice
    G = qc_perfect.qubit_topology()

    if circuit_width > len(G.nodes):
        raise ValueError("You must have circuit widths less than or equal to the number of qubits on a lattice.")

    # add active reset
    reset_prog = Program()
    if use_active_reset:
        reset_prog += RESET()

    data = []
    # loop over different graph sizes
    for depth, subgraph_size in itertools.product(range(1, circuit_depth+1),
                                                  range(1, circuit_width+1)):

        list_of_graphs = generate_connected_subgraphs(G, subgraph_size)
        wt = []
        for kdx in range(1, num_rand_subgraphs+1):
            # randomly choose a lattice from list
            lattice = random.choice(list_of_graphs)
            prog = generate_random_classial_circuit_with_depth(lattice, depth, in_x_basis)

            # perfect
            perfect_bitstring = qc_perfect.run_and_measure(prog, trials=1)
            perfect_bitstring_array = np.vstack(perfect_bitstring[q] for q in prog.get_qubits()).T

            # run on hardware or noisy QVM
            # only need to pre append active reset on something that may run on the hardware
            actual_bitstring = qc_noisy.run_and_measure(reset_prog+prog, trials=num_shots_per_circuit)
            actual_bitstring_array = np.vstack(actual_bitstring[q] for q in prog.get_qubits()).T
            wt.append(get_error_hamming_distance_from_results(perfect_bitstring_array, actual_bitstring_array))

        # for each graph size flatten the results
        wt_flat = flatten_list(wt)
        hamming_wt_distr = get_error_hamming_distributions_from_list(wt_flat, subgraph_size
                                                                    )
        # list of dicts. The keys are (depth, width, hamming_dist)
        data.append({'depth': depth, 'width': subgraph_size, 'hamming_dist': hamming_wt_distr})
    return data


# helper functions to manipulate the dataframes
def get_hamming_dist(df: pd.DataFrame, depth_val: int, width_val: int):
    '''
    Get  Hamming distance from a dataframe for a particular depth and width.

    :param df: dataframe generated from data from 'get_random_classical_circuit_results'
    :param depth_val: depth of quantum circuit
    :param width_val: width of quantum circuit
    :return: smaller dataframe
    '''
    idx = df.depth== depth_val
    jdx = df.width== width_val
    return df[idx&jdx].reset_index(drop=True)

def get_hamming_dists_fn_width(df: pd.DataFrame, depth_val: int):
    '''
    Get  Hamming distance from a dataframe for a particular depth.

    :param df: dataframe generated from data from 'get_random_classical_circuit_results'
    :param depth_val: depth of quantum circuit
    :return: smaller dataframe
    '''
    idx = df.depth== depth_val
    return df[idx].reset_index(drop=True)

def get_hamming_dists_fn_depth(df: pd.DataFrame, width_val: int):
    '''
    Get  Hamming distance from a dataframe for a particular width.

    :param df: dataframe generated from data from 'get_random_classical_circuit_results'
    :param width_val: width of quantum circuit
    :return: smaller dataframe
    '''
    jdx = df.width== width_val
    return df[jdx].reset_index(drop=True)

def basement_function(number: float):
    '''
    Once you are in the basement you can't go lower.
                                /
                                | 0,    if number <= 0
    basement_function(number) = |
                                | floor(number), if number > 0
                                \
    :param number: the basement function is applied to this number.
    :returns: basement of the number
    '''
    if number <= 0.0:
        basement_of_number = 0.0
    else:
        basement_of_number = np.floor(number)
    return basement_of_number

def interpolate_2d_landscape(points, values, resolution=200, interp_method='nearest'):
    """
    Convenience function for interpolating a list of points and corresponding list of values
    onto a 2D meshgrid suitable for plotting.

    See :py:func:`plot_2d_landscape`.

    :param points: A numpy array where the first column is x values and the second column
        is y values
    :param values: The value at each point (z)
    :param resolution: The number of points per side in the interpolated meshgrid
    :param interp_method: The scheme used for interpolation. "cubic" or "linear" will give
        you a prettier picture but "nearest" will prevent you from overconfidence.
    :return: meshgrid arrays (xx, yy, zz) suitable for plotting.
    """
    xx, yy = np.meshgrid(
        np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), resolution),
        np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), resolution),
    )
    zz = scipy.interpolate.griddata(points, values, (xx, yy), method=interp_method)
    return xx, yy, zz