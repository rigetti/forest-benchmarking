from typing import Tuple, Sequence, Callable, Any, List
from copy import copy
import networkx as nx
import numpy as np
import random
import itertools
import pandas as pd
from scipy.spatial.distance import hamming
from scipy.special import comb
from dataclasses import dataclass

from pyquil.quilbase import Pragma, Gate, DefGate
from pyquil.quil import Program, address_qubits, merge_programs
from pyquil.api import QuantumComputer, BenchmarkConnection
from pyquil.gates import CNOT, CCNOT, Z, X, I, H, CZ, MEASURE, RESET
from pyquil.unitary_tools import permutation_arbitrary
from rpcq.messages import TargetDevice
from rpcq._utils import RPCErrorError

from forest.benchmarking.randomized_benchmarking import get_rb_gateset
from forest.benchmarking.distance_measures import total_variation_distance as tvd
from forest.benchmarking.random_operators import haar_rand_unitary
from forest.benchmarking.compilation import basic_compile

#
# @dataclass(order=True)
# class Slice:
#     index: int
#     gates: Tuple[Program]
#     needs_compilation: bool = True

    # def __str__(self):
    #     return f'Index {self.index}:\n' + '\n'.join([str(comp) for comp in self.components]) + '\n'

#TODO: make concatenation of slices possible.

# @dataclass(order=True)
# class Layer:
#     depth: int
#     slices: Tuple[Slice]
#     needs_compilation: bool = True

    # def __str__(self):
    #     return f'Depth {self.depth}:\n' + '\n'.join([str(comp) for comp in self.components]) + '\n'


# @dataclass
# class Circuit:
#     layers: Tuple[Layer]
#     graph: nx.Graph
#     needs_compilation: bool = True
#     name: str = None

    # def __str__(self):
    #     return '\n'.join([str(lyr) for lyr in self.layers]) + '\n'


@dataclass
class CircuitTemplate:
    generators: List[Callable]
    #TODO: could allow CircuitTemplates, allow definition of depth, subunits...
    #TODO: add compilation?

    def append(self, other):
        self.generators += other.generators

    def __add__(self, other):
        """
        Concatenate two circuits together, returning a new one.

        :param Circuit other: Another circuit to add to this one.
        :return: A newly concatenated circuit.
        :rtype: Program
        """
        ckt = CircuitTemplate(self.generators)
        ckt.append(other)
        return ckt

    def __iadd__(self, other):
        """
        Concatenate two circuits together using +=, returning a new one.
        """
        self.append(other)
        return self

    def sample(self, qc, graph, width, depth, sequence = None, index=0):
        if sequence is None:
            sequence = []
        while index < depth:
            for generator in self.generators:
                if index == depth:
                    break
                prog, index = generator(qc, graph, width, depth, sequence, index)
                sequence.append(prog)
        return sequence

    # def __str__(self):
    #     return f'Depth {self.depth}:\n' + '\n'.join([str(comp) for comp in self.components]) + '\n'


# @dataclass(order=True)
# class LayerTemplate:
#     depth: int
#     slices: Tuple[SliceTemplate]
#     sandwich: bool = False

    # def __str__(self):
    #     return f'Depth {self.depth}:\n' + '\n'.join([str(comp) for comp in self.components]) + '\n'


# @dataclass
# class CircuitTemplate:
#     slices: Tuple[SliceTemplate]
#     graph: nx.Graph
#     sandwich: bool = False
#     name: str = None

    # def __str__(self):
    #     return '\n'.join([str(lyr) for lyr in self.layers]) + '\n'

# ==================================================================================================
# Gate Sets
# ==================================================================================================
def random_single_qubit_gates(graph: nx.Graph, gates: Sequence[Gate]):
    """
    Create a program comprised of single qubit gates randomly placed on the nodes of the
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


def random_two_qubit_gates(graph: nx.Graph, gates:  Sequence[Gate]):
    """
    Write a program to randomly place two qubit gates on edges of the specified graph.

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


def random_single_qubit_cliffords(graph: nx.Graph, bm: BenchmarkConnection):
    """
    Create a program comprised of single qubit Cliffords gates randomly placed on the nodes of
    the specified graph. Each uniformly random choice of Clifford is implemented in the native 
    gateset.

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


def random_two_qubit_cliffords(graph: nx.Graph, bm: BenchmarkConnection):
    """
    Write a program to place random two qubit Cliffords gates on edges of the graph.

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


def random_permutation(graph: nx.Graph, width):
    #TODO: find another way; this is too slow
    qubits = list(graph.nodes)
    measure_qubits = qubits[:width]  # arbitrarily pick the first width-many nodes
    permutation = np.random.permutation(range(len(measure_qubits)))
    matrix = permutation_arbitrary(permutation, len(measure_qubits))[0]

    gate_definition = DefGate("Perm" + "".join([str(measure_qubits[idx]) for idx in permutation]), matrix)
    PERMUTE = gate_definition.get_constructor()
    p = Program()
    p += gate_definition
    p += PERMUTE(*measure_qubits)
    return p


def random_su2_pairs(graph: nx.Graph, width):
    qubits = list(graph.nodes)[:width]  # arbitrarily pick the first width-many nodes
    gates = []
    for q1, q2 in zip(qubits[::2], qubits[1::2]):
        matrix = haar_rand_unitary(4)
        gate_definition = DefGate(f"RSU2({q1},{q2})", matrix)
        RSU2 = gate_definition.get_constructor()
        p = Program()
        p += gate_definition
        p += RSU2(q1, q2)
        gates.append(p)
    return gates


def quantum_volume_compilation(qc, graph, width, depth, sequence):
    prog = merge_programs(sequence)
    qubits = list(graph.nodes)
    measure_qubits = qubits[:width]  # arbitrarily pick the first width-many nodes

    ro = prog.declare("ro", "BIT", len(measure_qubits))
    for idx, qubit in enumerate(measure_qubits):
        prog.measure(qubit, ro[idx])

    # restrict compilation to chosen qubits
    isa_dict = qc.device.get_isa().to_dict()
    single_qs = isa_dict['1Q']
    two_qs = isa_dict['2Q']

    new_1q = {}
    for key, val in single_qs.items():
        if int(key) in qubits:
            new_1q[key] = val
    new_2q = {}
    for key, val in two_qs.items():
        q1, q2 = key.split('-')
        if int(q1) in qubits and int(q2) in qubits:
            new_2q[key] = val

    new_isa = {'1Q': new_1q, '2Q': new_2q}

    new_compiler = copy(qc.compiler)
    new_compiler.target_device = TargetDevice(isa=new_isa, specs=qc.device.get_specs().to_dict())
    # try to compile with the restricted qubit topology
    try:
        native_quil = new_compiler.quil_to_native_quil(prog)
    except RPCErrorError as e:
        if "Multiqubit instruction requested between disconnected components of the QPU graph:" \
                in str(e):
            raise ValueError("naive_program_generator could not generate a program using only the "
                             "qubits supplied; expand the set of allowed qubits or supply "
                             "a custom program_generator.")
        raise

    return native_quil

# ===========================================
# Layer tools
# ==================================================================================================
#
#
# def layer_1q_and_2q_rand_cliff(bm: BenchmarkConnection,
#                                graph: nx.Graph,
#                                layer_dagger: bool = False):
#     """
#     Creates a layer of random one qubit Cliffords followed by random two qubit Cliffords.
#
#     :param bm: A benchmark connection that will do the grunt work of generating the Cliffords
#     :param graph:  The graph. Nodes are used as arguments to gates, so they should be qubit-like.
#     :param layer_dagger: Bool if true will add the dagger to the layer, making the layer
#         effectively the identity
#     :return: program
#     """
#     prog = Program()
#     prog += random_single_qubit_cliffords(bm, graph)
#     prog += random_two_qubit_cliffords(bm, graph)
#     if layer_dagger:
#         prog += prog.dagger()
#     return prog
#
#
# def layer_1q_and_2q_rand_gates(graph: nx.Graph,
#                                one_q_gates,
#                                two_q_gates,
#                                layer_dagger: bool = False):
#     """
#     You pass in two lists of one and two qubit gates. This function creates a layer of random one
#     qubit gates followed by random two qubit gates
#
#     :param graph: The graph. Nodes are used as arguments to gates, so they should be qubit-like.
#     :param one_q_gates: list of one qubit gates
#     :param two_q_gates: list of two qubit gates e.g. [CZ, ID]
#     :param layer_dagger: Bool if true will add the dagger to the layer, making the layer
#         effectively the identity
#     :return: program
#     """
#     prog = Program()
#     prog += random_single_qubit_gates(graph, one_q_gates)
#     prog += random_two_qubit_gates(graph, two_q_gates)
#     if layer_dagger:
#         prog += prog.dagger()
#     return prog


# ==================================================================================================
# Sandwich tools
# ==================================================================================================
# def circuit_sandwich_rand_gates(graph: nx.Graph,
#                                 circuit_depth: int,
#                                 one_q_gates: list,
#                                 two_q_gates: list,
#                                 layer_dagger: bool = False,
#                                 sandwich_dagger: bool = False):
#     """
#     Create a sandwich circuit by adding layers.
#
#     :param graph: The graph. Nodes are used as arguments to gates, so they should be qubit-like.
#     :param circuit_depth: maximum depth of quantum circuit
#     :param one_q_gates: list of one qubit gates
#     :param two_q_gates: list of two qubit gates e.g. [CZ, ID]
#     :param layer_dagger: Bool if true will add the dagger to the layer, making the layer
#     :param sandwich_dagger: Bool if true the second half of the circuit will be the inverse of
#     the first.
#     :return: program
#     """
#     total_prog = Program()
#     total_prog += pre_trival(graph)
#
#     if sandwich_dagger:
#         circuit_depth = int(np.floor(circuit_depth / 2))
#
#     layer_progs = Program()
#     for _ in range(circuit_depth):
#         layer_progs += layer_1q_and_2q_rand_gates(graph,
#                                                   one_q_gates,
#                                                   two_q_gates,
#                                                   layer_dagger)
#     if sandwich_dagger:
#         layer_progs += layer_progs.dagger()
#
#     total_prog += layer_progs
#     total_prog += post_trival()
#     return total_prog
#
#
# def circuit_sandwich_clifford(bm: BenchmarkConnection,
#                               graph: nx.Graph,
#                               circuit_depth: int,
#                               layer_dagger: bool = False,
#                               sandwich_dagger: bool = False):
#     """
#
#     :param bm: A benchmark connection that will do the grunt work of generating the Cliffords
#     :param graph: The graph. Nodes are used as arguments to gates, so they should be qubit-like.
#     :param circuit_depth: maximum depth of quantum circuit
#     :param layer_dagger: Bool if true will add the dagger to the layer, making the layer
#     :param sandwich_dagger: Bool if true the second half of the circuit will be the inverse of
#     the first.
#     :return: program
#     """
#     total_prog = Program()
#
#     total_prog += pre_trival(graph)
#
#     if sandwich_dagger:
#         circuit_depth = int(np.floor(circuit_depth / 2))
#
#     layer_progs = Program()
#     for _ in range(circuit_depth):
#         layer_progs += layer_1q_and_2q_rand_cliff(bm, graph, layer_dagger)
#     if sandwich_dagger:
#         layer_progs += layer_progs.dagger()
#
#     total_prog += layer_progs
#     total_prog += post_trival()
#     return total_prog
#

# ==================================================================================================
# Generate and Acquire functions
# ==================================================================================================


def generate_sandwich_circuits_experiments(qc_noisy: QuantumComputer,
                                           circuit_depth: int,
                                           circuit_width: int,
                                           circuit_sandwich: callable,
                                           layer_dagger: bool = False,
                                           sandwich_dagger: bool = False,
                                           num_rand_subgraphs: int = 10,
                                           # peter claims that no speed diff 800 shots
                                           num_shots_per_circuit: int = 100,
                                           use_active_reset: bool = False) -> pd.DataFrame:
    """
    Return a DataFrame where the rows contain all the information needed to run random circuits
    of a certain width and depth on a particular lattice.

    :param qc_noisy: the noisy quantum resource (QPU or QVM)
    :param circuit_depth: maximum depth of quantum circuit
    :param circuit_width: maximum width of quantum circuit
    :param circuit_sandwich: callable. Regardless of the original arguments the function here
    must only have graph, circuit_depth, layer_dagger, and sandwich_dagger as remainig keywords.
    :param num_rand_subgraphs: number of random circuits of circuit_width to be sampled
    :param num_shots_per_circuit: number of shots per random circuit
    :param use_active_reset: if True uses active reset. Doing so will speed up execution on a QPU.
    :return: pandas DataFrame
    """
    # get the networkx graph of the lattice
    G = qc_noisy.qubit_topology()

    if circuit_width > len(G.nodes):
        raise ValueError("You must have circuit widths less than or equal to the number of qubits "
                         "on a lattice.")

    experiment = []
    # loop over different graph sizes
    for subgraph_size in range(1, circuit_width + 1):
        list_of_graphs = generate_connected_subgraphs(G, subgraph_size)

        for depth in range(1, circuit_depth + 1):
            for _ in range(num_rand_subgraphs):
                # randomly choose a lattice from list
                lattice = random.choice(list_of_graphs)
                prog = circuit_sandwich(graph=lattice,
                                        circuit_depth=depth,
                                        layer_dagger=layer_dagger,
                                        sandwich_dagger=sandwich_dagger)

                experiment.append({'Depth': depth,
                                   'Width': subgraph_size,
                                   'Lattice': lattice,
                                   'Layer Dagger': layer_dagger,
                                   'Sandwich Dagger': sandwich_dagger,
                                   'Active Reset': use_active_reset,
                                   'Program': prog,
                                   'Trials': num_shots_per_circuit,
                                   })
    return pd.DataFrame(experiment)


def acquire_circuit_sandwich_data(qc_noisy: QuantumComputer,
                                  circ_sand_expt: pd.DataFrame) -> pd.DataFrame:
    """
    Convenient wrapper for collecting the results of running circuits sandwiches on a
    particular lattice.

    It will run a series of random circuits with widths from [1, ...,circuit_width] and depths
    from [1, ..., circuit_depth].


    :param qc_noisy: the noisy quantum resource (QPU or QVM) to
    :param circ_sand_expt: pandas DataFrame where the rows contain experiments
    :return: pandas DataFrame
    """
    #:param qc_perfect: the "perfect" quantum resource (QVM) to determine the true outcome.
    # if qc_perfect.name == qc_noisy.name:
    #    raise ValueError("The noisy and perfect device can't be the same device.")

    data = []
    for index, row in circ_sand_expt.iterrows():
        prog = row['Program']
        use_active_reset = row['Active Reset']
        num_shots_per_circuit = row['Trials']

        # run on perfect QVM or Wavefunction simulator
        # perfect_bitstring = qc_perfect.run_and_measure(prog, trials=1)
        # perfect_bitstring_array = np.vstack(perfect_bitstring[q] for q in prog.get_qubits()).T

        # add active reset
        reset_prog = Program()
        if use_active_reset:
            reset_prog += RESET()

        # run on hardware or noisy QVM
        # only need to pre append active reset on something that may run on the hardware
        actual_bitstring = qc_noisy.run_and_measure(reset_prog + prog, trials=num_shots_per_circuit)
        actual_bitstring_array = np.vstack(actual_bitstring[q] for q in prog.get_qubits()).T

        # list of dicts.
        data.append({'Depth': row['Depth'],
                     'Width': row['Width'],
                     'Lattice': row['Lattice'],
                     # 'In X basis': row['In X basis'],
                     'Active Reset': use_active_reset,
                     'Program': prog,
                     'Trials': num_shots_per_circuit,
                     # 'Answer': perfect_bitstring_array,
                     'Samples': actual_bitstring_array,
                     })
    return pd.DataFrame(data)


# ==================================================================================================
# Analysis
# ==================================================================================================
def estimate_random_classical_circuit_errors(qc_perfect: QuantumComputer,
                                             df: pd.DataFrame) -> pd.DataFrame:
    """
    asdf

    :param df: pandas DataFrame containing experimental results
    :return: pandas DataFrame containing estiamted errors and experimental results
    """

    results = []
    for _, row in df.iterrows():
        wt = []
        prog = row['Program']
        # run on perfect QVM or Wavefunction simulator
        perfect_bitstring = qc_perfect.run_and_measure(prog, trials=1)
        perfect_bitstring_array = np.vstack(perfect_bitstring[q] for q in prog.get_qubits()).T
        # perfect_bitstring_array = np.asarray(row['Answer'])
        actual_bitstring_array = np.asarray(row['Samples'])
        wt.append(get_error_hamming_distance_from_results(perfect_bitstring_array,
                                                          actual_bitstring_array))
        wt_flat = flatten_list(wt)

        # Hamming weight distributions
        wt_dist_data = np.asarray(
            get_error_hamming_distributions_from_list(wt_flat, row['Width']))  # data
        wt_dist_rand = np.asarray(hamming_dist_rand(row['Width']))  # random guessing
        wt_dist_ideal = np.zeros_like(wt_dist_rand)  # perfect
        wt_dist_ideal[0] = 1

        # Total variation distance
        tvd_data_ideal = tvd(wt_dist_data, wt_dist_ideal)
        tvd_data_rand = tvd(wt_dist_data, wt_dist_rand)

        # Probablity of success
        pr_suc_data = wt_dist_data[0]
        pr_suc_rand = wt_dist_rand[0]

        # Probablity of success with basement[ log_2(width) - 1 ] errors
        # I.e. error when you allow for a logarithmic number of bit flips from the answer
        num_bit_flips_allowed_from_answer = int(basement_function(np.log2(row['Width']) - 1))
        pr_suc_log_err_data = sum(
            [wt_dist_data[idx] for idx in range(0, num_bit_flips_allowed_from_answer + 1)])
        pr_suc_log_err_rand = sum(
            [wt_dist_rand[idx] for idx in range(0, num_bit_flips_allowed_from_answer + 1)])

        results.append({'Depth': row['Depth'],
                        'Width': row['Width'],
                        'Lattice': row['Lattice'],
                        # 'In X basis': row['In X basis'],
                        'Active Reset': row['Active Reset'],
                        'Program': row['Program'],
                        'Trials': row['Trials'],
                        'Answer': perfect_bitstring_array,
                        'Samples': actual_bitstring_array,
                        'Hamming dist. data': wt_dist_data,
                        'Hamming dist. rand': wt_dist_rand,
                        'Hamming dist. ideal': wt_dist_ideal,
                        'TVD(data, ideal)': tvd_data_ideal,
                        'TVD(data, rand)': tvd_data_rand,
                        'Pr. success data': pr_suc_data,
                        'Pr. success rand': pr_suc_rand,
                        'loge = basement[log_2(Width)-1]': num_bit_flips_allowed_from_answer,
                        'Pr. success loge data': pr_suc_log_err_data,
                        'Pr. success loge rand': pr_suc_log_err_rand,
                        })
    return pd.DataFrame(results)


def get_error_hamming_distance_from_results(perfect_bit_string, results):
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

    hamming_wt_distr = [0. for _ in range(n_bits + 1)]
    # record the fraction of shots that resulted in an error of the given weight
    for wdx in range(n_bits):
        hamming_wt_distr[int(wdx)] = wt_list.count(wdx) / num_shots
    return hamming_wt_distr


def hamming_dist_rand(num_bits: int, pad: int = 0):
    """Return a list representing the Hamming distribution of
    a particular bit string, of length num_bits, to randomly drawn bits.

    :param num_bits: number of bits in string
    :param pad: number of zero elements to pad
    returns: list of hamming weights with zero padding
    """
    N = 2 ** num_bits
    pr = [comb(num_bits, ndx) / (2 ** num_bits) for ndx in range(0, num_bits + 1)]
    padding = [0 for _ in range(pad)]
    return flatten_list([pr, padding])


def flatten_list(xlist):
    """Flattens a list of lists.

    :param xlist: list of lists
    :returns: a flattened list
    """
    return [item for sublist in xlist for item in sublist]


# helper functions to manipulate the dataframes
def get_hamming_dist(df: pd.DataFrame, depth_val: int, width_val: int):
    """
    Get  Hamming distance from a dataframe for a particular depth and width.

    :param df: dataframe generated from data from 'get_random_classical_circuit_results'
    :param depth_val: depth of quantum circuit
    :param width_val: width of quantum circuit
    :return: smaller dataframe
    """
    idx = df.Depth == depth_val
    jdx = df.Width == width_val
    return df[idx & jdx].reset_index(drop=True)


def get_hamming_dists_fn_width(df: pd.DataFrame, depth_val: int):
    """
    Get  Hamming distance from a dataframe for a particular depth.

    :param df: dataframe generated from data from 'get_random_classical_circuit_results'
    :param depth_val: depth of quantum circuit
    :return: smaller dataframe
    """
    idx = df.Depth == depth_val
    return df[idx].reset_index(drop=True)


def get_hamming_dists_fn_depth(df: pd.DataFrame, width_val: int):
    """
    Get  Hamming distance from a dataframe for a particular width.

    :param df: dataframe generated from data from 'get_random_classical_circuit_results'
    :param width_val: width of quantum circuit
    :return: smaller dataframe
    """
    jdx = df.Width == width_val
    return df[jdx].reset_index(drop=True)


def basement_function(number: float):
    """
    Once you are in the basement you can't go lower. Defined as

    basement_function(number) = |floor(number)*heaviside(number,0)|,

    where heaviside(number,0) implies the value of the step function is
    zero if number is zero.

    :param number: the basement function is applied to this number.
    :returns: basement of the number
    """
    basement_of_number = np.abs(np.floor(number) * np.heaviside(number, 0))
    return basement_of_number


def CNOT_X_basis(control, target) -> Program:
    """
    The CNOT in the X basis, i.e.

    CNOTX = |+X+| * I + |-X-| * Z

    where |+> and |-> are the +/- eigenstate of the Pauli X operator and * denotes a tensor product.

    :param control: qubit label
    :param target: qubit label
    :return: program
    """
    prog = Program()
    prog += H(control)
    prog += CZ(control, target)
    prog += H(control)
    return prog


# ==================================================================================================
# Graph tools
# ==================================================================================================


def generate_connected_subgraphs(G: nx.Graph, n_vert: int):
    """
    Given a lattice on the QPU or QVM, specified by a networkx graph, return a list of all
    subgraphs with n_vert connect vertices.

    :params n_vert: number of vertices of connected subgraph.
    :params G: networkx Graph
    :returns: list of subgraphs with n_vert connected vertices
    """
    subgraph_list = []
    for sub_nodes in itertools.combinations(G.nodes(), n_vert):
        subg = G.subgraph(sub_nodes)
        if nx.is_connected(subg):
            subgraph_list.append(subg)
    return subgraph_list
