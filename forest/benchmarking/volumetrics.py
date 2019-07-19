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
from functools import partial

from pyquil.quilbase import Pragma, Gate, DefGate, DefPermutationGate
from pyquil.quilatom import QubitPlaceholder
from pyquil.quil import Program, address_qubits, merge_programs
from pyquil.api import QuantumComputer, BenchmarkConnection
from pyquil.gates import CNOT, CCNOT, Z, X, I, H, CZ, MEASURE, RESET
from rpcq.messages import TargetDevice
from rpcq._utils import RPCErrorError

from forest.benchmarking.randomized_benchmarking import get_rb_gateset
from forest.benchmarking.distance_measures import total_variation_distance as tvd
from forest.benchmarking.operator_tools.random_operators import haar_rand_unitary


@dataclass
class CircuitTemplate:
    generators: List[Callable]

    # def create_unit(self):
    #     # returns a function that can be used as a generator in another template
    #     return lambda qc, graph, width, depth, sequence: sum(gen(qc, graph, width, depth,
    #                                                              sequence) for gen in
    #                                                          self.generators)

    def append(self, other):
        self.generators += other.generators

        # TODO: store the pattern?
        # TODO: add reps keyword to pattern?

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

    def sample_sequence(self, graph, repetitions, qc=None, width=None, sequence=None, pattern=None):
        if width is not None:
            graph = random.choice(generate_connected_subgraphs(graph, width))

        if pattern is None:
            # by default sweep over each generator in sequence repetitions many times
            pattern = range(len(self.generators))

        if sequence is None:
            sequence = []

        def _do_pattern(patt):
            for elem in patt:
                if isinstance(elem, int):
                    # the elem is an index; we use the generator at this index to generate the
                    # next program in the sequence
                    sequence.append(self.generators[elem](graph=graph, qc=qc, width=width,
                                                          sequence=sequence))
                elif len(elem) == 2:
                    # elem[0] is a pattern that we will execute elem[1] many times
                    for _ in range(elem[1]):
                        _do_pattern(elem[0])
                else:
                    raise ValueError('Pattern is malformed. A pattern is a list where each element '
                                     'can either be a generator index or a (pattern_i, num) tuple, '
                                     'where num is an integer indicating how many times to '
                                     'repeat the associated pattern_i.')

        for _ in range(repetitions):
            _do_pattern(pattern)

        return sequence

    def sample_program(self, graph, repetitions, qc=None, width=None, sequence = None,
                       pattern = None):
        return merge_programs(self.sample_sequence(graph, repetitions, qc, width, sequence, pattern))


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


def random_single_qubit_cliffords(bm: BenchmarkConnection, graph: nx.Graph):
    """
    Create a program comprised of single qubit Cliffords gates randomly placed on the nodes of
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


def random_two_qubit_cliffords(bm: BenchmarkConnection, graph: nx.Graph):
    """
    Write a program to place random two qubit Cliffords gates on edges of the graph.

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
    # do the two coloring with pragmas?
    # no point until fencing is over
    for edges, clif in zip(graph.edges, rand_cliffords):
        gate = address_qubits(clif, qubit_mapping={q_placeholders[0]: edges[0],
                                                   q_placeholders[1]: edges[1]})
        prog += gate
    return prog


def dagger_all_prior(sequence: List[Program]):
    return merge_programs(sequence).dagger()


def dagger_previous(sequence: List[Program]):
    return sequence[-1].dagger()


def _qubit_perm_to_bitstring_perm(qubit_permutation: List[int]):
    bitstring_permutation = []
    for bitstring in range(2**len(qubit_permutation)):
        permuted_bitstring = 0
        for idx, q in enumerate(qubit_permutation):
            permuted_bitstring |= ((bitstring >> q) & 1) << idx
        bitstring_permutation.append(permuted_bitstring)
    return bitstring_permutation


def random_qubit_permutation(graph: nx.Graph):
    qubits = list(graph.nodes)
    permutation = list(np.random.permutation(range(len(qubits))))

    gate_definition = DefPermutationGate("Perm" + "".join([str(q) for q in permutation]),
                                         _qubit_perm_to_bitstring_perm(permutation))
    PERMUTE = gate_definition.get_constructor()
    p = Program()
    p += gate_definition
    p += PERMUTE(*qubits)
    return p


def random_su4_pairs(graph: nx.Graph):
    qubits = list(graph.nodes)
    prog = Program()
    for q1, q2 in zip(qubits[::2], qubits[1::2]):
        matrix = haar_rand_unitary(4)
        gate_definition = DefGate(f"RSU4_{q1}_{q2}", matrix)
        RSU4 = gate_definition.get_constructor()
        prog += gate_definition
        prog += RSU4(q1, q2)
    return prog


def graph_restricted_compilation(qc, graph, program):
    qubits = list(graph.nodes)

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
        native_quil = new_compiler.quil_to_native_quil(program)
    except RPCErrorError as e:
        if "Multiqubit instruction requested between disconnected components of the QPU graph:" \
                in str(e):
            raise ValueError("The program could not be compiled onto the given subgraph.")
        raise

    return native_quil


###
# Templates
###

def get_rand_1q_template(gates: Sequence[Gate]):
    def func(graph, **kwargs):
        partial_func = partial(random_single_qubit_gates, gates=gates)
        return partial_func(graph)
    return CircuitTemplate([func])


def get_rand_2q_template(gates: Sequence[Gate]):
    def func(graph, **kwargs):
        partial_func = partial(random_two_qubit_gates, gates=gates)
        return partial_func(graph)
    return CircuitTemplate([func])


def get_rand_1q_cliff_template(bm: BenchmarkConnection):
    def func(graph, **kwargs):
        partial_func = partial(random_single_qubit_cliffords, bm=bm)
        return partial_func(graph=graph)
    return CircuitTemplate([func])


def get_rand_2q_cliff_template(bm: BenchmarkConnection):
    def func(graph, **kwargs):
        partial_func = partial(random_two_qubit_cliffords, bm=bm)
        return partial_func(graph=graph)
    return CircuitTemplate([func])


def get_dagger_all_template():
    def func(qc, sequence, **kwargs):
        prog = dagger_all_prior(sequence)
        native_quil = qc.compiler.quil_to_native_quil(prog)
        # remove gate definition and HALT
        return Program([instr for instr in native_quil.instructions][:-1])
    return CircuitTemplate([func])


def get_dagger_previous():
    def func(qc, sequence, **kwargs):
        prog = dagger_previous(sequence)
        native_quil = qc.compiler.quil_to_native_quil(prog)
        # remove gate definition and HALT
        return Program([instr for instr in native_quil.instructions][:-1])
    return CircuitTemplate([func])


def get_rand_qubit_perm_template():
    def func(graph, qc, **kwargs):
        prog = random_qubit_permutation(graph)
        native_quil = qc.compiler.quil_to_native_quil(prog)
        # remove gate definition and HALT
        return Program([instr for instr in native_quil.instructions][:-1])
    return CircuitTemplate([func])


def get_rand_su4_template():
    def func(graph, qc, **kwargs):
        prog = random_su4_pairs(graph)
        native_quil = graph_restricted_compilation(qc, graph, prog)
        # remove gate definitions and HALT
        return Program([instr for instr in native_quil.instructions][:-1])
    return CircuitTemplate([func])


def generate_volumetric_program_array(qc: QuantumComputer, ckt: CircuitTemplate, widths: List[int],
                                      depths: List[int], num_circuit_samples: int,
                                      graph: nx.Graph = None, pattern = None):
    if graph is None:
        graph = qc.qubit_topology()

    programs = {width: {depth: []} for width in widths for depth in depths}

    for width, depth_array in programs.items():
        for depth, prog_list in depth_array.items():
            for _ in range(num_circuit_samples):
                prog = ckt.sample_program(graph, repetitions=depth, width=width,
                                          qc=qc, pattern=pattern)
                prog_list.append(prog)

    return programs


def acquire_volumetric_data(qc: QuantumComputer, program_array, num_shots: int = 500,
                                  use_active_reset:  bool = False,
                                  use_compiler: bool = False):
    reset_prog = Program()
    if use_active_reset:
        reset_prog += RESET()

    results = {width: {depth: []} for width, depth_array in program_array.items()
               for depth in depth_array.keys()}

    for width, depth_array in program_array.items():
        for depth, prog_list in depth_array.items():
            for program in prog_list:
                prog = program.copy()

                # TODO: provide some way to ensure spectator qubits measured when relevant.
                qubits = sorted(list(program.get_qubits()))

                ro = prog.declare('ro', 'BIT', len(qubits))
                for idx, q in enumerate(qubits):
                    prog += MEASURE(q, ro[idx])

                prog.wrap_in_numshots_loop(num_shots)

                if use_compiler:
                    prog = qc.compiler.quil_to_native_quil(prog)

                exe = qc.compiler.native_quil_to_executable(prog)
                shots = qc.run(exe)
                results[width][depth].append(shots)

    return results


# def do_volumetric_measurements(qc: QuantumComputer, ckt: CircuitTemplate, widths: List[int],
#                                   depths: List[int],
#                                   num_circuit_samples: int, graph: nx.Graph = None, pattern = None,
#                                   num_shots: int = 500,
#                                   use_active_reset:  bool = False,
#                                   compile_circuits: bool = False):
#
#
#     prog_array = generate_volumetric_program_array(qc, ckt, widths, depths, num_circuit_samples,
#                                                    graph, pattern)
#
#     return []



# ==================================================================================================
# Analysis
# ==================================================================================================
def get_error_hamming_weight_distributions(noisy_results, perfect_results):
    distrs = {width: {depth: []} for width, depth_array in noisy_results.items()
               for depth in depth_array.keys()}

    for width, depth_array in distrs.items():
        for depth, samples in depth_array.items():

            noisy_ckt_sample_results = noisy_results[width][depth]
            perfect_ckt_sample_results = perfect_results[width][depth]

            for noisy_shots, ideal_result in zip(noisy_ckt_sample_results,
                                                 perfect_ckt_sample_results):

                hamm_dist_per_shot = [hamming_distance(ideal_result, shot) for shot in noisy_shots]

                # Hamming weight distribution
                hamm_wt_distr =  get_hamming_wt_distr_from_list(hamm_dist_per_shot, width)
                samples.append(np.asarray(hamm_wt_distr))
    return distrs
                # TODO: separate these out
                # wt_dist_rand = np.asarray(hamming_dist_rand(width))  # random guessing
                # wt_dist_ideal = np.zeros_like(wt_dist_rand)  # perfect
                # wt_dist_ideal[0] = 1

                # Total variation distance
                # tvd_data_ideal = tvd(wt_dist_data, wt_dist_ideal)
                # tvd_data_rand = tvd(wt_dist_data, wt_dist_rand)

                # Probability of success
                # pr_suc_data = hamm_wt_distr[0]
                # pr_suc_rand = wt_dist_rand[0]

                # Probability of success with basement[ log_2(width) - 1 ] errors
                # I.e. error when you allow for a logarithmic number of bit flips from the answer
                # num_bit_flips_allowed_from_answer = int(basement_function(np.log2(width) - 1))
                # pr_suc_log_err_data = sum(
                #     [wt_dist_data[idx] for idx in range(0, num_bit_flips_allowed_from_answer + 1)])
                # pr_suc_log_err_rand = sum(
                #     [wt_dist_rand[idx] for idx in range(0, num_bit_flips_allowed_from_answer + 1)])
                #
                #
                # sample_stats = {
                #     'Hamming dist. data': wt_dist_data,
                #     'TVD(data, ideal)': tvd_data_ideal,
                #     'TVD(data, rand)': tvd_data_rand,
                #     'Pr. success data': pr_suc_data,
                #     # 'Pr. success rand': pr_suc_rand,
                #     'loge = basement[log_2(Width)-1]': num_bit_flips_allowed_from_answer,
                #     'Pr. success loge data': pr_suc_log_err_data}
                #     # 'Pr. success loge rand': pr_suc_log_err_rand}
                #
                # samples.append(sample_stats)

    # return stats


def hamming_distance(arr1, arr2):
    """
    Compute the hamming distance between arr1 and arr2, or the total number of indices which
    differ between them.

    The hamming distance is equivalently the hamming weight of the 'error vector' between the
    two arrays.

    :return: hamming distance between arr1 and arr2
    """
    n_bits = np.asarray(arr1).size
    if not n_bits == np.asarray(arr2).size:
        raise ValueError('Arrays must be equal size.')

    return hamming(arr1, arr2) * n_bits


def get_hamming_wt_distr_from_list(wt_list, n_bits):
    """
    Get the distribution of the hamming weight of the error vector.

    :param wt_list:  a list of length num_shots containing the hamming weight.
    :param n_bits:  the number of bit in the original binary strings. The hamming weight is an
    integer between 0 and n_bits.
    :return: the relative frequency of observing each hamming weight
    """
    num_shots = len(wt_list)

    if n_bits < max(wt_list):
        raise ValueError("Hamming weight can't be larger than the number of bits in a string.")

    # record the fraction of shots that resulted in an error of the given weight
    return [wt_list.count(weight) / num_shots for weight in range(n_bits + 1)]


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
