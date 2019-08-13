from typing import Tuple, Sequence, Callable, Dict, List, Union
from copy import copy
import networkx as nx
import numpy as np
import random
import itertools
from scipy.spatial.distance import hamming
from scipy.special import comb
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

from pyquil.quilbase import Pragma, Gate, DefGate, DefPermutationGate
from pyquil.quilatom import QubitPlaceholder
from pyquil.quil import Program, address_qubits, merge_programs
from pyquil.api import QuantumComputer, BenchmarkConnection
from pyquil.gates import *
from pyquil.paulis import exponential_map, sX, sZ
from rpcq.messages import TargetDevice
from rpcq._utils import RPCErrorError

from forest.benchmarking.randomized_benchmarking import get_rb_gateset
from forest.benchmarking.distance_measures import total_variation_distance as tvd
from forest.benchmarking.operator_tools.random_operators import haar_rand_unitary


def make_default_pattern(num_generators):
    """
    By default sweep over each generator in sequence n many times

    :param num_generators:
    :return:
    """
    return [(list(range(num_generators)), 'n')]

# TODO: perhaps best for pattern to be sample-time specified, given ambiguity in append; however,
#  it is convenient to keep a persistent state. Appending sequence_transforms is also not well
#  motivated, so instead maybe it is better to remove support for appending CircuitTemplates
#  altogether?

@dataclass
class CircuitTemplate:
    """
    We want to be able to specify various families of circuits and, once specified, randomly
    sample from the family circuits of various width and depth. 'Width' is simply the number of
    qubits. 'Depth' is not simply circuit depth, but rather the number of some repeated group of
    gates that constitute some distinct unit. A depth d circuit could consist of d consecutive
    rounds of random single qubit, then two qubit gates. It could also mean d consecutive
    random Cliffords followed by the d conjugated Cliffords that invert the first d gates.

    Because these families of circuits are quite diverse, specifying the family and drawing
    samples can potentially require a wide variety of parameters. The compiler may be required to
    map an abstract circuit into native quil; a sample acting on a specific qubit topology
    may be desired; the sequence of 'layers' generated so far may be necessary to compute an
    inverse.

    The primary purpose of this class is to sample circuits, which we represent by a list of
    pyquil Programs, or a 'sequence'; this core functionality is found in :func:`sample_sequence`.
    In this function `generators` are applied in series according to the order specified by
    `pattern`. Each call to a generator will contribute an element to the output sequence,
    and some combination of the generators will constitute a unit of depth. After a sequence is
    generated from the output of the various `generators`, each `sequence_transform` is then
    applied in series on the sequence to create a final output sequence. See
    :func:`sample_sequence` for more information.

    .. [Vol] A volumetric framework for quantum computer benchmarks.
        Blume-Kohout and Young.
        arXiv:1904.05546v2 (2019)
        https://arxiv.org/pdf/1904.05546.pdf
    """
    generators: List[Callable] = field(default_factory=lambda : [])
    sequence_transforms: List[Callable] = field(default_factory=lambda : [])
    pattern: List[Union[int, Tuple[List, int], Tuple[List, str]]] = field(init=False, repr=False)

    def __post_init__(self):
        self.pattern  = make_default_pattern(len(self.generators))

    def append(self, other):
        """
        Mutates the CircuitTemplate object by appending new generators. It is ambiguous how to
        append patterns, so we reset the pattern to the default.

        :param other:
        :return:
        """
        if isinstance(other, list):
            self.generators += other
        elif isinstance(other, CircuitTemplate):
            self.generators += other.generators
            self.sequence_transforms += other.sequence_transforms
            # make default pattern since it is unclear how to compose general patterns.
            self.pattern = make_default_pattern(len(self.generators))
        else:
            raise ValueError(f'Cannot append type {type(other)}.')

    def __add__(self, other):
        """
        Concatenate two circuits together, returning a new one.

        :param Circuit other: Another circuit to add to this one.
        :return: A newly concatenated circuit.
        :rtype: Program
        """
        ckt = CircuitTemplate()
        ckt.append(self)
        ckt.append(other)
        return ckt

    def __iadd__(self, other):
        """
        Concatenate two circuits together using +=, returning a new one.
        """
        self.append(other)
        return self

    def sample_sequence(self, graph, repetitions, qc=None, width=None, sequence=None, pattern=None):
        """
        The introduction of `pattern` is an attempt to enable some flexibility in specifying what
        exactly constitutes a single unit of 'depth'. The default behavior is to sample from each
        generator in series and consider these combined samples as a single unit. Thus,
        the default pattern is

            [(list(range(num_generators)), 'n')]

        indicating that we combine samples from the generators at sequential indices and repeat
        this depth many, or 'n' times.

        Another common family this will enable is 'do depth many layers of gates, then invert
        them at the end'. If the last generator is the inversion generator this is specified by the
        pattern

            [(list(range(num_generators - 1)), 'n'), -1]

        In general, a `pattern` is a list whose elements are either

            1) an index of a generator
            2) a tuple of a `pattern` and a number of repetitions
            3) a tuple of a `pattern` and 'n', indicating depth many repetitions

        The sequence_transforms are distinct from generators in that they take in a sequence and
        output a new sequence. These are applied in series after the entire sequence has been
        generated. A family of interest that is not easily generated by generators + patterns
        alone is given by

            C_0 P_0 C_1 P_1 ... P_{N-1} C_N P_N C_N^t P_{N+1} ... C_1^t P_{2N-1} C_0^t

        where C_j is a clifford, P_j is a random local Pauli. We could accomplish this with a
        bespoke 'alternate conjugate and random local pauli layer' that is applied as the last step
        after P_N is added to the sequence and steps through the entire sequence in reverse.
        Instead, we introduce sequence_transforms that Conjugation of a sequence and Pauli frame
        randomization are sequence level operations that can be conceptually distinguished.

        :param graph:
        :param repetitions:
        :param qc:
        :param width:
        :param sequence:
        :param pattern:
        :return:
        """
        if width is not None:
            graph = random.choice(generate_connected_subgraphs(graph, width))

        if pattern is None:
            pattern = self.pattern

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
                    if elem[1] == 'n':
                        # n indicates `repetitions` number of times
                        reps = repetitions
                    elif isinstance(elem[1], int) and elem[1]>=0:
                        reps = elem[1]
                    else:
                        raise ValueError('Repetitions must be specified by int or `n`.')

                    for _ in range(reps):
                        _do_pattern(elem[0])
                else:
                    raise ValueError('Pattern is malformed. A pattern is a list where each element '
                                     'can either be a generator index or a (pattern_i, num) tuple, '
                                     'where num is an integer indicating how many times to '
                                     'repeat the associated pattern_i.')

        _do_pattern(pattern)

        for sequence_transform in self.sequence_transforms:
            sequence = sequence_transform(graph=graph, qc=qc, width=width, sequence=sequence)

        return sequence

    def sample_program(self, graph, repetitions, qc=None, width=None, sequence=None,
                       pattern = None):
        return merge_programs(self.sample_sequence(graph, repetitions, qc, width, sequence, pattern))


# ==================================================================================================
# Generators
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
    # TODO: two coloring with pragmas
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


def dagger_previous(sequence: List[Program], n: int = 1):
    return merge_programs(sequence[-n:]).dagger()


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
    # ignore the edges in the graph
    for q1, q2 in zip(qubits[::2], qubits[1::2]):
        matrix = haar_rand_unitary(4)
        gate_definition = DefGate(f"RSU4_{q1}_{q2}", matrix)
        RSU4 = gate_definition.get_constructor()
        prog += gate_definition
        prog += RSU4(q1, q2)
    return prog


def maxcut_cost_unitary(graph: nx.Graph, layer_number):
    prog = Program()
    theta = prog.declare('theta_' + str(layer_number), memory_type='REAL')
    for edge in graph.edges:
        exponential_map(sZ(edge[0] * sZ(edge[1])))(theta)
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
# Sequence Transforms
###
def dagger_sequence(sequence: List[Program], **kwargs):
    return sequence + [prog.dagger() for prog in reversed(sequence)]


def pauli_frame_randomize_sequence(sequence: List[Program], graph: nx.Graph, **kwargs):
    paulis = [I, X, Y, Z]
    random_paulis = [random_single_qubit_gates(graph, paulis) for _ in range(len(sequence) + 1)]
    new_sequence = [None for _ in range(2*len(sequence) + 1)]
    new_sequence[::2] = random_paulis
    new_sequence[1::2] = sequence
    return new_sequence
###
# Templates
###
def get_rand_1q_template(gates: Sequence[Gate]):
    def func(graph, **kwargs):
        return random_single_qubit_gates(graph, gates=gates)
    return CircuitTemplate([func])


def get_rand_2q_template(gates: Sequence[Gate]):
    def func(graph, **kwargs):
        return random_two_qubit_gates(graph, gates=gates)
    return CircuitTemplate([func])


def get_rand_1q_cliff_template(bm: BenchmarkConnection):
    def func(graph, **kwargs):
        return random_single_qubit_cliffords(bm, graph)
    return CircuitTemplate([func])


def get_rand_2q_cliff_template(bm: BenchmarkConnection):
    def func(graph, **kwargs):
        return random_two_qubit_cliffords(bm, graph)
    return CircuitTemplate([func])


def get_dagger_all_template():
    def func(qc, sequence, **kwargs):
        prog = dagger_previous(sequence, len(sequence))
        native_quil = qc.compiler.quil_to_native_quil(prog)
        # remove gate definition and HALT
        return Program([instr for instr in native_quil.instructions][:-1])
    return CircuitTemplate([func])


def get_dagger_previous(n: int = 1):
    def func(qc, sequence, **kwargs):
        prog = dagger_previous(sequence, n)
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


def get_switch_basis_x_z_template():
    def func(graph, **kwargs):
        prog = Program()
        for node in graph.nodes:
            prog.inst(H(node))
        return prog
    return CircuitTemplate([func])


def get_all_H_template():
    return get_switch_basis_x_z_template()


def get_param_local_RX_template():
    # remember that RX(theta) = e^(i theta X/2)
    def func(graph, sequence, **kwargs):
        prog = Program()
        theta = prog.declare('theta_' + str(len(sequence)), memory_type='REAL')
        for node in graph.nodes:
            prog += H(node)
            prog += RZ(theta, node)
            prog += H(node)
        return prog
    return CircuitTemplate([func])


def get_param_maxcut_graph_cost_template(graph_family: Callable[[int], nx.Graph] = None):
    if graph_family is None:
        def default_func(graph, qc, sequence, **kwargs):
            prog = maxcut_cost_unitary(graph, len(sequence))
            native_quil = qc.compiler.quil_to_native_quil(prog)
            # remove gate definition and HALT
            return Program([instr for instr in native_quil.instructions][:-1])
        return CircuitTemplate([default_func])
    else:
        def func(graph, qc, sequence, **kwargs):
            maxcut_graph = graph_family(len(graph.nodes))
            if len(maxcut_graph.nodes) > len(graph.nodes):
                raise ValueError("The maxcut graph must have fewer nodes than the number of "
                                 "qubits.")
            prog = maxcut_cost_unitary(maxcut_graph, len(sequence))
            native_quil = graph_restricted_compilation(qc, graph, prog)
            # remove gate definitions and HALT
            return Program([instr for instr in native_quil.instructions][:-1])
        return CircuitTemplate([func])

# ==================================================================================================
# Data acquisition
# ==================================================================================================
def generate_volumetric_program_array(qc: QuantumComputer, ckt: CircuitTemplate, widths: List[int],
                                      depths: List[int], num_circuit_samples: int,
                                      graph: nx.Graph = None, pattern = None):
    if graph is None:
        graph = qc.qubit_topology()

    programs = {width: {depth: [] for depth in depths} for width in widths}

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

    results = {width: {depth: [] for depth in depth_array.keys()}
               for width, depth_array in program_array.items()}

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

# TODO:
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
def get_error_hamming_weight_distributions(noisy_results, ideal_results):

    # allow for ideal result to depend only on width (pass in a list)
    if not isinstance(ideal_results, dict):
        ideal_results = {width: {depth: ideal_results[width] for depth in depth_array.keys()}
              for width, depth_array in noisy_results.items()}

    distrs = {width: {depth: [] for depth in depth_array.keys()}
              for width, depth_array in noisy_results.items()}

    for width, depth_array in distrs.items():
        for depth, samples in depth_array.items():

            noisy_ckt_sample_results = noisy_results[width][depth]
            ideal_ckt_sample_results = ideal_results[width][depth]

            for noisy_shots, ideal_result in zip(noisy_ckt_sample_results,
                                                 ideal_ckt_sample_results):
                if len(ideal_result) > 1:
                    raise ValueError("You have provided ideal results with more than one shot; "
                                     "this method is intended to analyze results where the ideal "
                                     "result is deterministic, which makes multiple shots "
                                     "unnecessary.")

                hamm_dist_per_shot = [hamming_distance(ideal_result, shot) for shot in
                                      noisy_shots]

                # Hamming weight distribution
                hamm_wt_distr =  get_hamming_wt_distr_from_list(hamm_dist_per_shot, width)
                samples.append(np.asarray(hamm_wt_distr))
    return distrs


def get_single_target_success_probabilities(noisy_results, ideal_results,
                              allowed_errors: Union[int, Callable[[int], int]] = 0):
    if isinstance(allowed_errors, int):
        error_func = lambda num_bits: allowed_errors
    else:
        error_func = allowed_errors

    hamming_distrs = get_error_hamming_weight_distributions(noisy_results, ideal_results)

    return {w: {d: [sum(distr[0:error_func(w)+1]) for distr in distrs]
                for d, distrs in d_distrs.items()} for w, d_distrs in hamming_distrs.items()}


def determine_successes_from_ckt_success_probs(ckt_success_probs,
                                               threshold_probability: float = 2/3):
    return {w: {d: prob > threshold_probability for d, prob in d_ckt_succ_probs.items()}
            for w, d_ckt_succ_probs in ckt_success_probs.items()}


def average_distributions(distrs):
    """
    E.g. take in output of :func:`get_error_hamming_weight_distributions` or
    :func:`get_single_target_success_probabilities`
    :param distrs:
    :return:
    """
    return {w: {d: sum([np.asarray(distr) for distr in distr_list]) / len(distr_list)
                for d, distr_list in d_arr.items()} for w, d_arr in distrs.items()}


def get_total_variation_dist(distr1, distr2):
    return tvd(np.asarray([distr1]).T, np.asarray([distr2]).T)


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


def get_random_hamming_wt_distr(num_bits: int):
    """
    Return the distribution of Hamming weight for randomly drawn bitstrings of length num_bits.

    This is equivalent to the error distribution, e.g. from
    :func:`get_error_hamming_weight_distributions` where the `noisy_results` are entirely random.
    Comparing real data against this distribution may be a useful benchmark in determining
    whether the real data contains any actual information.

    :param num_bits: number of bits in string
    returns: list of hamming weights
    """
    # comb(N, k) = N choose k
    return [comb(num_bits, num_ones) / (2 ** num_bits) for num_ones in range(0, num_bits + 1)]


def plot_error_distributions(distr_arr: Dict[int, Dict[int, Sequence[float]]], widths=None,
                             depths=None, plot_rand_distr=False):
    """
    For each width and depth plot the distribution of errors provided in distr_arr.

    :param distr_arr:
    :param widths:
    :param depths:
    :param plot_rand_distr:
    :return:
    """
    if widths is None:
        widths = list(distr_arr.keys())

    if depths is None:
        depths = list(list(distr_arr.values())[0].keys())

    legend = ['data']
    if plot_rand_distr:
        legend.append('random')

    fig = plt.figure(figsize=(18, 6 * len(depths)))
    axs = fig.subplots(len(depths), len(widths), sharex='col', sharey=True)

    for w_idx, w in enumerate(widths):
        x_labels = np.arange(0, w + 1)
        depth_distrs = distr_arr[w]

        if plot_rand_distr:
            rand_distr = get_random_hamming_wt_distr(w)

        for d_idx, d in enumerate(depths):
            distr = depth_distrs[d]

            idx = d_idx * len(widths) + w_idx
            if len(widths) == len(depths) == 1:
                ax = axs
            else:
                ax = axs.flatten()[idx]
            ax.bar(x_labels, distr, width=0.61, align='center')

            if plot_rand_distr:
                ax.bar(x_labels, rand_distr, width=0.31, align='center')

            ax.set_xticks(x_labels)
            ax.grid(axis='y', alpha=0.75)
            ax.set_title(f'w = {w}, d = {d}', size=20)

            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(15)

            for tick in ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(15)

    fig.legend(legend, loc='right', fontsize=15)
    plt.ylim(0, 1)
    fig.text(0.5, 0.05, 'Hamming Weight of Error', ha='center', va='center', fontsize=20)
    fig.text(0.06, 0.5, 'Relative Frequency of Occurrence', ha='center', va='center',
             rotation='vertical', fontsize=20)
    plt.subplots_adjust(wspace=0, hspace=.15, left=.1)

    return fig, axs


def plot_success(successes, title, widths=None, depths=None, boxsize=1500):
    """
    Plot the given successes at each width and depth.

    If a given (width, depth) is not recorded in successes then nothing is plotted for that
    point. Successes are displayed as filled boxes while failures are simply box outlines.

    :param successes:
    :param title:
    :param widths:
    :param depths:
    :param boxsize:
    :return:
    """
    if widths is None:
        widths = list(successes.keys())

    if depths is None:
        depths = list(set(d for w in successes.keys() for d in successes[w].keys()))

    fig_width = min(len(widths), 15)
    fig_depth = min(len(depths), 15)

    fig, ax = plt.subplots(figsize=(fig_width, fig_depth))

    margin = .5
    ax.set_xlim(-margin, len(widths) + margin - 1)
    ax.set_ylim(-margin, len(depths) + margin - 1)
    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels(widths)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels(depths)
    ax.set_xlabel('Width')
    ax.set_ylabel('Depth')

    colors = ['white', 'lightblue']

    for w_idx, w in enumerate(widths):
        if w not in successes.keys():
            continue
        depth_succ = successes[w]
        for d_idx, d in enumerate(depths):
            if d not in depth_succ.keys():
                continue
            color = colors[0]
            if depth_succ[d]:
                color = colors[1]
            ax.scatter(w_idx, d_idx, marker='s', s=boxsize, color=color,
                       edgecolors='black')

    # legend
    labels = ['Fail', 'Pass']
    for color, label in zip(colors, labels):
        plt.scatter([], [], marker='s', c=color, label=label, edgecolors='black')
    ax.legend()

    ax.set_title(title)

    return fig, ax


def plot_pareto_frontier(successes, title, widths=None, depths=None):
    """
    Given the successes at measured widths and depths, draw the frontier that separates success
    from failure.

    Specifically, the frontier is drawn as follows::

        For a given width, draw a line separating all low-depth successes from the minimum
        depth failure. For each depth smaller than the minimum failure depth, draw a line
        separating the neighboring (width +/- 1, depth) cell if depth is less than the
        minimum depth failure for that neighboring width.

    If a requested (width, depth) cell is not specified in successes then no lines will be drawn
    around that cell.

    :param successes:
    :param title:
    :param widths:
    :param depths:
    :return:
    """
    if widths is None:
        widths = list(successes.keys())

    if depths is None:
        depths = list(set(d for w in successes.keys() for d in successes[w].keys()))

    fig_width = min(len(widths), 15)
    fig_depth = min(len(depths), 15)

    fig, ax = plt.subplots(figsize=(fig_width, fig_depth))

    margin = .5
    ax.set_xlim(-margin, len(widths) + margin - 1)
    ax.set_ylim(-margin, len(depths) + margin - 1)
    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels(widths)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels(depths)
    ax.set_xlabel('Width')
    ax.set_ylabel('Depth')

    min_depth_failure_at_width = []
    for w_idx, w in enumerate(widths):
        if w not in successes.keys():
            min_depth_failure_at_width.append(None)
            continue

        depth_succ = successes[w]
        min_depth_failure = len(depths)
        for d_idx, d in enumerate(depths):
            if d not in depth_succ.keys():
                continue
            if not depth_succ[d]:
                min_depth_failure = d_idx
                break
        min_depth_failure_at_width.append(min_depth_failure)

    for idx, depth in enumerate(min_depth_failure_at_width):
        if depth is None:
            continue  # the depth was not determined, so leave this boundary open

        # horizontal line for this width
        if depth < len(depths):
            ax.plot((idx - margin, idx + margin), (depth - margin, depth - margin), color='black')

        # vertical lines
        if idx < len(min_depth_failure_at_width) - 1:
            for d_idx in range(len(depths)):
                if depths[d_idx] not in [d for d in successes[widths[idx]].keys()]:
                    continue  # do not plot line if this depth was not measured
                if depth > d_idx >= min_depth_failure_at_width[idx + 1]:
                    ax.plot((idx + margin, idx + margin), (d_idx - margin, d_idx + margin),
                            color='black')

    ax.set_title(title)
    return fig, ax


def basement_log_function(number: float):
    return basement_function(np.log2(number))


def basement_function(number: float):
    """
    Return the floor of the number, or 0 if the number is negative.

    :param number: the basement function is applied to this number.
    :returns: basement of the number
    """
    return max(int(np.floor(number)), 0)


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
