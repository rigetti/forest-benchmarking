from typing import Callable, Dict, List, Union
import networkx as nx
import numpy as np
import random
import itertools
from scipy.spatial.distance import hamming
from scipy.special import comb
from dataclasses import dataclass, field

from pyquil.quil import Program, address_qubits, merge_programs
from pyquil.api import QuantumComputer
from pyquil.gates import MEASURE, RESET

from forest.benchmarking.distance_measures import total_variation_distance as tvd


@dataclass
class CircuitTemplate:
    """
    This dataclass enables us to specify various families of circuits and sample from a specified
    family random circuits of various width and depth acting on different groups of qubits.

    'Width' is simply the number of qubits measured at then end of the circuit. 'Depth' is not
    simply circuit depth, but rather the number of repeated structured groups of gates,
    each of which constitutes some distinct unit. A depth d circuit could  consist of d
    consecutive rounds of random single qubit, then two qubit gates. It could also mean d
    consecutive random Cliffords followed by the d conjugated Cliffords that invert the first d
    gates.

    Because these families of circuits are quite diverse, specifying the family and drawing
    samples can potentially require a wide variety of parameters. The compiler may be required to
    map an abstract circuit into native quil; a sample acting on a specific qubit topology
    may be desired; the sequence of 'layers' generated so far may be necessary to compute an
    inverse.

    We represent each sampled circuit as a list of PyQuil Programs, which we call a 'sequence'
    since each element of the list holds a distinctly structured group of gates that,
    when applied altogether in series, constitute the circuit. This core functionality is found in
    :func:`sample_sequence`. In this function `generators` are applied in series in a loop
    `repetitions` number of times. Each call to a generator will contribute an element to the
    output sequence (some combination of which will constitute a unit of depth). After a
    sequence is generated from the output of the various `generators`, each `sequence_transform`
    is then applied in series on the generated sequence to create a final output sequence. The
    sequence transforms account for any features of the circuit that do increase with depth,
    cannot neatly be fit into repeated units, or otherwise require performing a global
    transformation on the sequence. See :func:`sample_sequence` for more information.

    This functionality is intended to enable creation and use of any of a wide variety of
    'volumetric benchmarks' described in the sources below.

    .. [Vol] A volumetric framework for quantum computer benchmarks.
        Blume-Kohout and Young.
        arXiv:1904.05546v2 (2019)
        https://arxiv.org/pdf/1904.05546.pdf

    .. [QVol] Validating quantum computers using randomized model circuits.
        Cross et al.
        arXiv:1811.12926v1  (2018).
        https://arxiv.org/abs/1811.12926
    """
    generators: List[Callable] = field(default_factory=lambda: [])
    sequence_transforms: List[Callable] = field(default_factory=lambda: [])

    def append(self, other):
        """
        Mutates the CircuitTemplate object by appending new generators.
        TODO: The behavior of sequence_transforms may not conform with expectations.
        """
        if isinstance(other, list):
            self.generators += other
        elif isinstance(other, CircuitTemplate):
            self.generators += other.generators
            self.sequence_transforms += other.sequence_transforms
        else:
            raise ValueError(f'Cannot append type {type(other)}.')

    def __add__(self, other):
        """
        Concatenate two circuits together, returning a new one.
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

    def sample_sequence(self, graph: nx.Graph, repetitions: int, qc: QuantumComputer = None,
                        width: int = None, sequence: List[Program] = None):
        """
        The sequence_transforms are distinct from generators in that they take in a sequence and
        output a new sequence. These are applied in series after the entire sequence has been
        generated. A family of interest that motivates this distinction is

            C_0 P_0 C_1 P_1 ... P_{N-1} C_N P_N C_N^t P_{N+1} ... C_1^t P_{2N-1} C_0^t

        where C_j is a clifford, P_j is a random local Pauli. We can specify this family by a
        generator of random Cliffords, a conjugation sequence transform, and a Pauli frame
        randomization transform.

        :param graph: the qubit topology on which the circuit should act. Unless width is
            specified, the number of qubits in the graph should be considered circuit width.
        :param repetitions: the number of times the loop of generators should be applied.
        :param qc: a quantum computer, likely the one on which the circuit will be run, providing
            access to the full chip topology and associated compiler.
        :param width: the number of qubits that will be measured at the end of the circuit. If
            the supplied graph contains more qubits, an induced subgraph of width-many qubits
            will be selected uniformly at random from the graph.
        :param sequence: an optional initialization of a sequence to build off of/append to.
        :return: the list of programs whose sum constitutes a circuit sample from the family of
            circuits specified by the generators and sequence_transforms.
        """
        if width is not None:
            graph = random.choice(generate_connected_subgraphs(graph, width))

        if sequence is None:
            sequence = []

        # run through the generators 'repetitions' many times; append each generated program to
        # the sequence.
        for _ in range(repetitions):
            for generator in self.generators:
                sequence.append(generator(graph=graph, qc=qc, width=width, sequence=sequence))

        for sequence_transform in self.sequence_transforms:
            sequence = sequence_transform(graph=graph, qc=qc, width=width, sequence=sequence)

        return sequence

    def sample_program(self, graph, repetitions, qc=None, width=None, sequence=None):
        return merge_programs(self.sample_sequence(graph, repetitions, qc, width, sequence))


def generate_volumetric_program_array(qc: QuantumComputer, ckt: CircuitTemplate,
                                      dimensions: Dict[int, List[int]], num_circuit_samples: int,
                                      graphs: Dict[int, List[nx.Graph]] = None) \
        -> Dict[int, Dict[int, List[Program]]]:
    """
    Creates a dictionary containing random circuits sampled from the input `ckt` family for each
    width and depth.

    :param qc:
    :param ckt:
    :param dimensions
    :param num_circuit_samples:
    :param graphs:
    :return:
    """
    if graphs is None:
        graphs = {w: sample_random_connected_graphs(qc.qubit_topology(), w,
                                                    len(depths) * num_circuit_samples)
                  for w, depths in dimensions.items()}

    programs = {width: {depth: [] for depth in depths} for width, depths in dimensions.items()}

    for width, depth_array in programs.items():
        circuit_number = 0
        for depth, prog_list in depth_array.items():
            for _ in range(num_circuit_samples):
                graph = graphs[width][circuit_number]
                circuit_number += 1
                prog = ckt.sample_program(graph, repetitions=depth, width=width, qc=qc)
                prog_list.append(prog)

    return programs


def sample_random_connected_graphs(graph: nx.Graph, width: int, num_ckts: int):
    """
    Helper to uniformly randomly sample `num_ckts` many connected induced subgraphs of
    `graph` of `width` many qubits.

    :param graph:
    :param width:
    :param num_ckts:
    :return:
    """
    connected_subgraphs = generate_connected_subgraphs(graph, width)
    random_indices = np.random.choice(range(len(connected_subgraphs)), size=num_ckts)
    return [connected_subgraphs[idx] for idx in random_indices]


def generate_connected_subgraphs(graph: nx.Graph, n_vert: int):
    """
    Given a lattice on the QPU or QVM, specified by a networkx graph, return a list of all
    subgraphs with n_vert connect vertices.

    :params n_vert: number of vertices of connected subgraph.
    :params graph: networkx graph
    :returns: list of subgraphs with n_vert connected vertices
    """
    subgraph_list = []
    for sub_nodes in itertools.combinations(graph.nodes(), n_vert):
        subg = graph.subgraph(sub_nodes)
        if nx.is_connected(subg):
            subgraph_list.append(subg)
    return subgraph_list


def acquire_volumetric_data(qc: QuantumComputer, program_array: Dict[int, Dict[int, List[Program]]],
                            num_shots: int = 500,
                            measure_qubits: Dict[int, Dict[int, List[int]]] = None,
                            use_active_reset: bool = False, use_compiler: bool = False) \
        -> Dict[int, Dict[int, List[np.ndarray]]]:
    """
    Runs each program in `program_array` on the qc and stores the results, organized again by
    width and depth.

    :param qc:
    :param program_array:
    :param num_shots:
    :param measure_qubits:
    :param use_active_reset:
    :param use_compiler:
    :return:
    """
    reset_prog = Program()
    if use_active_reset:
        reset_prog += RESET()

    results = {width: {depth: [] for depth in depth_array.keys()}
               for width, depth_array in program_array.items()}

    for width, depth_array in program_array.items():
        for depth, prog_list in depth_array.items():
            for idx, program in enumerate(prog_list):
                prog = program.copy()

                if measure_qubits is not None:
                    qubits = measure_qubits[width][depth][idx]
                else:
                    qubits = sorted(list(program.get_qubits()))

                ro = prog.declare('ro', 'BIT', len(qubits))
                for ro_idx, q in enumerate(qubits):
                    prog += MEASURE(q, ro[ro_idx])

                prog.wrap_in_numshots_loop(num_shots)

                if use_compiler:
                    prog = qc.compiler.quil_to_native_quil(prog)

                exe = qc.compiler.native_quil_to_executable(prog)
                shots = qc.run(exe)
                results[width][depth].append(shots)

    return results


def get_error_hamming_weight_distributions(noisy_results: Dict[int, Dict[int, List[np.ndarray]]],
                                           ideal_results: Dict[int, Dict[int, List[np.ndarray]]]):
    """
    Calculate the hamming distance to the ideal for each noisy shot of each circuit sampled for
    each width and depth.

    Note that this method is only appropriate when the ideal result for each circuit is a single
    deterministic (circuit-dependent) output; therefore, ideal_results should only contain one
    shot per circuit.

    :param noisy_results:
    :param ideal_results:
    :return:
    """
    distrs = {width: {depth: [] for depth in depth_array.keys()}
              for width, depth_array in noisy_results.items()}

    for width, depth_array in distrs.items():
        for depth, samples in depth_array.items():

            noisy_ckt_sample_results = noisy_results[width][depth]
            ideal_ckt_sample_results = ideal_results[width][depth]

            # iterate over circuits
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
                hamm_wt_distr = get_hamming_wt_distr_from_list(hamm_dist_per_shot, width)
                samples.append(np.asarray(hamm_wt_distr))
    return distrs


def get_single_target_success_probabilities(noisy_results, ideal_results,
                                            allowed_errors: Union[int, Callable[[int], int]] = 0):
    """
    For circuit results of various width and depth, calculate the fraction of noisy results
    that match the single ideal result for each circuit.

    Note that this method is only appropriate when the ideal result for each circuit is a single
    deterministic (circuit-dependent) output.

    :param noisy_results: noisy shots from each circuit sampled for each width and depth
    :param ideal_results: a single ideal result for each circuit
    :param allowed_errors: either a number indicating the maximum hamming distance from the ideal
        result is still considered a success, or a function which returns the max hamming
        distance allowed for a given width.
    :return:
    """
    if isinstance(allowed_errors, int):
        def error_func(num_bits):
            return allowed_errors
    else:
        error_func = allowed_errors

    hamming_distrs = get_error_hamming_weight_distributions(noisy_results, ideal_results)

    return {w: {d: [sum(distr[0:error_func(w) + 1]) for distr in distrs]
                for d, distrs in d_distrs.items()}
            for w, d_distrs in hamming_distrs.items()}


def average_distributions(distrs):
    """
    E.g. take in output of :func:`get_error_hamming_weight_distributions` or
    :func:`get_single_target_success_probabilities`

    :param distrs:
    :return:
    """
    return {w: {d: sum([np.asarray(distr) for distr in distr_list]) / len(distr_list)
                for d, distr_list in d_arr.items()}
            for w, d_arr in distrs.items()}


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


def basement_log_function(number: float):
    return basement_function(np.log2(number))


def basement_function(number: float):
    """
    Return the floor of the number, or 0 if the number is negative.

    :param number: the basement function is applied to this number.
    :returns: basement of the number
    """
    return max(int(np.floor(number)), 0)
