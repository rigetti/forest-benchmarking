from typing import Tuple, Sequence, Callable, Dict, List, Union, Optional
from copy import copy
import networkx as nx
import numpy as np
import random
import itertools
from scipy.spatial.distance import hamming
from scipy.special import comb
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from statistics import median

from pyquil.quilbase import Pragma, Gate, DefGate, DefPermutationGate
from pyquil.quilatom import QubitPlaceholder
from pyquil.quil import Program, address_qubits, merge_programs
from pyquil.api import QuantumComputer, BenchmarkConnection
from pyquil.gates import *
from pyquil.paulis import exponential_map, sX, sZ
from pyquil.numpy_simulator import NumpyWavefunctionSimulator
from rpcq.messages import TargetDevice
from rpcq._utils import RPCErrorError

from forest.benchmarking.randomized_benchmarking import get_rb_gateset
from forest.benchmarking.distance_measures import total_variation_distance as tvd
from forest.benchmarking.operator_tools.random_operators import haar_rand_unitary
from forest.benchmarking.utils import bit_array_to_int
from forest.benchmarking.compilation import basic_compile


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
    generators: List[Callable] = field(default_factory=lambda : [])
    sequence_transforms: List[Callable] = field(default_factory=lambda : [])

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


def graph_restricted_compilation(qc: QuantumComputer, graph: nx.Graph,
                                 program: Program) -> Program:
    """
    A useful helper that temporarily modifies the supplied qc's qubit topology to match the
    supplied graph so that the given program may be compiled onto the graph topology.

    :param qc: a qc object with a compiler where the given graph is a subraph of the qc's qubit
        topology.
    :param graph: The desired subraph of the qc's full topology on which we wish to run a program.
    :param program: a program we wish to run on a particular graph on the qc.
    :return: the program compiled into native quil gates respecting the graph topology.
    """
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
        if (int(q1), int(q2)) in graph.edges:
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


# ==================================================================================================
# Generators
# ==================================================================================================
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


def random_two_qubit_gates(graph: nx.Graph, gates:  Sequence[Gate]) -> Program:
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


def maxcut_cost_unitary(graph: nx.Graph, idx_label: int) -> Program:
    """
    Creates a parameterized program used in QAOA that enacts commuting parameterized 2 qubit
    gates on every edge of the graph.

    :param graph:
    :param idx_label: a label that uniquely identifies the set of gate definitions used in the
        output program. This prevents subsequent calls to this method from producing a program
        with definitions that overwrite definitions in previously generated programs.
    :return:
    """
    prog = Program()
    gamma = prog.declare('gamma_' + str(idx_label), memory_type='REAL')
    for edge in graph.edges:
        prog += exponential_map(sZ(edge[0]) * sZ(edge[1]))(gamma)
    return prog


###
# Sequence Transforms
###
def hadamard_sandwich(sequence: List[Program], graph: nx.Graph, **kwargs) -> List[Program]:
    """
    Insert a Hadamard gate on each qubit at the beginning and end of the sequence.

    This can be viewed as switching from the computational Z basis to the X basis.

    :param sequence: the sequence to be sandwiched by Hadamards
    :param graph: the graph containing the qubits to be acted on by Hadamards
    :param kwargs: extraneous arguments
    :return: a new sequence which is the input sequence with new starting and ending layers of
        Hadamards.
    """
    prog = Program()
    for node in graph.nodes:
        prog.inst(H(node))
    return [prog] + sequence + [prog.copy()]


def dagger_sequence(sequence: List[Program], **kwargs):
    """
    Returns the original sequence with its layer-by-layer inverse appended on the end.

    The net result of the output sequence is the Identity.

    .. CAUTION::
          Merging this sequence and compiling the resulting program will result in a trivial
          empty program. To avoid this, consider using a sequence transform to compile each
          element of the sequence first, then combine the result. For example, see
          :func:`compile_individual_sequence_elements`. Using :func:`compile_merged_sequence`
          with `use_basic_compile` set to True will also avoid this issue, but will not compile
          gate definitions and will not compile gates onto the chip topology.

    :param sequence: the sequence of programs comprising a circuit that will be inverted and
        appended to the sequence.
    :param kwargs: extraneous arguments
    :return: a new sequence the input sequence and its inverse
    """
    return sequence + [prog.dagger() for prog in reversed(sequence)]


def pauli_frame_randomize_sequence(sequence: List[Program], graph: nx.Graph, **kwargs) \
        -> List[Program]:
    """
    Inserts random single qubit Pauli gates on each qubit in between elements of the input sequence.

    :param sequence:
    :param graph: a graph containing qubits that will be randomly paired together. Note that
        the graph topology (the edges) are ignored.
    :param kwargs: extraneous arguments
    :return:
    """
    paulis = [I, X, Y, Z]
    random_paulis = [random_single_qubit_gates(graph, paulis) for _ in range(len(sequence) + 1)]
    new_sequence = [None for _ in range(2*len(sequence) + 1)]
    new_sequence[::2] = random_paulis
    new_sequence[1::2] = sequence
    return new_sequence


def compile_individual_sequence_elements(qc: QuantumComputer, sequence: List[Program],
                                         graph: nx.Graph, **kwargs) -> List[Program]:
    """
    Returns the sequence where each element is individually compiled into native quil in a way
    that respects the given graph topology.

    :param qc:
    :param sequence:
    :param graph:
    :param kwargs: extraneous arguments
    :return:
    """
    compiled_sequence = []
    for prog in sequence:
        native_quil = graph_restricted_compilation(qc, graph, prog)
        # remove gate definitions and HALT
        compiled_sequence.append(Program([instr for instr in native_quil.instructions][:-1]))
    return compiled_sequence


def compile_merged_sequence(qc: QuantumComputer, sequence: List[Program], graph: nx.Graph,
                            use_basic_compile: bool = False, **kwargs) -> List[Program]:
    """
    Merges the sequence into a Program and returns a 'sequence' comprised of the corresponding
    compiled native quil program that respects the given graph topology.

    .. CAUTION::
        The option to only use basic_compile will only result in native quil if the merged
        sequence contains no gate definitions and if all multi-qubit gates already respect
        the graph topology. If this is not the case, the output program may not be able to be
        converted properly to an executable that can be run on the qc.

    :param qc:
    :param sequence:
    :param graph:
    :param use_basic_compile:
    :param kwargs: extraneous arguments
    :return:
    """
    merged = merge_programs(sequence)
    if use_basic_compile:
        return [basic_compile(merged)]
    else:
        native_quil = graph_restricted_compilation(qc, graph, merged)
        # remove gate definitions and terminous HALT
        return [Program([instr for instr in native_quil.instructions][:-1])]


###
# Templates
###
def get_rand_1q_template(gates: Sequence[Gate]):
    """
    Creates a CircuitTemplate representing the family of circuits generated by repeated layers of
    random single qubit gates pulled from the input set of gates.

    :param gates:
    :return:
    """
    def func(graph, **kwargs):
        return random_single_qubit_gates(graph, gates=gates)
    return CircuitTemplate([func])


def get_rand_2q_template(gates: Sequence[Gate]):
    """
    Creates a CircuitTemplate representing the family of circuits generated by repeated layers of
    random two qubit gates pulled from the input set of gates.

    :param gates:
    :return:
    """
    def func(graph, **kwargs):
        return random_two_qubit_gates(graph, gates=gates)
    return CircuitTemplate([func])


def get_rand_1q_cliff_template(bm: BenchmarkConnection):
    """
    Creates a CircuitTemplate representing the family of circuits generated by repeated layers of
    random single qubit Clifford gates.
    """
    def func(graph, **kwargs):
        return random_single_qubit_cliffords(bm, graph)
    return CircuitTemplate([func])


def get_rand_2q_cliff_template(bm: BenchmarkConnection):
    """
    Creates a CircuitTemplate representing the family of circuits generated by repeated layers of
    random two qubit Clifford gates.
    """
    def func(graph, **kwargs):
        return random_two_qubit_cliffords(bm, graph)
    return CircuitTemplate([func])


def get_dagger_previous(n: int = 1):
    """
    Creates a CircuitTemplate that can be appended to another template to generate families of
    circuits with repeated (layer, inverse-layer) units.
    """
    def func(sequence, **kwargs):
        return dagger_previous(sequence, n)
    return CircuitTemplate([func])


def get_rand_su4_template():
    """
    Creates a CircuitTemplate representing the family of circuits generated by repeated layers of
    Haar-random two qubit gates acting on random pairs of qubits. This is the generator used in
    quantum volume [QVol]_ .
    """
    def func(graph, sequence, **kwargs):
        return random_su4_pairs(graph, len(sequence))
    return CircuitTemplate([func])


def get_quantum_volume_template():
    """
    Creates a quantum volume CircuitTemplate. See [QVol]_ .
    """
    template = get_rand_su4_template()
    template.sequence_transforms.append(compile_merged_sequence)
    return template


def get_param_local_RX_template():
    # remember that RX(theta) = e^(i theta X/2)
    def func(graph, sequence, **kwargs):
        prog = Program()
        theta = prog.declare('beta_' + str(len(sequence)), memory_type='REAL')
        for node in graph.nodes:
            prog += H(node)
            prog += RZ(theta, node)
            prog += H(node)
        return prog
    return CircuitTemplate([func])


def get_param_maxcut_graph_cost_template(graph_family: Callable[[int], nx.Graph] = None):
    if graph_family is None:
        def default_func(graph, qc, sequence, **kwargs):
            return maxcut_cost_unitary(graph, len(sequence))
        return CircuitTemplate([default_func])
    else:
        def func(graph, qc, sequence, **kwargs):
            maxcut_graph = graph_family(len(graph.nodes))
            if len(maxcut_graph.nodes) > len(graph.nodes):
                raise ValueError("The maxcut graph must have fewer nodes than the number of "
                                 "qubits.")
            return maxcut_cost_unitary(maxcut_graph, len(sequence))
        return CircuitTemplate([func])


def get_maxcut_qaoa_template(graph_family: Callable[[int], nx.Graph] = None):
    cost_layer = get_param_maxcut_graph_cost_template(graph_family)
    rotation_layer = get_param_local_RX_template()
    qaoa_template = cost_layer + rotation_layer

    def initialize(sequence: List[Program], graph: nx.Graph, **kwargs) -> List[Program]:
        """
        Insert a Hadamard gate on each qubit at the beginning of the sequence.
        """
        prog = Program()
        for node in graph.nodes:
            prog.inst(H(node))
        return [prog] + sequence

    qaoa_template.sequence_transforms.append(initialize)
    qaoa_template.sequence_transforms.append(compile_merged_sequence)
    return qaoa_template


# ==================================================================================================
# Data acquisition
# ==================================================================================================
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
                                                    len(depths)*num_circuit_samples)
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


def acquire_volumetric_data(qc: QuantumComputer, program_array: Dict[int, Dict[int, List[Program]]],
                            num_shots: int = 500,
                            measure_qubits: Dict[int,  Dict[int, List[int]]] = None,
                            use_active_reset:  bool = False, use_compiler: bool = False) \
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
                for idx, q in enumerate(qubits):
                    prog += MEASURE(q, ro[idx])

                prog.wrap_in_numshots_loop(num_shots)

                if use_compiler:
                    prog = qc.compiler.quil_to_native_quil(prog)

                exe = qc.compiler.native_quil_to_executable(prog)
                shots = qc.run(exe)
                results[width][depth].append(shots)

    return results


def collect_heavy_outputs(wfn_sim: NumpyWavefunctionSimulator,
                          program_array: Dict[int, Dict[int, List[Program]]],
                          measure_qubits: Optional[Dict[int,  Dict[int, List[int]]]] = None) \
        -> Dict[int, Dict[int, List[List[int]]]]:
    """
    Collects and returns those 'heavy' bitstrings which are output with greater than median
    probability among all possible bitstrings on the given qubits.

    The method uses the provided wfn_sim to calculate the probability of measuring each bitstring
    from the output of the circuit comprised of the given permutations and gates.

    :param wfn_sim: a NumpyWavefunctionSimulator that can simulate the provided program
    :param program_array: a collection of PyQuil Programs sampled from the circuit family for
        each (width, depth) pair.
    :param measure_qubits: optional list of qubits to measure for each Program in
        `program_array`. By default all qubits in the Program are measured
    :return: a list of the heavy outputs of the circuit, represented as ints
    """
    heavy_output_array = {w: {d: [] for d in d_arr.keys()} for w, d_arr in program_array.items()}

    for w, d_progs in program_array.items():
        for d, ckts in d_progs.items():
            for idx, ckt in enumerate(ckts):
                wfn_sim.reset()
                for gate in ckt:
                    wfn_sim.do_gate(gate)

                if measure_qubits is not None:
                    qubits = measure_qubits[w][d][idx]
                else:
                    qubits = sorted(list(ckt.get_qubits()))

                # Note that probabilities are ordered lexicographically with qubit 0 leftmost.
                # we need to restrict attention to the subset `qubits`
                probs = abs(wfn_sim.wf)**2
                probs = probs.reshape([2] * wfn_sim.n_qubits)
                marginal = probs
                for q in reversed(range(wfn_sim.n_qubits)):
                    if q in qubits:
                        continue
                    marginal = np.sum(marginal, axis=q)

                probabilities = marginal.reshape(-1)

                median_prob = median(probabilities)

                # store the integer indices, which implicitly represent the bitstring outcome.
                heavy_outputs = [idx for idx, prob in enumerate(probabilities) if prob > median_prob]
                heavy_output_array[w][d].append(heavy_outputs)

    return heavy_output_array


# ==================================================================================================
# Analysis
# ==================================================================================================
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
                hamm_wt_distr =  get_hamming_wt_distr_from_list(hamm_dist_per_shot, width)
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
        error_func = lambda num_bits: allowed_errors
    else:
        error_func = allowed_errors

    hamming_distrs = get_error_hamming_weight_distributions(noisy_results, ideal_results)

    return {w: {d: [sum(distr[0:error_func(w)+1]) for distr in distrs]
                 for d, distrs in d_distrs.items()}
            for w, d_distrs in hamming_distrs.items()}


def get_success_probabilities(noisy_results, ideal_results):
    """
    For circuit results of various width and depth, calculate the fraction of noisy results
    that are also found in the collection of ideal results for each circuit.

    Quantum volume employs this method to calculate success_probabilities where the ideal_results
    are the heavy hitters of each circuit.

    :param noisy_results: noisy shots from each circuit sampled for each width and depth
    :param ideal_results: a collection of ideal results for each circuit; membership of a noisy
        shot from a particular circuit in the corresponding set of ideal_results constitutes a
        success.
    :return: the estimated success probability for each circuit.
    """
    prob_success = {width: {depth: [] for depth in depth_array.keys()}
                    for width, depth_array in noisy_results.items()}

    assert set(noisy_results.keys()) == set(ideal_results.keys())

    for width, depth_array in prob_success.items():
        for depth in depth_array.keys():

            noisy_ckt_sample_results = noisy_results[width][depth]
            ideal_ckt_sample_results = ideal_results[width][depth]

            # iterate over circuits
            for noisy_shots, targets in zip(noisy_ckt_sample_results,
                                                 ideal_ckt_sample_results):
                if not isinstance(targets[0], int):
                    targets = [bit_array_to_int(res) for res in targets]

                pr_success = 0
                # determine if each result bitstring is a success, i.e. matches an ideal_result
                for result in noisy_shots:
                    # convert result to int for comparison with heavy outputs.
                    output = bit_array_to_int(result)
                    if output in targets:
                         pr_success += 1 / len(noisy_shots)
                prob_success[width][depth].append(pr_success)

    return prob_success


def calculate_success_prob_est_and_err(num_success: int, num_circuits: int, num_shots: int) \
        -> Tuple[float, float]:
    """
    Helper to calculate the estimate for the probability of sampling a successful output at a
    particular depth as well as the 2 sigma one-sided confidence interval on this estimate.

    :param num_success: total number of successful outputs sampled at particular depth across all
        circuits and shots
    :param num_circuits: the total number of model circuits of a particular depth and width whose
        output was sampled
    :param num_shots: the total number of shots taken for each circuit
    :return: estimate for the probability of sampling a successful output at a particular depth as
        well as the 2 sigma one-sided confidence interval on this estimate.
    """
    total_sampled_outputs = num_circuits * num_shots
    prob_sample_heavy = num_success / total_sampled_outputs

    # Eq. (C3) of [QVol]. Assume that num_heavy/num_shots is worst-case binomial with param
    # num_circuits and take gaussian approximation. Get 2 sigma one-sided confidence interval.
    one_sided_confidence_interval = prob_sample_heavy - \
        2 * np.sqrt(num_success * (num_shots - num_success / num_circuits)) / total_sampled_outputs

    return prob_sample_heavy, one_sided_confidence_interval


def determine_prob_success_lower_bounds(ckt_success_probs, num_shots_per_ckt):
    """
    Wrapper around `calculate_success_prob_est_and_err` to determine success lower bounds for a
    collection of circuits at various depths and widths.

    :param ckt_success_probs:
    :param num_shots_per_ckt:
    :return:
    """
    return {w: {d: calculate_success_prob_est_and_err(
                         sum(np.asarray(succ_probs) * num_shots_per_ckt),
                         len(succ_probs),
                         num_shots_per_ckt)[1]
                for d, succ_probs in d_ckt_succ_probs.items()}
            for w, d_ckt_succ_probs in ckt_success_probs.items()}


def determine_successes(ckt_success_probs: Dict[int, Dict[int, List[float]]], num_shots_per_ckt,
                                     success_threshold: float = 2 / 3):
    """
    Indicate whether the collection of circuit success probabilities for given width and depth
    recorded in `ckt_success_probs` is considered a success with respect to the specified
    `success_threshold` and given the number of shots used to estimate each success probability.

    :param ckt_success_probs:
    :param num_shots_per_ckt:
    :param success_threshold:
    :return:
    """
    lower_bounds = determine_prob_success_lower_bounds(ckt_success_probs, num_shots_per_ckt)
    return {w: {d: lb > success_threshold for d, lb in d_lower_bounds.items()}
            for w, d_lower_bounds in lower_bounds.items()}


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

    min_depth_idx_failure_at_width = []
    for w_idx, w in enumerate(widths):
        if w not in successes.keys():
            min_depth_idx_failure_at_width.append(None)
            continue

        depth_succ = successes[w]
        min_depth_failure = len(depths)
        for d_idx, d in enumerate(depths):
            if d not in depth_succ.keys():
                continue
            if not depth_succ[d]:
                min_depth_failure = d_idx
                break
        min_depth_idx_failure_at_width.append(min_depth_failure)

    for w_idx, failure_idx in enumerate(min_depth_idx_failure_at_width):
        if failure_idx is None:
            continue  # this width was not measured, so leave the boundary open

        # horizontal line for this width
        if failure_idx < len(depths): # measured a failure
            ax.plot((w_idx - margin, w_idx + margin), (failure_idx - margin, failure_idx - margin),
                    color='black')

        # vertical lines
        if w_idx < len(widths) - 1:  # check not at max width
            for d_idx in range(len(depths)):
                # check that the current depth was measured for this width
                if depths[d_idx] not in [d for d in successes[widths[w_idx]].keys()]:
                    continue  # do not plot line if this depth was not measured

                # if the adjacent width is not measured leave the boundary open
                if min_depth_idx_failure_at_width[w_idx + 1] is None:
                    continue

                # check if in the interior but adjacent to exterior
                # or if in the exterior but adjacent to interior
                if failure_idx > d_idx >= min_depth_idx_failure_at_width[w_idx + 1] \
                        or failure_idx <= d_idx < min_depth_idx_failure_at_width[w_idx + 1]:
                    ax.plot((w_idx + margin, w_idx + margin), (d_idx - margin, d_idx + margin),
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
