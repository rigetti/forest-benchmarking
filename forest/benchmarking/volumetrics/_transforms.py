from typing import List
import networkx as nx
from copy import copy

from pyquil.quil import Program, address_qubits, merge_programs
from pyquil.api import QuantumComputer
from pyquil.gates import *
from rpcq.messages import TargetDevice
from rpcq._utils import RPCErrorError
from forest.benchmarking.compilation import basic_compile
from forest.benchmarking.volumetrics._generators import random_single_qubit_gates


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
    new_sequence = [None for _ in range(2 * len(sequence) + 1)]
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
