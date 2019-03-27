"""
Circuit primitives for classical reversible logic

At the moment it is primarily using the simple adder construction in

[CDKM96] A new quantum ripple-carry addition circuit
         Cuccaro, Draper, Kutin, and Moulton
         https://arxiv.org/abs/quant-ph/0410184

There are many other classical logic primitives that can be coded see
e.g.

[VBE96] Quantum networks for elementary arithmetic operations
        Vedral,  Barenco, Ekert
        Phys. Rev. A 54, 147 (1996)
        https://doi.org/10.1103/PhysRevA.54.147
        https://arxiv.org/abs/quant-ph/9511018
"""
from typing import Sequence, Tuple
import networkx as nx
import numpy as np
from numpy import pi
from scipy.spatial.distance import hamming

from pyquil.gates import CNOT, CCNOT, X, I, H, CZ, MEASURE, RESET
from pyquil import Program
from pyquil.quil import Pragma
from pyquil.api import QuantumComputer
from pyquil.unitary_tools import all_bitstrings

from forest.benchmarking.readout import _readout_group_parameterized_bitstring
from forest.benchmarking.classical_logic.primitives import *
from forest.benchmarking.utils import bit_array_to_int, int_to_bit_array


def assign_registers_to_line_or_cycle(start: int, graph: nx.Graph, num_length: int) \
        -> Tuple[Sequence[int], Sequence[int], int, int]:
    """
    From the start node assign registers as they are laid out in the ideal circuit diagram in
    [CDKM96].

    Assumes that the there are no dead ends in the graph, and any available neighbor can be
    selected from the start without any further checks.

    :param start: a node in the graph from which to start the assignment
    :param graph: a graph with an unambiguous assignment from the start node, e.g. a cycle or line
    :param num_length: the length of the bitstring representation of one summand
    :return: the necessary registers and ancilla labels for implementing an adder program to add
        the numbers a and b. The output can be passed directly to adder()
    """
    if 2 * num_length + 2 > nx.number_of_nodes(graph):
        raise ValueError("There are not enough qubits in the graph to support the computation.")

    graph = graph.copy()

    register_a = []
    register_b = []

    # set the node at start, and assign the carry_ancilla to this node.
    node = start
    carry_ancilla = node
    neighbors = list(graph.neighbors(node))

    idx = 0
    while idx < 2 * num_length:
        # remove the last assigned node to ensure it is not reassigned.
        last_node = node
        graph.remove_node(last_node)

        # crawl to an arbitrary neighbor node if possible. If not, the assignment has failed.
        if len(neighbors) == 0:
            raise ValueError("Encountered dead end; assignment failed.")
        node = neighbors[0]
        neighbors = list(graph.neighbors(node))

        # alternate between assigning nodes to the b register and a register, starting with b
        if (idx % 2) == 0:
            register_b.append(node)
        else:
            register_a.append(node)

        idx += 1
    # assign the z_ancilla to a neighbor of the last assignment to a
    z_ancilla = next(graph.neighbors(node))

    return register_a, register_b, carry_ancilla, z_ancilla


def get_qubit_registers_for_adder(qc: QuantumComputer, num_length: int,
                                  qubits: Sequence[int] = None) \
        -> Tuple[Sequence[int], Sequence[int], int, int]:
    """
    Searches for a layout among the given qubits for the two n-bit registers and two additional
    ancilla that matches the simple layout given in figure 4 of [CDKM96].

    This method ignores any considerations of physical characteristics of the qc aside from the
    qubit layout. An error is thrown if the appropriate layout is not found.

    :param qc: the quantum resource on which an adder program will be executed.
    :param num_length: the length of the bitstring representation of one summand
    :param qubits: the available qubits on which to run the adder program.
    :returns the necessary registers and ancilla labels for implementing an adder
        program to add the numbers a and b. The output can be passed directly to adder()
    """
    if qubits is None:
        unavailable = []  # assume this means all qubits in qc are available
    else:
        unavailable = [qubit for qubit in qc.qubits() if qubit not in qubits]

    graph = qc.qubit_topology()
    for qubit in unavailable:
        graph.remove_node(qubit)

    # network x only provides subgraph isomorphism, but we want a subgraph monomorphism, i.e. we
    # specifically want to match the edges desired_layout with some subgraph of graph. To
    # accomplish this, we swap the nodes and edges of graph by making a line graph.
    line_graph = nx.line_graph(graph)

    # We want a path of n nodes, which has n-1 edges. Since we are matching edges of graph with
    # nodes of layout we make a layout of n-1 nodes.
    num_desired_nodes = 2 * num_length + 2
    desired_layout = nx.path_graph(num_desired_nodes - 1)

    g_matcher = nx.algorithms.isomorphism.GraphMatcher(line_graph, desired_layout)

    try:
        # pick out a subgraph isomorphic to the desired_layout if one exists
        # this is an isomorphic mapping from edges in graph (equivalently nodes of line_graph) to
        # nodes in desired_layout (equivalently edges of a path graph with one more node)
        edge_iso = next(g_matcher.subgraph_isomorphisms_iter())
    except IndexError:
        raise Exception("An appropriate layout for the qubits could not be found among the "
                        "provided qubits.")

    # pick out the edges of the isomorphism from the original graph
    subgraph = nx.Graph(graph.edge_subgraph(edge_iso.keys()))

    # pick out an endpoint of our path to start the assignment
    start_node = -1
    for node in subgraph.nodes:
        if subgraph.degree(node) == 1:  # found an endpoint
            start_node = node
            break

    return assign_registers_to_line_or_cycle(start_node, subgraph, num_length)


def prepare_bitstring(bitstring: Sequence[int], register: Sequence[int], in_x_basis: bool = False):
    """
    Creates a program to prepare the input bitstring on the qubits given by the corresponding
    label in the register.

    :param bitstring:
    :param register: a list of qubits on which to prepare the bitstring. The first
    :param in_x_basis: if true, prepare the bitstring-representation of the numbers in the x basis.
    :returns: state_prep_prog - program
    """
    state_prep_prog = Program()

    for bit, qubit_label in zip(bitstring, register):
        if bit == 1:
            state_prep_prog += X(qubit_label)

        # if we are doing logic in X basis, follow each bit preparation with a Hadamard
        # H |0> = |+> and H |1> = |-> where + and - label the X basis vectors.
        if in_x_basis:
            state_prep_prog += H(qubit_label)

    return state_prep_prog


def adder(num_a: Sequence[int], num_b: Sequence[int], register_a: Sequence[int],
          register_b: Sequence[int], carry_ancilla: int, z_ancilla: int, in_x_basis: bool = False,
          use_param_program: bool = False) -> Program:
    """
    Produces a program implementing reversible adding on a quantum computer to compute a + b.

    This implementation is based on [CDKM96], which is easy to implement, if not the most
    efficient. Each register of qubit labels should be provided such that the first qubit in
    each register is expected to carry the least significant bit of the respective number. This
    method also requires two extra ancilla, one initialized to 0 that acts as a dummy initial
    carry bit and another (which also probably ought be initialized to 0) that stores the most
    significant bit of the addition (should there be a final carry). The most straightforward
    ordering of the registers and two ancilla for adding n-bit numbers follows the pattern

        carry_ancilla
        b_0
        a_0
        ...
        b_j
        a_j
        ...
        b_n
        a_n
        z_ancilla

    With this layout, all gates in the circuit act on sets of three adjacent qubits. Such a
    layout is provided by calling get_qubit_registers_for_adder on the quantum resource. Note
    that even with this layout some of the gates used to implement the circuit may not be native.
    In particular there are CCNOT gates which must be decomposed and CNOT(q1, q3) gates acting on
    potentially non-adjacenct qubits (the layout only ensures q2 is adjacent to both q1 and q3).

    The output of the circuit falls on the qubits initially labeled by the b bits (and z_ancilla).

    The default option is to compute the addition in the computational (aka Z) basis. By setting
    in_x_basis true, the gates CNOT_X_basis and CCNOT_X_basis (defined above) will replace CNOT
    and CCNOT so that the computation happens in the X basis.

        [CDKM96]
        "A new quantum ripple-carry addition circuit"
        S. Cuccaro, T. Draper, s. Kutin, D. Moulton
        https://arxiv.org/abs/quant-ph/0410184

    :param num_a: the bitstring representation of the number a with least significant bit last
    :param num_b: the bitstring representation of the number b with least significant bit last
    :param register_a: list of qubit labels for register a, with least significant bit labeled first
    :param register_b: list of qubit labels for register b, with least significant bit labeled first
    :param carry_ancilla: qubit labeling a zero-initialized qubit, ideally adjacent to b_0
    :param z_ancilla: qubit label, a zero-initialized qubit, ideally adjacent to register_a[-1]
    :param in_x_basis: if true, prepare the bitstring-representation of the numbers in the x basis
        and subsequently performs all addition logic in the x basis.
    :param use_param_program: if true, the input num_a and num_b should be arrays of the proper
        length, but their contents will be disregarded. Instead, the program returned will be
        parameterized and the input bitstrings to add must be specified at run time.
    :return: pyQuil program that implements the addition a+b, with output falling on the qubits
        formerly storing the input b. The output of a measurement will list the lsb as the last bit.
    """
    if len(num_a) != len(num_b):
        raise ValueError("Numbers being added must be equal length bitstrings")

    # First, generate a set preparation program in the desired basis.
    prog = Program(Pragma('PRESERVE_BLOCK'))
    if use_param_program:
        # do_measure set to False makes the returned program a parameterized prep program
        input_register = register_a + register_b
        prog += _readout_group_parameterized_bitstring(input_register[::-1], do_measure=False)
        if in_x_basis:
            prog += [H(qubit) for qubit in input_register]
    else:
        prog += prepare_bitstring(num_a[::-1], register_a, in_x_basis)
        prog += prepare_bitstring(num_b[::-1], register_b, in_x_basis)

    if in_x_basis:
        prog += [H(carry_ancilla), H(z_ancilla)]

    # preparation complete; end the preserve block
    prog += Pragma("END_PRESERVE_BLOCK")

    prog_to_rev = Program()
    current_carry_label = carry_ancilla
    for (a, b) in zip(register_a, register_b):
        prog += majority_gate(a, b, current_carry_label, in_x_basis)
        prog_to_rev += unmajority_add_gate(a, b, current_carry_label, in_x_basis).dagger()
        current_carry_label = a

    undo_and_add_prog = prog_to_rev.dagger()
    if in_x_basis:
        prog += CNOT_X_basis(register_a[-1], z_ancilla)
        # need to switch back to computational (z) basis before measuring
        for qubit in register_b:  # answer lays on the b qubit register
            undo_and_add_prog.inst(H(qubit))
        undo_and_add_prog.inst(H(z_ancilla))
    else:
        prog += CNOT(register_a[-1], z_ancilla)
    prog += undo_and_add_prog

    ro = prog.declare('ro', memory_type='BIT', memory_size=len(register_b) + 1)
    for idx, qubit in enumerate(register_b):
        prog += MEASURE(qubit, ro[len(register_b) - idx])
    prog += MEASURE(z_ancilla, ro[0])

    return prog


def get_n_bit_adder_results(qc: QuantumComputer, n_bits: int,
                            registers: Tuple[Sequence[int], Sequence[int], int, int] = None,
                            qubits: Sequence[int] = None, in_x_basis: bool = False,
                            num_shots: int = 100, use_param_program: bool = True,
                            use_active_reset: bool = True) -> Sequence[Sequence[Sequence[int]]]:
    """
    Convenient wrapper for collecting the results of addition for every possible pair of n_bits
    long summands.

    :param qc: the quantum resource on which to run each addition
    :param n_bits: the number of bits of one of the summands (each summand is the same length)
    :param registers: optional explicit qubit layout of each of the registers passed to adder()
    :param qubits: available subset of qubits of the qc on which to run the circuits.
    :param in_x_basis: if true, prepare the bitstring-representation of the numbers in the x basis
        and subsequently performs all addition logic in the x basis.
    :param num_shots: the number of times to sample the output of each addition
    :param use_param_program: whether or not to use a parameterized program for state preparation.
        Doing so should speed up overall execution on a QPU.
    :param use_active_reset: whether or not to use active reset. Doing so will speed up execution
        on a QPU.
    :return: A list of n_shots many outputs for each possible summation of two n_bit long summands,
        listed in increasing numerical order where the label is the 2n bit number represented by
        num = a_bits | b_bits for the addition of a + b.
    """
    if registers is None:
        registers = get_qubit_registers_for_adder(qc, n_bits, qubits)

    reset_prog = Program()
    if use_active_reset:
        reset_prog += RESET()

    add_prog = Program()
    if use_param_program:
        dummy_num = [0 for _ in range(n_bits)]
        add_prog = adder(dummy_num, dummy_num, *registers, in_x_basis=in_x_basis,
                         use_param_program=True)

    all_results = []
    # loop over all binary strings of length n_bits
    for bits in all_bitstrings(2 * n_bits):
        # split the binary number into two numbers
        # which are the binary numbers the user wants to add.
        # They are written from (MSB .... LSB) = (a_n, ..., a_1, a_0)
        num_a = bits[:n_bits]
        num_b = bits[n_bits:]

        if not use_param_program:
            add_prog = adder(num_a, num_b, *registers, in_x_basis=in_x_basis,
                             use_param_program=False)

        prog = reset_prog + add_prog
        prog.wrap_in_numshots_loop(num_shots)
        nat_quil = qc.compiler.quil_to_native_quil(prog)
        exe = qc.compiler.native_quil_to_executable(nat_quil)

        # Run it on the QPU or QVM
        if use_param_program:
            results = qc.run(exe, memory_map={'target': [bit * pi for bit in bits]})
        else:
            results = qc.run(exe)
        all_results.append(results)

    return all_results


def get_success_probabilities_from_results(results: Sequence[Sequence[Sequence[int]]]) \
        -> Sequence[float]:
    """
    Get the probability of a successful addition for each possible pair of two n_bit summands
    from the results output by get_n_bit_adder_results

    :param results: a list of results output from a call to get_n_bit_adder_results
    :return: the success probability for the summation of each possible pair of n_bit summands
    """
    num_shots = len(results[0])
    n_bits = len(results[0][0]) - 1

    probabilities = []
    # loop over all binary strings of length n_bits
    for result, bits in zip(results, all_bitstrings(2 * n_bits)):
        # Input nums are written from (MSB .... LSB) = (a_n, ..., a_1, a_0)
        num_a = bit_array_to_int(bits[:n_bits])
        num_b = bit_array_to_int(bits[n_bits:])

        # add the numbers
        ans = num_a + num_b
        ans_bits = int_to_bit_array(ans, n_bits + 1)

        # a success occurs if a shot matches the expected ans bit for bit
        probability = 0
        for shot in result:
            if np.array_equal(ans_bits, shot):
                probability += 1. / num_shots
        probabilities.append(probability)

    return probabilities


def get_error_hamming_distributions_from_results(results: Sequence[Sequence[Sequence[int]]]) \
        -> Sequence[Sequence[float]]:
    """
    Get the distribution of the hamming weight of the error vector (number of bits flipped
    between output and expected answer) for each possible pair of two n_bit summands using
    results output by get_n_bit_adder_results

    :param results: a list of results output from a call to get_n_bit_adder_results
    :return: the relative frequency of observing each hamming weight, 0 to n_bits+1, for the error
        that occurred when adding each pair of two n_bit summands
    """
    num_shots = len(results[0])
    n_bits = len(results[0][0]) - 1

    hamming_wt_distrs = []
    # loop over all binary strings of length n_bits
    for result, bits in zip(results, all_bitstrings(2 * n_bits)):
        # Input nums are written from (MSB .... LSB) = (a_n, ..., a_1, a_0)
        num_a = bit_array_to_int(bits[:n_bits])
        num_b = bit_array_to_int(bits[n_bits:])

        # add the numbers
        ans = num_a + num_b
        ans_bits = int_to_bit_array(ans, n_bits + 1)

        # record the fraction of shots that resulted in an error of the given weight
        hamming_wt_distr = [0. for _ in range(len(ans_bits) + 1)]
        for shot in result:
            # multiply relative hamming distance by the length of the output for the weight
            wt = len(ans_bits) * hamming(ans_bits, shot)
            hamming_wt_distr[int(wt)] += 1. / num_shots

        hamming_wt_distrs.append(hamming_wt_distr)

    return hamming_wt_distrs
