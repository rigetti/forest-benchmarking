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
from pyquil.gates import CNOT, CCNOT, X, I, H, CZ, MEASURE, RESET
from pyquil import Program
from pyquil.api import QuantumComputer
from pyquil.unitary_tools import all_bitstrings

from forest_benchmarking.readout import _readout_group_parameterized_bitstring


def CNOT_X_basis(control, target) -> Program:
    """
    The CNOT in the X basis, i.e.

    CNOTX = |+X+| otimes I + |-X-| otimes Z

    where |+> and |-> are the +/- eigenstate of the Pauli X operator.

    :param control: qubit label
    :param target: qubit label
    :return: program
    """
    prog = Program()
    prog += H(control)
    prog += CZ(control, target)
    prog += H(control)
    return prog


def CCNOT_X_basis(control1, control2, target) -> Program:
    """
    The CCNOT (Toffoli) in the X basis, i.e.

    CCNOTX = |+X+| * |+X+| * I +
             |+X+| * |-X-| * I +
             |-X-| * |+X+| * I +
             |-X-| * |-X-| * Z

    where |+> and |-> are the +/- eigenstate of the Pauli X operator and * denotes a tensor product.

    :param control1: qubit label
    :param control2: qubit label
    :param target: qubit label
    :return: program
    """
    prog = Program()
    prog += H(control1)
    prog += H(control2)
    prog += H(target)
    prog += CCNOT(control1, control2, target)
    prog += H(control1)
    prog += H(control2)
    prog += H(target)
    return prog


def majority_gate(a: int, b: int, c: int, in_x_basis: bool = False) -> Program:
    """
    The majority gate.

    Computes (a * b) xor (a * c) xor  (b * c)
    where * is multiplication mod 2.

    The default option is to compute this in the computational (aka Z) basis. By passing in
    CNOTfun and CCNOTfun as CNOT_X_basis and CCNOT_X_basis the computation happens in the X basis.

    See [CDKM96] reference in adder() docstring
    
    :param a: qubit label
    :param b: qubit label
    :param c: qubit label
    :param in_x_basis: if true, the returned program performs the equivalent logic in the X basis.
    :return: program which results in (c xor a) on the c line, (b xor a) on the b line,
        and the output (majority of the inputs) on the a line.
    """
    if in_x_basis:
        cnot_gate = CNOT_X_basis
        ccnot_gate = CCNOT_X_basis
    else:
        cnot_gate = CNOT
        ccnot_gate = CCNOT

    prog = Program()
    prog += cnot_gate(a, b)
    prog += cnot_gate(a, c)
    prog += ccnot_gate(c, b, a)
    return prog


def unmajority_add_gate(a: int, b: int, c: int, in_x_basis: bool = False) -> Program:
    """
    The UnMajority and Add or UMA gate

    The default option is to compute this in the computational (aka Z) basis. By passing in
    CNOTfun and CCNOTfun as CNOT_X_basis and CCNOT_X_basis the computation happens in the X basis.
    
    See [CDKM96] reference in adder() docstring

    :param a: qubit label
    :param b: qubit label
    :param c: qubit label
    :param in_x_basis: if true, the returned program performs the equivalent logic in the X basis.
    :return: program which when run on the output of majority_gate(a,b,c) returns the input to
        majority_gate on the c and a lines, and outputs the sum of a+b+c (mod 2) on the b line.
    """
    if in_x_basis:
        cnot_gate = CNOT_X_basis
        ccnot_gate = CCNOT_X_basis
    else:
        cnot_gate = CNOT
        ccnot_gate = CCNOT

    prog = Program()
    prog += ccnot_gate(c, b, a)
    prog += cnot_gate(a, c)
    prog += cnot_gate(c, b)
    return prog


def unmajority_add_parallel_gate(a: int, b: int, c: int, in_x_basis: bool = False) -> Program:
    """
    The UnMajority and Add or UMA gate

    3-CNOT version but admits greater parallelism
    
    See [CDKM96] reference in adder() docstring

    Computes
    :param a: qubit label
    :param b: qubit label
    :param c: qubit label
    :param in_x_basis: if true, the returned program performs the equivalent logic in the X basis.
    :return: program that executes the same logic as unmajority_add_gate but with different gates
    """
    if in_x_basis:
        cnot_gate = CNOT_X_basis
        ccnot_gate = CCNOT_X_basis
    else:
        cnot_gate = CNOT
        ccnot_gate = CCNOT

    prog = Program()
    prog += X(b)
    prog += cnot_gate(a, b)
    prog += ccnot_gate(a, b, c)
    prog += X(b)
    prog += cnot_gate(c, a)
    prog += cnot_gate(c, b)
    return prog


def assign_registers_to_line_or_cycle(start: int, graph: nx.Graph, num_length: int):
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
        # H |0> = |+> and H |1> = |-> where + and - label the x basis vectors.
        if in_x_basis:
            state_prep_prog += H(qubit_label)

    return state_prep_prog


def adder(num_a: Sequence[int], num_b: Sequence[int], register_a: Sequence[int] = None,
          register_b: Sequence[int] = None, carry_ancilla: int = None, z_ancilla: int = None,
          in_x_basis: bool = False) -> Program:
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

    With this ordering, all gates in the circuit act on sets of three adjacent qubits. The output of
    the circuit correspondingly falls on the qubits initially labeled by the b bits (and z_ancilla).
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
    :return: pyQuil program that implements the addition a+b, with output falling on the qubits
        formerly storing the input b. The output of a measurement will list the lsb as the last bit.
    """
    if len(num_a) != len(num_b):
        raise ValueError("Numbers being added must be equal length bitstrings")

    if register_a is None:
        register_a = [q*2+2 for q in range(len(num_a))]
    if register_b is None:
        register_b = [q*2+1 for q in range(len(num_a))]
    if carry_ancilla is None:
        carry_ancilla = 0
    if z_ancilla is None:
        z_ancilla = 2*len(num_a) + 1

    prep_a = prepare_bitstring(num_a[::-1], register_a, in_x_basis)
    prep_b = prepare_bitstring(num_b[::-1], register_b, in_x_basis)

    prog = prep_a + prep_b
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

    ro = prog.declare('ro', memory_type='BIT', memory_size=len(register_b)+1)
    for idx, qubit in enumerate(register_b):
        prog += MEASURE(qubit, ro[len(register_b)-idx])
    prog += MEASURE(z_ancilla, ro[0])

    return prog


def bit_array_to_int(bit_array: Sequence[int]) -> int:
    """
    Converts a bit array into an integer where the right-most bit is least significant.

    :param bit_array: an array of bits with right-most bit considered least significant.
    :return: the integer corresponding to the bitstring.
    """
    output = 0
    for bit in bit_array:
        output = (output << 1) | bit
    return output


def int_to_bit_array(num: int, n_bits: int) ->  Sequence[int]:
    """
    Converts a number into an array of bits where the right-most bit is least significant.

    :param num: the integer corresponding to the bitstring.
    :param n_bits: the number of bits to report
    :return:  an array of n_bits bits with right-most bit considered least significant.
    """
    return [num >> bit & 1 for bit in range(n_bits - 1, -1, -1)]


def get_n_bit_adder_results(qc: QuantumComputer, n_bits: int,
                            registers: Tuple[Sequence[int], Sequence[int], int, int] = None,
                            in_x_basis: bool = False, num_shots: int = 10,
                            use_param_program: bool = True, use_active_reset: bool = True) \
                            -> Tuple[Sequence[float], Sequence[float]]:
    """

    :param qc:
    :param n_bits:
    :param registers:
    :param in_x_basis:
    :param num_shots:
    :param use_param_program:
    :param use_active_reset:
    :return:
    """
    all_results = []
    # loop over all binary strings of length n_bits
    for bits in all_bitstrings(2 * n_bits):
        # split the binary number into two numbers
        # which are the binary numbers the user wants to add.
        # They are written from (MSB .... LSB) = (a_n, ..., a_1, a_0)
        num_a = bits[:n_bits]
        num_b = bits[n_bits:]

        add_prog = Program()
        if use_active_reset:
            add_prog += RESET()

        # create the program and compile appropriately
        if registers is None:
            add_prog = adder(num_a, num_b, in_x_basis=in_x_basis)
            add_prog.wrap_in_numshots_loop(num_shots)
            add_exe = qc.compile(add_prog)
        else:
            add_prog = adder(num_a, num_b, *registers, in_x_basis=in_x_basis)
            add_prog.wrap_in_numshots_loop(num_shots)
            add_exe = qc.compiler.native_quil_to_executable(add_prog)

        # Run it on the QPU or QVM
        results = qc.run(add_exe)
        all_results.append(results)

    return all_results


def construct_bit_flip_error_histogram(wt, n):
    """
    From experimental data construct the Hamming weight histogram of answers relative to a the
    length of binary numbers being added.
    
    :params wt: numpy array 2**(2n) by number_of_trials
    :params n: number of bits being added
    :returns: numpy histogram with bins corresponding to [0,...,n+3] 
    """
    # determine hamming weight histogram
    histy = np.zeros([2 ** (2 * n), n + 2])
    for sdx in range(2 ** (2 * n)):
        hist, bins = np.histogram(wt[sdx, :], list(np.arange(0, n + 3)))
        histy[sdx] = hist
    return histy
