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
from typing import Sequence, Tuple, List
import networkx as nx
import warnings

import numpy as np
from pyquil.gates import CNOT, CCNOT, X, I, H, CZ, MEASURE
from pyquil import Program
from pyquil.api import QuantumComputer


def CNOT_X_basis(control, target):
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


def CCNOT_X_basis(control1, control2, target):
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


def _alternate_assign_from_start(start, graph: nx.Graph, num_a, num_b):
    register_a = []
    register_b = []

    carry_ancilla = start
    last_node = carry_ancilla
    next_node = list(graph.neighbors(last_node))[0]
    graph.remove_node(last_node)
    idx = 0
    while idx < len(num_a) + len(num_b):
        if (idx % 2) == 0:
            register_b.append(next_node)
        else:
            register_a.append(next_node)

        last_node = next_node
        next_node = list(graph.neighbors(last_node))[0]
        graph.remove_node(last_node)
        idx += 1
    z_ancilla = next_node

    return register_a, register_b, carry_ancilla, z_ancilla


def get_qubit_registers_for_adder(qc: QuantumComputer, num_a: Sequence[int], num_b: Sequence[int],
                                  qubits: Sequence[int] = None) \
        -> Tuple[List[int], List[int], int, int]:
    """
    Searches for a layout among the given qubits for the two n-bit registers and two additional
    ancilla that matches the simple layout given in figure 4 of [CDKM96]. If such a layout is not
    found then a heuristic is applied.

    This method ignores any considerations of physical characteristics of the qc aside from the
    qubit layout. The search is specifically tailored to the topology of current Aspen QPUs. In
    particular it is assumed that there are no nodes of degree more than 3.

    :param qc: the quantum resource on which an adder program will be executed.
    :param num_a: the bitstring representation of the number a with least significant bit last
    :param num_b: the bitstring representation of the number b with least significant bit last
    :param qubits: the available qubits on which to run the adder program.
    :returns the necessary registers and ancilla labels for implementing an adder
        program to add the numbers a and b. The output can be passed directly to adder()
    """
    if qubits is None:
        unavailable = []
    else:
        unavailable = [qubit for qubit in qc.qubits() if qubit not in qubits]

    if len(num_a) != len(num_b):
        raise ValueError("A and B bitstrings must be equal length.")

    graph = qc.qubit_topology()
    for qubit in unavailable:
        graph.remove_node(qubit)

    graph = max(nx.connected_component_subgraphs(graph), key=len)
    if len(num_a) + len(num_b) + 2 > nx.number_of_nodes(graph):
        raise ValueError("The largest connected component among the available qubits is not large"
                         "enough to support the adder computation.")
    deg_one_nodes = []
    deg_three_nodes = []
    for node in nx.nodes(graph):
        degree = graph.degree(node)
        if degree == 1:
            deg_one_nodes.append(node)
        if degree == 3:
            deg_three_nodes.append(node)
        if degree > 3:
            warnings.warn("This method is specifically tailored to Aspen QPU topology."
                          " An arbitrary assignment will be made.")
            qubits = list(graph.nodes)
            return qubits[:len(num_a)], qubits[len(num_a):2*len(num_a)], qubits[-2], qubits[-1]

    num_deg_three_nodes = len(deg_three_nodes)  # break into special cases around this info
    if num_deg_three_nodes == 0:  # no branches, look for loop or line
        if len(deg_one_nodes) == 0:  # must be a loop. Pick a node and go around.
            start_node = list(graph.nodes)[0]
        else:
            start_node = deg_one_nodes[0]  # must be a line. Pick an endpoint to start at.
    else:
        # cycles = nx.algorithms.cycles.cycle_basis(largest_connected_component)
        if num_deg_three_nodes == 1:
            branch_node = deg_three_nodes[0]
            branches = _explore_branches(graph, branch_node)
            min_length_branch = (0, 0, 0, 0)
            for branch in branches:
                if branch[2] == 3:  # looped back to itself.
                    # can cut loop and start assignment at the new endpoint.
                    graph.remove_edge(branch_node, branch[0])
                    return _alternate_assign_from_start(branch[0], graph, num_a, num_b)
                length = branch[3]
                if length < min_length_branch[3]:
                    min_length_branch = branch
            # remove the smallest branch and assign starting from a remaining endpoint.
            graph.remove_edge(branch_node, min_length_branch[0])
            # get largest connected component
            graph = max(nx.connected_component_subgraphs(graph), key=len)

            if nx.number_of_nodes(graph) < 2 * len(num_a) + 2:
                warnings.warn("The method failed to find an appropriate assignment.")
                qubits = list(graph.nodes)
                return qubits[:len(num_a)], qubits[len(num_a):2 * len(num_a)], qubits[-2], \
                       qubits[-1]
            else:
                start_node = (set(deg_one_nodes) - {min_length_branch[1]}).pop()
        else:
            # give up on special cases.
            warnings.warn("The method failed to find an appropriate assignment.")
            qubits = list(graph.nodes)
            return qubits[:len(num_a)], qubits[len(num_a):2 * len(num_a)], qubits[-2], qubits[-1]

    return _alternate_assign_from_start(start_node, graph, num_a, num_b)


def _explore_branches(graph, branch_node):
    branches = []

    for node in graph.neighbors(branch_node):
        branch_origin = node

        length = 1
        last_node = branch_node
        current_node = node
        neighbors = set(graph.neighbors(current_node))

        while len(neighbors) == 2:
            neighbor = neighbors - {last_node}
            last_node = current_node
            current_node = neighbor.pop()
            length += 1
            neighbors = set(graph.neighbors(current_node))

        branches.append((branch_origin, current_node, len(neighbors), length))

    return branches


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


def adder(num_a: Sequence[int], num_b: Sequence[int], register_a: Sequence[int],
          register_b: Sequence[int], carry_ancilla: int, z_ancilla: int, in_x_basis: bool = False)\
        -> Program:
    """
    Produces a program implementing reversible adding on a quantum computer to compute a + b.

    This implementation is based on [CDKM96], which is easy to implement, if not the most
    efficient. Each regesiter of qubit labels should be provided such that the first qubit in
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
    The default option is to compute the addition in the computational (aka Z) basis. By passing in
    CNOTfunc and CCNOTfunc as CNOT_X_basis and CCNOT_X_basis (defined above) the computation
    happens in the X basis.

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
    if len(register_b) != len(register_a):
        raise ValueError("Registers must be equal length")

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
