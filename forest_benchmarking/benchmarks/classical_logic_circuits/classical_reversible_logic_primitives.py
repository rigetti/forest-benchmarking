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
from typing import Sequence

import numpy as np
from pyquil.gates import CNOT, CCNOT, X, I, H, CZ
from pyquil.quil import Program


def get_qubit_labels(num_a):
    """
    A naive choice qubits to run the adder.

    :param num_a: A tuple of strings.
    :returns qubit_labels: A list of ints.
    """
    # this part can be optimized by hand
    qbit_labels = list(range(2 * len(num_a) + 2))
    return qbit_labels


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


def majority_gate(a: int, b: int, c: int, CNOTfunc: Callable[[int, int], Program] = CNOT,
                  CCNOTfunc: Callable[[int, int, int], Program] = CCNOT) -> Program:
    """
    The majority gate.

    Computes (a * b) xor (a * c) xor  (b * c)
    where * is multiplication mod 2.

    The default option is to compute this in the computational (aka Z) basis. By passing in
    CNOTfun and CCNOTfun as CNOT_X_basis and CCNOT_X_basis the computation happens in the X basis.

    See https://arxiv.org/abs/quant-ph/0410184 .
    
    :param a: qubit label
    :param b: qubit label
    :param c: qubit label
    :param CNOTfunc: either CNOT or CNOT_X_basis
    :param CCNOTfunc: either CCNOT or CCNOT_X_basis
    :return: program which results in (c xor a) on the c line, (b xor a) on the b line,
        and the output (majority of the inputs) on the a line.
    """
    prog = Program()
    prog += CNOTfunc(a, b)
    prog += CNOTfunc(a, c)
    prog += CCNOTfunc(c, b, a)
    return prog


def unmajority_add_gate(a: int, b: int, c: int, CNOTfunc: Callable[[int, int], Program] = CNOT,
                        CCNOTfunc: Callable[[int, int, int], Program] = CCNOT) -> Program:
    """
    The UnMajority and Add or UMA gate

    See https://arxiv.org/abs/quant-ph/0410184 .

    The default option is to compute this in the computational (aka Z) basis. By passing in
    CNOTfun and CCNOTfun as CNOT_X_basis and CCNOT_X_basis the computation happens in the X basis.

    :param a: qubit label
    :param b: qubit label
    :param c: qubit label
    :param CNOTfunc: either CNOT or CNOT_X_basis
    :param CCNOTfunc: either CCNOT or CCNOT_X_basis
    :return: program which when run on the output of majority_gate(a,b,c) returns the input to
        majority_gate on the c and a lines, and outputs the sum of a+b+c (mod 2) on the b line.
    """
    prog = Program()
    prog += CCNOTfunc(c, b, a)
    prog += CNOTfunc(a, c)
    prog += CNOTfunc(c, b)
    return prog


def unmajority_add_parallel_gate(a: int, b: int, c: int,
                                 CNOTfunc: Callable[[int, int], Program] = CNOT,
                                 CCNOTfunc: Callable[[int, int, int], Program] = CCNOT) -> Program:
    """
    The UnMajority and Add or UMA gate

    3-CNOT version but admits greater parallelism
    
    See https://arxiv.org/abs/quant-ph/0410184 .

    Computes
    :param a: qubit label
    :param b: qubit label
    :param c: qubit label
    :return: program that executes the same logic as unmajority_add_gate but with different gates
    """
    prog = Program()
    prog += X(b)
    prog += CNOTfunc(a, b)
    prog += CCNOTfunc(a, b, c)
    prog += X(b)
    prog += CNOTfunc(c, a)
    prog += CNOTfunc(c, b)
    return prog


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

    This implementation is based on [ADD-CKT], which is easy to implement, if not the most
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

        [ADD-CKT]
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
        formerly storing the input b.
    """
    if len(register_b) != len(register_a):
        raise ValueError("Registers must be equal length")

    prep_a = prepare_bitstring(reversed(num_a), register_a, in_x_basis)
    prep_b = prepare_bitstring(reversed(num_b), register_b, in_x_basis)

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
