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
from pyquil.quil import Program
from pyquil.gates import CNOT, CCNOT, X


def majority_gate(a, b, c):
    """
    The majority gate.

    Computes (a * b) xor (a * c) xor  (b * c)
    where * is multiplication mod 2

    See https://arxiv.org/abs/quant-ph/0410184 .
    
    :param a: qubit label
    :param b: qubit label
    :param c: qubit label
    :return: program
    """
    prog = Program()
    prog += CNOT(c, b)
    prog += CNOT(c, a)
    prog += CCNOT(a, b, c)
    return prog


def unmajority_add_gate(a, b, c):
    """
    The UnMajority and Add or UMA gate

    See https://arxiv.org/abs/quant-ph/0410184 .

    :param a: qubit label
    :param b: qubit label
    :param c: qubit label
    :return: program
    """
    prog = Program()
    prog += CCNOT(a, b, c)
    prog += CNOT(c, a)
    prog += CNOT(a, b)
    return prog


def unmajority_add_parallel_gate(a, b, c):
    """
    The UnMajority and Add or UMA gate

    3-CNOT version but admits greater parallelism
    
    See https://arxiv.org/abs/quant-ph/0410184 .

    Computes
    :param a: qubit label
    :param b: qubit label
    :param c: qubit label
    :return: program
    """
    prog = Program()
    prog += X(b)
    prog += CNOT(a, b)
    prog += CCNOT(a, b, c)
    prog += X(b)
    prog += CNOT(c, a)
    prog += CNOT(c, b)
    return prog

def adder(register_a, register_b, carry_ancilla=None, z_ancilla=None):
    """
    Reversable adding on a quantum computer.

    This implementation is based on:

    "A new quantum ripple-carry addition circuit"
    S. Cuccaro, T. Draper, s. Kutin, D. Moulton
    https://arxiv.org/abs/quant-ph/0410184

    It is not the most efficent but it is easy to implement.

    This method requires two extra ancilla, one for a carry bit 
    and one for fully reversable computing.

    :param register_a: list of qubit labels for register a
    :param register_b: list of qubit labels for register b
    :return: pyQuil program of adder
    :rtype: Program
    """
    if len(register_b) != len(register_a):
        raise ValueError("Registers must be equal length")

    if carry_ancilla is None:
        carry_ancilla = max(register_a + register_b) + 1
    if z_ancilla is None:
        z_ancilla = max(register_a + register_b) + 2

    prog = Program()
    # program to add the numbers on the old QPU the Pragma was required.
    # will delete this after it has been tested on the hardware...
    # prog += Pragma("PRESERVE_BLOCK")
    prog_to_rev = Program()
    carry_ancilla_temp = carry_ancilla
    for (a, b) in zip(register_a, register_b):
        prog += majority_gate(carry_ancilla_temp, b, a)
        prog_to_rev += unmajority_add_gate(carry_ancilla_temp, b, a).dagger()
        carry_ancilla_temp = a

    prog += CNOT(register_a[-1], z_ancilla)
    prog += prog_to_rev.dagger()
    #prog += Pragma("END_PRESERVE_BLOCK")
    return prog

# helper programs
def check_binary_number_length(num_a,num_b):
    """
    Validates the input binary strings.
    
    :param num_a: A tuple of strings.
    :param num_b: A tuple of strings.
    :returns: nothing.
    """
    # do some checks then create the qubit labels
    if len(num_a) != len(num_b):
        raise ValueError("Registers must be equal length")
    elif len(num_a)==0 or len(num_b)==0:
        raise ValueError("Registers must be of nonzero length")
    elif len(num_a)> num_of_qubits:
        raise ValueError("Number is too big to add on the QVM or QPU")

def get_qubit_labels(num_a):
    """
    A naive choice qubits to run the adder.
    
    :param num_a: A tuple of strings.
    :returns qubit_labels: A list of ints.
    """
    # this part can be optimized by hand
    qbit_labels = list(range(2*len(num_a)+2))
    return qbit_labels

def prepare_binary_numbers(num_a,num_b,qbit_labels):
    """
    Takes the input binary numbers and creates a program
    to prepare that input string on qubits in to quantum 
    registers that are interleaved in the approprite way for 
    the ripple carry adder.
    
    :param num_a: tuple of strings representing the first binary number.
    :param num_b: tuple of strings representing the second binary number.
    :param qbit_labels: list of qubits the adder will run on.
    :returns: tuple containing the following objects
            state_prep_prog - program 
            register_a - qubit labels of register a
            register_b - qubit labels of register a
            carry_ancilla - qubit label of the carry bit
            z_ancilla - necessary additional ancilla 
    """
    register_a = []
    register_b = []
    num_a_idx =len(num_a)-1
    num_b_idx =len(num_a)-1

    # this is a hack because Quil wont let you have blank programs
    state_prep_prog = Program().inst(I(0))
    # We actually want to run the gates this prevents the compiler from messing with things
    #state_prep_prog += Pragma("PRESERVE_BLOCK")
    
    state_prep_prog += I(qbit_labels[0])
    
    # The numbers in the "a" and "b" register are interleaved in the correct way
    # for the ripple carry adder.
    for qbit_idx in qbit_labels[1:-1]:
        
        # even qubits are "register a"
        if qbit_idx%2 == 0:
            if num_a[num_a_idx] ==1:
                state_prep_prog += X(qbit_idx)
            else:
                state_prep_prog += I(qbit_idx)
            register_a += [qbit_idx] 
            num_a_idx -=1
        
        # odd qubits are "register b"
        if (qbit_idx%2) != 0:
            if num_b[num_b_idx] ==1:
                state_prep_prog += X(qbit_idx)
            else:
                state_prep_prog += I(qbit_idx)
            register_b += [qbit_idx]
            num_b_idx -=1
    
    state_prep_prog += I(qbit_labels[-1])
    #state_prep_prog += Pragma("END_PRESERVE_BLOCK")

    carry_ancilla = 0
    z_ancilla = qbit_labels[-1]
    
    return (state_prep_prog,register_a, register_b, carry_ancilla, z_ancilla)

def construct_all_possible_input_numbers(n):
    """
    Construct a list of lists that contains all binary strings of length 2n.
    We will split this into two lists of length n. These represent all possible inputs to the adder.
    
    :param n: integer representing the length of binary numbers to add.
    :return bin_str: a list of lists that contains all binary strings of length 2n
    """
    # count in binary save as a list
    for bdx in range(0,2**(2*n)):
        bin_str.append([int(x) for x in format(bdx,'0'+str(2*n)+'b')])
    return bin_str

def construct_bit_flip_error_histogram(wt, n):
    """
    From experimental data construct the Hamming 
    weight histogram of answers relative to a the length 
    of binary numbers being added.
    
    :params wt: numpy array 2**(2n) by number_of_trials
    :params n: number of bits being added
    :returns: numpy histogram with bins corresponding to [0,...,n+3] 
    """
    # determine hamming weight histogram
    histy = np.zeros([2**(2*n),n+2])
    for sdx in range(2**(2*n)):
        hist, bins = np.histogram(wt[sdx,:],list(np.arange(0,n+3)))
        histy[sdx] = hist
    return histy