import itertools
from collections import OrderedDict
from random import random, seed

import numpy as np
from numpy import pi

from pyquil.gates import I, RX, RY, RZ
from pyquil.paulis import PauliTerm
from pyquil.quil import Program


def pack_shot_data(shot_data):
    return np.packbits(shot_data, axis=1)


def str_to_pauli_term(pauli_str: str, qubit_labels=None):
    """
    Convert a string into a pyquil.paulis.PauliTerm.

    >>> str_to_pauli_term('XY', [])

    :param str pauli_str: The input string, made of of 'I', 'X', 'Y' or 'Z'
    :param set qubit_labels: The integer labels for the qubits in the string, given in reverse
    order. If None, default to the range of the length of pauli_str.
    :return: the corresponding PauliTerm
    :rtype: pyquil.paulis.PauliTerm
    """
    if qubit_labels is None:
        labels_list = [idx for idx in reversed(range(len(pauli_str)))]
    else:
        labels_list = sorted(qubit_labels)[::-1]
    pauli_term = PauliTerm.from_list(list(zip(pauli_str, labels_list)))
    return pauli_term


def local_sic_prep(label, qubit):
    """

    :param label:
    :param qubit:
    :return:
    """
    theta = 2*np.arccos(1/np.sqrt(3))
    zx_plane_rotation = Program(RX(-pi/2, qubit)).inst(RZ(theta - pi, qubit)).inst(RX(-pi/2, qubit))
    if label == 'SIC0':
        gate = I(qubit)
    elif label == 'SIC1':
        gate = zx_plane_rotation
    elif label == 'SIC2':
        gate = zx_plane_rotation.inst(RZ(-2*pi/3, qubit))
    elif label == 'SIC3':
        gate = zx_plane_rotation.inst(RZ(2*pi/3, qubit))
    else:
        raise ValueError('Unknown gate operation')
    return gate


def prepare_prod_sic_state(ops):
    prog = Program()
    for op in ops:
        label, qubit = op.split('_')
        prog.inst(local_sic_prep(label, int(qubit)))
    return prog


def all_sic_terms(qubit_count: int, qubit_labels=None):
    SICS = ['SIC' + str(j) for j in range(4)]
    labels = [op for op in itertools.product(SICS, repeat=qubit_count)]
    if qubit_labels is None:
        qubit_labels = range(qubit_count)
    qubit_labels = sorted(qubit_labels)[::-1]
    return [tuple([op[q] + '_' + str(qubit) for q, qubit in enumerate(qubit_labels)]) for op in labels]


def all_pauli_terms(qubit_count: int, qubit_labels=None):
    """
    Generate list of all Pauli terms (with weight > 0) on N qubits.

    :param int qubit_count: The number of qubits
    :param set qubit_labels: The integer labels for the qubits
    :return: list of `PauliTerm`s
    :rtype: list
    """
    # we exclude the all identity string since that maps to no preparation and no measurement
    all_ixyz_strs = [''.join(x) for x in itertools.product('IXYZ', repeat=qubit_count)][1:]
    list_of_terms = [str_to_pauli_term(s, qubit_labels) for s in all_ixyz_strs]
    return list_of_terms


def all_pauli_z_terms(qubit_count: int, qubit_labels=None):
    """
    Generate list of all Pauli Z terms (with weight > 0) on N qubits

    :param int n: The number of qubits
    :param set qubit_labels: The integer labels for the qubits
    :return: list of `PauliTerm`s
    :rtype: list
    """
    # we exclude the all identity string since that maps to no preparation and no measurement
    all_iz_strs = [''.join(x) for x in itertools.product('IZ', repeat=qubit_count)][1:]
    list_of_terms = [str_to_pauli_term(s, qubit_labels) for s in all_iz_strs]
    return list_of_terms


def local_pauli_eig_prep(prog, op, idx):
    """
    Generate gate sequence to prepare a the +1 eigenstate of a Pauli operator, assuming
    we are starting from the ground state ( the +1 eigenstate of Z^{\\otimes n}), and append it to
    the PyQuil program given. This mutates prog.

    :param Program prog: The program to which state preparation will be attached.
    :param str op: A string representation of the Pauli operator whose eigenstate we'd like to prepare.
    :param int idx: The index of the qubit that the preparation is acting on
    :return: The mutated Program.
    """
    if op == 'X':
        gate = RY(pi / 2, idx)
    elif op == 'Y':
        gate = RX(-pi / 2, idx)
    elif op == 'Z':
        gate = I(idx)
    else:
        raise ValueError('Unknown gate operation')
    prog.inst(gate)
    return prog


def local_pauli_eigs_prep(op, idx):
    """
    Generate all gate sequences to prepare all eigenstates of a (local) Pauli operator, assuming
    we are starting from the ground state.

    :param str op: A string representation of the Pauli operator whose eigenstate we'd like to prepare.
    :param int idx: The index of the qubit that the preparation is acting on
    :rtype list: A list of programs
    """
    if op == 'X':
        gates = [RY(pi / 2, idx), RY(-pi / 2, idx)]
    elif op == 'Y':
        gates = [RX(-pi / 2, idx), RX(pi / 2, idx)]
    elif op == 'Z':
        gates = [I(idx), RX(pi, idx)]
    else:
        raise ValueError('Unknown gate operation')
    return gates


def random_local_pauli_eig_prep(prog, op, idx, random_seed=None):
    """
    Generate gate sequence to prepare a random local eigenstate of a Pauli operator, assuming
    we are starting from the ground state.

    :param Program prog: The `pyquil.quil.Program` object to which preparation pulses will be
     appended
    :param str op: Single character string representing the Pauli operator
     (one of 'I', 'X', 'Y', 'Z')
    :param int idx: index of Qubit the preparation acts on
    :param int random_seed: A seed to seed the RNG with.
    :return: A string description of the eigenstate prepared.
    """
    # TODO:
    #   + Return only the sign of the Pauli operator (more compact representation)
    #   + When given the identity, prepare random Pauli eigenstate
    #   + replace calls to random with sampling of random integers
    if random_seed is not None:
        seed(random_seed)
    if op == 'X':
        if random() > 0.5:
            gate = RY(pi / 2, idx)
            descr = '+X'
        else:
            gate = RY(-pi / 2, idx)
            descr = '-X'
    elif op == 'Y':
        if random() > 0.5:
            gate = RX(-pi / 2, idx)
            descr = '+Y'
        else:
            gate = RX(pi / 2, idx)
            descr = '-Y'
    elif op == 'Z':
        if random() > 0.5:
            gate = I(idx)
            descr = '+Z'
        else:
            gate = RX(pi, idx)
            descr = '-Z'
    else:
        raise ValueError('Unknown gate operation')
    prog.inst(gate)
    return descr


def local_pauli_eig_meas(prog, op, idx):
    """
    Generate gate sequence to measure in the eigenbasis of a Pauli operator, assuming
    we are only able to measure in the Z eigenbasis.
    """
    if op == 'X':
        gate = RY(-pi / 2, idx)
    elif op == 'Y':
        gate = RX(pi / 2, idx)
    elif op == 'Z':
        gate = I(idx)
    else:
        raise ValueError('Unknown gate operation')
    prog.inst(gate)
    return


def prepare_prod_pauli_eigenstate(pauli_term: PauliTerm):
    """Returns a circuit to prepare a +1 eigenstate of the Pauli operator described in PauliTerm.

    :param pauli_term: The PauliTerm whose eigenstate we will prepare.
    :return: A program corresponding to the correct rotation into the eigenbasis for pauli_term."""
    opset = pauli_term.operations_as_set()
    prog = Program()
    for (idx, op) in opset:
        local_pauli_eig_prep(prog, op, idx)
    return prog


def measure_prod_pauli_eigenstate(prog: Program, pauli_term: PauliTerm):
    opset = pauli_term.operations_as_set()
    for (idx, op) in opset:
        local_pauli_eig_meas(prog, op, idx)
    return prog


def prepare_random_prod_pauli_eigenstate(pauli_term: PauliTerm):
    opset = pauli_term.operations_as_set()
    prog = Program()
    s = ''.join([random_local_pauli_eig_prep(prog, op, idx) for (idx, op) in opset])
    return prog


def prepare_all_prod_pauli_eigenstates(pauli_term: PauliTerm):
    opset = pauli_term.operations_as_set()
    prod_preps = itertools.product(*[local_pauli_eigs_prep(op, idx) for (idx, op) in opset])
    return [Program().inst(list(prod)) for prod in prod_preps]


class OperatorBasis(object):
    """
    Encapsulate a complete set of basis operators.
    """

    def __init__(self, labels_ops):
        """
        Encapsulates a set of linearly independent operators.

        :param (list|tuple) labels_ops: Sequence of tuples (label, operator) where label is a string
            and operator is a numpy.ndarray/
        """
        self.ops_by_label = OrderedDict(labels_ops)
        self.labels = list(self.ops_by_label.keys())
        self.ops = list(self.ops_by_label.values())
        self.dim = len(self.ops)

    def product(self, *bases):
        """
        Compute the tensor product with another basis.

        :param bases: One or more additional bases to form the product with.
        :return (OperatorBasis): The tensor product basis as an OperatorBasis object.
        """
        if len(bases) > 1:
            basis_rest = bases[0].product(*bases[1:])
        else:
            assert len(bases) == 1
            basis_rest = bases[0]

        labels_ops = [(b1l + b2l, np.kron(b1, b2)) for (b1l, b1), (b2l, b2) in
                      itertools.product(self, basis_rest)]

        return OperatorBasis(labels_ops)

    def __iter__(self):
        """
        Iterate over tuples of (label, basis_op)

        :return: Yields the labels and qutip operators corresponding to the vectors in this basis.
        :rtype: tuple (str, qutip.qobj.Qobj)
        """
        for l, op in zip(self.labels, self.ops):
            yield l, op

    def __pow__(self, n):
        """
        Create the n-fold tensor product basis.

        :param int n: The number of identical tensor factors.
        :return: The product basis.
        :rtype: OperatorBasis
        """
        if not isinstance(n, int):
            raise TypeError("Can only accept an integer number of factors")
        if n < 1:
            raise ValueError("Need positive number of factors")
        if n == 1:
            return self
        return self.product(*([self] * (n - 1)))

    def __repr__(self):
        return "<span[{}]>".format(",".join(self.labels))


sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1.j], [1.j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

pauli_label_ops = [('I', np.eye(2)), ('X', sigma_x), ('Y', sigma_y), ('Z', sigma_z)]


def pauli_basis_measurements(qubit):
    """
    Generates the Programs required to measure the expectation values of the pauli operators.

    :param qubit: Required argument (so that the caller has a reference).
    """
    pauli_label_meas_progs = [Program(), Program(RY(-np.pi / 2, qubit)), Program(RX(-np.pi / 2, qubit)), Program()]
    return pauli_label_meas_progs


PAULI_BASIS = OperatorBasis(pauli_label_ops)


def n_qubit_pauli_basis(n):
    """
    Construct the tensor product operator basis of `n` PAULI_BASIS's.

    :param int n: The number of qubits.
    :return: The product Pauli operator basis of `n` qubits
    :rtype: OperatorBasis
    """
    if n >= 1:
        return PAULI_BASIS ** n
    else:
        raise ValueError("n = {} should be at least 1.".format(n))


def transform_pauli_moments_to_bit(mean_p, var_p):
    """
    Changes the first of a Pauli operator to the moments of a bit (a Bernoulli process).

    E.g. if the original mean is on [-1,+1] the returned mean is on [0,1].

    :param mean_p: mean of some Pauli operator
    :param var_p: variance of a Pauli operator
    :return: bit mean and variance.
    """
    mean_out = (mean_p + 1) / 2
    var_out = var_p / 4
    return mean_out, var_out


def transform_bit_moments_to_pauli(mean_c, var_c):
    """
    Changes the first two moments of a bit (a Bernoulli process) to Pauli operator moments.

    E.g. if the original mean is on [0,1] the returned mean is on [-1,+1].

    :param mean_c: bit probability of heads or tails.
    :param var_c: variance of bit
    :return: Pauli operator mean and variance.
    """
    mean_out = 2 * mean_c - 1
    var_out = 4 * var_c
    return mean_out, var_out


def partial_trace(rho, keep, dims, optimize=False):
    """Calculate the partial trace.

    Consider a joint state ρ on the Hilbert space H_a \\otimes H_b. We wish to trace over H_b e.g.

    ρ_a = Tr_b(ρ).

    :param rho: 2D array, the matrix to trace.
    :param keep: An array of indices of the spaces to keep after being traced. For instance,
                 if the space is A x B x C x D and we want to trace out B and D, keep = [0,2].
    :param dims: An array of the dimensions of each space. For example, if the space is
                 A x B x C x D, dims = [dim_A, dim_B, dim_C, dim_D].
    :return:  ρ_a, a 2D array i.e. the traced matrix
    """
    # Code from
    # https://scicomp.stackexchange.com/questions/30052/calculate-partial-trace-of-an-outer-product-in-python
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[keep])

    idx1 = [i for i in range(Ndim)]
    idx2 = [Ndim + i if i in keep else i for i in range(Ndim)]
    rho_a = rho.reshape(np.tile(dims, 2))
    rho_a = np.einsum(rho_a, idx1 + idx2, optimize=optimize)
    return rho_a.reshape(Nkeep, Nkeep)
