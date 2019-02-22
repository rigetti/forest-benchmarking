import itertools
from collections import OrderedDict
from random import random, seed
from typing import Sequence, List, Set, Tuple
from datetime import date, datetime
from git import Repo
import numpy as np
from numpy import pi
import networkx as nx
from networkx.algorithms.approximation.clique import clique_removal
import pandas as pd
from pandas import DataFrame

from pyquil.gates import I, RX, RY, RZ
from pyquil.paulis import PauliTerm
from pyquil.quil import Program
from pyquil.api import QuantumComputer


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


def int_to_bit_array(num: int, n_bits: int) -> Sequence[int]:
    """
    Converts a number into an array of bits where the right-most bit is least significant.

    :param num: the integer corresponding to the bitstring.
    :param n_bits: the number of bits to report
    :return:  an array of n_bits bits with right-most bit considered least significant.
    """
    return [num >> bit & 1 for bit in range(n_bits - 1, -1, -1)]


def determine_simultaneous_grouping(experiments: Sequence[DataFrame],
                                    equivalent_column_label: str = None) -> List[Set[int]]:
    """
    Determines a grouping of experiments acting on disjoint sets of qubits that can be run
    simultaneously.

    :param experiments:
    :return: a list of the simultaneous groups, each specified by a set of indices of each grouped
        experiment in experiments
    """
    g = nx.Graph()
    nodes = np.arange(len(experiments))
    g.add_nodes_from(nodes)
    qubits = [expt["Qubits"].values[0] for expt in experiments]

    need_equiv = None
    if equivalent_column_label is not None:
        need_equiv = [expt[equivalent_column_label].values for expt in experiments]

    for node1 in nodes:
        qbs1 = qubits[node1]
        for node2 in nodes[node1+1:]:
            if len(qbs1.intersection(qubits[node2])) == 0:
                # check that the requested columns are equivalent
                if equivalent_column_label is not None:
                    if not np.array_equal(need_equiv[node1], need_equiv[node2]):
                        continue
                # no shared qubits, and requested columns are identical, so add edge
                g.add_edge(node1, node2)

    # get the largest groups of nodes with shared edges, as each can be run simultaneously
    _, cliqs = clique_removal(g)

    return cliqs


def bloch_vector_to_standard_basis(theta: float, phi: float) -> Tuple[complex, complex]:
    """
    Converts the Bloch vector representation of a 1q state given in spherical coordinates to the
    standard representation of that state in the computational basis.

    :param theta: azimuthal angle given in radians
    :param phi: polar angle given in radians
    :return: tuple of the two coefficients a and b for the state a|0> + b|1> where a is real
    """
    return np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)


def standard_basis_to_bloch_vector(qubit_state: Sequence[complex]) -> Tuple[float, float]:
    """
    Converts a standard representation of a single qubit state in the computational basis to the
    spherical coordinates theta, phi of its representation on the Bloch sphere.

    :param qubit_state: a sequence of the two coefficients a and b for the state a|0> + b|1>
    :return: the azimuthal and polar angle, theta and phi, representing a point on the Bloch sphere.
    """
    alpha, beta = qubit_state
    phi = np.angle(beta)
    if alpha.imag != 0:
        # take out the global phase so that alpha is real
        phi -= np.angle(alpha)
        alpha = abs(alpha)
    theta = np.arccos(alpha)*2
    return theta, phi


def prepare_state_on_bloch_sphere(qubit: int, theta: float, phi: float):
    """
    Returns a program which prepares the given qubit in the state (theta, phi) on the bloch sphere,
    assuming the initial state |0> where (theta=0, phi=0).

    Theta and phi are the usual polar coordinates, given in radians. Theta is the angle of the
    state from the +Z axis, or zero state, and phi is the rotation angle from the XZ plane.
    Equivalently, the state
        alpha |0> + beta |1>
    in these coordinates has alpha = cos(theta/2) and e^(i*phi) = beta.imag, modulo some global
    phase.

    See https://en.wikipedia.org/wiki/Qubit#Bloch_sphere_representation for more information.

    :param qubit: the qubit to prepare in the given state
    :param theta: azimuthal angle given in radians
    :param phi: polar angle given in radians
    :return: a program preparing the qubit in the specified state, implemented in native gates
    """
    prep = Program()
    prep += RX(pi / 2, qubit)
    prep += RZ(theta, qubit)
    prep += RX(-pi / 2, qubit)
    prep += RZ(phi, qubit)
    return prep


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
        labels_list = [qubit for qubit in reversed(range(len(pauli_str)))]
    else:
        labels_list = sorted(qubit_labels)[::-1]
    pauli_term = PauliTerm.from_list(list(zip(pauli_str, labels_list)))
    return pauli_term


def local_sic_prep(label, qubit):
    """
    TODO: !
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


def local_pauli_eig_prep(op, qubit):
    """
    Generate gate sequence to prepare a the +1 eigenstate of a Pauli operator, assuming
    we are starting from the ground state ( the +1 eigenstate of Z^{\\otimes n})

    :param str op: A string representation of the Pauli operator whose eigenstate we'd like to prepare.
    :param int qubit: The index of the qubit that the preparation is acting on
    :return: The preparation Program.
    """
    if op == 'X':
        gate = RY(pi / 2, qubit)
    elif op == 'Y':
        gate = RX(-pi / 2, qubit)
    elif op == 'Z':
        gate = I(qubit)
    else:
        raise ValueError('Unknown gate operation')
    prog = Program(gate)
    return prog


def local_pauli_eigs_prep(op, qubit):
    """
    Generate all gate sequences to prepare all eigenstates of a (local) Pauli operator, assuming
    we are starting from the ground state.

    :param str op: A string representation of the Pauli operator whose eigenstate we'd like to prepare.
    :param int qubit: The index of the qubit that the preparation is acting on
    :rtype list: A list of programs
    """
    if op == 'X':
        gates = [RY(pi / 2, qubit), RY(-pi / 2, qubit)]
    elif op == 'Y':
        gates = [RX(-pi / 2, qubit), RX(pi / 2, qubit)]
    elif op == 'Z':
        gates = [I(qubit), RX(pi, qubit)]
    else:
        raise ValueError('Unknown gate operation')
    return [Program(gate) for gate in gates]


def random_local_pauli_eig_prep(prog, op, qubit, random_seed=None):
    """
    Generate gate sequence to prepare a random local eigenstate of a Pauli operator, assuming
    we are starting from the ground state.

    :param Program prog: The `pyquil.quil.Program` object to which preparation pulses will be
     appended
    :param str op: Single character string representing the Pauli operator
     (one of 'I', 'X', 'Y', 'Z')
    :param int qubit: index of Qubit the preparation acts on
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
            gate = RY(pi / 2, qubit)
            descr = '+X'
        else:
            gate = RY(-pi / 2, qubit)
            descr = '-X'
    elif op == 'Y':
        if random() > 0.5:
            gate = RX(-pi / 2, qubit)
            descr = '+Y'
        else:
            gate = RX(pi / 2, qubit)
            descr = '-Y'
    elif op == 'Z':
        if random() > 0.5:
            gate = I(qubit)
            descr = '+Z'
        else:
            gate = RX(pi, qubit)
            descr = '-Z'
    else:
        raise ValueError('Unknown gate operation')
    prog.inst(gate)
    return descr


def local_pauli_eig_meas(op, qubit):
    """
    Generate gate sequence to measure in the eigenbasis of a Pauli operator, assuming
    we are only able to measure in the Z eigenbasis.
    """
    if op == 'X':
        gate = RY(-pi / 2, qubit)
    elif op == 'Y':
        gate = RX(pi / 2, qubit)
    elif op == 'Z':
        gate = I(qubit)
    else:
        raise ValueError('Unknown gate operation')
    prog = Program(gate)
    return prog


def prepare_prod_pauli_eigenstate(pauli_term: PauliTerm):
    """Returns a circuit to prepare a +1 eigenstate of the Pauli operator described in PauliTerm.

    :param pauli_term: The PauliTerm whose eigenstate we will prepare.
    :return: A program corresponding to the correct rotation into the eigenbasis for pauli_term."""
    opset = pauli_term.operations_as_set()
    prog = Program()
    for (qubit, op) in opset:
        prog += local_pauli_eig_prep(op, qubit)
    return prog


def measure_prod_pauli_eigenstate(pauli_term: PauliTerm):
    opset = pauli_term.operations_as_set()
    prog = Program()
    for (qubit, op) in opset:
        prog += local_pauli_eig_meas(op, qubit)
    return prog


def prepare_random_prod_pauli_eigenstate(pauli_term: PauliTerm):
    opset = pauli_term.operations_as_set()
    prog = Program()
    s = ''.join([random_local_pauli_eig_prep(prog, op, qubit) for (qubit, op) in opset])
    return prog


def prepare_all_prod_pauli_eigenstates(pauli_term: PauliTerm):
    opset = pauli_term.operations_as_set()
    prod_preps = itertools.product(*[local_pauli_eigs_prep(op, qubit) for (qubit, op) in opset])
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
    r"""Calculate the partial trace.

    Consider a joint state ρ on the Hilbert space H_a \otimes H_b. We wish to trace over H_b e.g.

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


def metadata_save(qc: QuantumComputer,
                  repo_path: str = None,
                  filename: str = None) -> pd.DataFrame:
    '''
    This helper function saves metadata related to your run on a Quantum computer.

    Basic data saved includes the date and time. Additionally information related to the quantum
    computer is saved: device name, topology, qubit labels, edges that have two qubit gates,
    device specs

    If a path is passed to this function information related to the Git Repository is saved,
    specifically the repository: name, branch and commit.

    :param qc: The QuantumComputer to run the experiment on.
    :param repo_path: path to repository e.g. '../'
    :param filename: The name of the file to write JSON-serialized results to.
    :return: pandas DataFrame
    '''

    # Git related things
    if repo_path is not None:
        repo = Repo(repo_path)
        branch = repo.active_branch
        sha = repo.head.object.hexsha
        short_sha = repo.git.rev_parse(sha, short=7)
        the_repo = repo.git_dir
        the_branch = branch.name
        the_commit = short_sha
    else:
        the_repo = None
        the_branch = None
        the_commit = None

    metadata = {
        # Time and date
        'Date': str(date.today()),
        'Time': str(datetime.now().time()),
        # Git stuff
        'Repository': the_repo,
        'Branch': the_branch,
        'Git_commit': the_commit,
        # QPU data
        'Device_name': qc.name,
        'Topology': qc.qubit_topology(),
        'Qubits': list(qc.qubit_topology().nodes),
        'Two_Qubit_Gates': list(qc.qubit_topology().edges),
        'Device_Specs': qc.device.get_specs(),
    }
    if filename:
        pd.DataFrame(metadata).to_json(filename)
    return pd.DataFrame(metadata)
