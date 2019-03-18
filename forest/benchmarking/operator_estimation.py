"""
Utilities for estimating expected values of Pauli terms given pyquil programs
"""
from collections import namedtuple
from functools import reduce

import numpy as np
from pyquil.paulis import (PauliSum, PauliTerm, commuting_sets, sI,
                           term_with_coeff, is_identity)
from pyquil.quil import Program
from pyquil.gates import RX, RY, RZ, MEASURE
from forest.benchmarking.readout import estimate_confusion_matrix
from forest.benchmarking.compilation import basic_compile

STANDARD_NUMSHOTS = 10000

class CommutationError(ValueError):
    """
    Raised error when two items do not commute as promised
    """
    pass


def remove_identity(psum):
    """
    Remove the identity term from a Pauli sum

    :param PauliSum psum: PauliSum object to remove identity
    :return: The new pauli sum and the identity term.
    """
    new_psum = []
    identity_terms = []
    for term in psum:
        if not is_identity(term):
            new_psum.append(term)
        else:
            identity_terms.append(term)
    return sum(new_psum), sum(identity_terms)


def remove_imaginary(pauli_sums):
    """
    Remove the imaginary component of each term in a Pauli sum

    :param PauliSum pauli_sums: The Pauli sum to process.
    :return: a purely hermitian Pauli sum.
    :rtype: PauliSum
    """
    if not isinstance(pauli_sums, PauliSum):
        raise TypeError("not a pauli sum. please give me one")
    new_term = sI(0) * 0.0
    for term in pauli_sums:
        new_term += term_with_coeff(term, term.coefficient.real)

    return new_term


def get_confusion_matrices(quantum_resource, qubits, num_sample_ubound):
    """
    Get the confusion matrices from the quantum resource given number of samples

    This allows the user to change the accuracy at which they estimate the
    confusion matrix

    :param quantum_resource: Quantum Abstract Machine connection object
    :param qubits: qubits to measure 1-qubit readout confusion matrices
    :return: dictionary of confusion matrices indexed by the qubit label
    :rtype: dict
    """
    confusion_mat_dict = {q: estimate_confusion_matrix(quantum_resource, q, num_sample_ubound) for q in qubits}
    return confusion_mat_dict


def get_rotation_program(pauli_term):
    """
    Generate a rotation program so that the pauli term is diagonal

    :param PauliTerm pauli_term: The Pauli term used to generate diagonalizing
                                 one-qubit rotations.
    :return: The rotation program.
    :rtype: Program
    """
    meas_basis_change = Program()
    for index, gate in pauli_term:
        if gate == 'X':
            meas_basis_change.inst(RY(-np.pi / 2, index))
        elif gate == 'Y':
            meas_basis_change.inst(RX(np.pi / 2, index))
        elif gate == 'Z':
            pass
        else:
            raise ValueError()

    return meas_basis_change


def get_parity(pauli_terms, bitstring_results):
    """
    Calculate the eigenvalues of Pauli operators given results of projective measurements

    The single-qubit projective measurement results (elements of
    `bitstring_results`) are listed in physical-qubit-label numerical order.

    An example:

    Consider a Pauli term Z1 Z5 Z6 I7 and a collection of single-qubit
    measurement results corresponding to measurements in the z-basis on qubits
    {1, 5, 6, 7}. Each element of bitstring_results is an element of
    :math:`\{0, 1\}^{\otimes 4}`.  If [0, 0, 1, 0] and [1, 0, 1, 1]
    are the two projective measurement results in `bitstring_results` then
    this method returns a 1 x 2 numpy array with values [[-1, 1]]

    :param List pauli_terms: A list of Pauli terms operators to use
    :param bitstring_results: A list of projective measurement results.  Each
                              element is a list of single-qubit measurements.
    :return: Array (m x n) of {+1, -1} eigenvalues for the m-operators in
             `pauli_terms` associated with the n measurement results.
    :rtype: np.ndarray
    """
    qubit_set = []
    for term in pauli_terms:
        qubit_set.extend(list(term.get_qubits()))
    active_qubit_indices = sorted(list(set(qubit_set)))
    index_mapper = dict(zip(active_qubit_indices,
                            range(len(active_qubit_indices))))

    results = np.zeros((len(pauli_terms), len(bitstring_results)))

    # convert to array so we can fancy index into it later.
    # list() is needed to cast because we don't really want a map object
    bitstring_results = list(map(np.array, bitstring_results))
    for row_idx, term in enumerate(pauli_terms):
        memory_index = np.array(list(map(lambda x: index_mapper[x],
                                         sorted(term.get_qubits()))))

        results[row_idx, :] = [-2 * (sum(x[memory_index]) % 2) +
                               1 for x in bitstring_results]
    return results


EstimationResult = namedtuple('EstimationResult',
                              ('expected_value', 'pauli_expectations',
                               'covariance', 'variance', 'n_shots'))
EstimationResult.__doc__ = '''\
A namedtuple describing Monte-Carlo averaging results

:param expected_value: expected value of the PauliSum
:param pauli_expectations:  individual expected values of elements in the
                            PauliSum.  These values are scaled by the
                            coefficients associated with each PauliSum.
:param covariance: The covariance matrix computed from the shot data.
:param variance: Sample variance computed from the covariance matrix and
                 number of shots taken to obtain the data.
:param n_shots: Number of readouts collected.
'''


def estimate_pauli_sum(pauli_terms,
                       basis_transform_dict,
                       program,
                       variance_bound,
                       quantum_resource,
                       commutation_check=True,
                       symmetrize=True,
                       rand_samples=16):
    """
    Estimate the mean of a sum of pauli terms to set variance

    The sample variance is calculated by

    .. math::
        \begin{align}
        \mathrm{Var}[\hat{\langle H \rangle}] = \sum_{i, j}h_{i}h_{j}
        \mathrm{Cov}(\hat{\langle P_{i} \rangle}, \hat{\langle P_{j} \rangle})
        \end{align}

    The expectation value of each Pauli operator (term and coefficient) is
    also returned.  It can be accessed through the named-tuple field
    `pauli_expectations'.

    :param pauli_terms: list of pauli terms to measure simultaneously or a
                        PauliSum object
    :param basis_transform_dict: basis transform dictionary where the key is
                                 the qubit index and the value is the basis to
                                 rotate into. Valid basis is [I, X, Y, Z].
    :param program: program generating a state to sample from.  The program
                    is deep copied to ensure no mutation of gates or program
                    is perceived by the user.
    :param variance_bound:  Bound on the variance of the estimator for the
                            PauliSum. Remember this is the SQUARE of the
                            standard error!
    :param quantum_resource: quantum abstract machine object
    :param Bool commutation_check: Optional flag toggling a safety check
                                   ensuring all terms in `pauli_terms`
                                   commute with each other
    :param Bool symmetrize: Optional flag toggling symmetrization of readout
    :param Int rand_samples: number of random realizations for readout symmetrization
    :return: estimated expected value, expected value of each Pauli term in
             the sum, covariance matrix, variance of the estimator, and the
             number of shots taken.  The objected returned is a named tuple with
             field names as follows: expected_value, pauli_expectations,
             covariance, variance, n_shots.
             `expected_value' == coef_vec.dot(pauli_expectations)
    :rtype: EstimationResult
    """
    if not isinstance(pauli_terms, (list, PauliSum)):
        raise TypeError("pauli_terms needs to be a list or a PauliSum")

    if isinstance(pauli_terms, PauliSum):
        pauli_terms = pauli_terms.terms

    # check if each term commutes with everything
    if commutation_check:
        if len(commuting_sets(sum(pauli_terms))) != 1:
            raise CommutationError("Not all terms commute in the expected way")

    program = program.copy()
    pauli_for_rotations = PauliTerm.from_list(
        [(value, key) for key, value in basis_transform_dict.items()])

    program += get_rotation_program(pauli_for_rotations)

    qubits = sorted(list(basis_transform_dict.keys()))
    if symmetrize:
        theta = program.declare("ro_symmetrize", "REAL", len(qubits))
        for (idx, q) in enumerate(qubits):
            program += [RZ(np.pi/2, q), RY(theta[idx], q), RZ(-np.pi/2, q)]

    ro = program.declare("ro", "BIT", memory_size=len(qubits))
    for num, qubit in enumerate(qubits):
        program.inst(MEASURE(qubit, ro[num]))

    coeff_vec = np.array(
        list(map(lambda x: x.coefficient, pauli_terms))).reshape((-1, 1))

    # upper bound on samples given by IV of arXiv:1801.03524
    num_sample_ubound = 10 * int(np.ceil(np.sum(np.abs(coeff_vec))**2 / variance_bound))
    if num_sample_ubound <= 2:
        raise ValueError("Something happened with our calculation of the max sample")

    if symmetrize:
        if min(STANDARD_NUMSHOTS, num_sample_ubound)//rand_samples == 0:
            raise ValueError(f"The number of shots must be larger than {rand_samples}.")

        program = program.wrap_in_numshots_loop(min(STANDARD_NUMSHOTS, num_sample_ubound)//rand_samples)
    else:
        program = program.wrap_in_numshots_loop(min(STANDARD_NUMSHOTS, num_sample_ubound))

    binary = quantum_resource.compiler.native_quil_to_executable(basic_compile(program))

    results = None
    sample_variance = np.infty
    number_of_samples = 0
    tresults = np.zeros((0, len(qubits)))
    while (sample_variance > variance_bound and number_of_samples < num_sample_ubound):
        if symmetrize:
            # for some number of times sample random bit string
            for r in range(rand_samples):
                rand_flips = np.random.randint(low=0, high=2, size=len(qubits))
                temp_results = quantum_resource.run(binary, memory_map={'ro_symmetrize': np.pi * rand_flips})
                tresults = np.vstack((tresults, rand_flips ^ temp_results))
        else:
            tresults = quantum_resource.run(binary)

        number_of_samples += len(tresults)
        parity_results = get_parity(pauli_terms, tresults)

        # Note: easy improvement would be to update mean and variance on the fly
        # instead of storing all these results.
        if results is None:
            results = parity_results
        else:
            results = np.hstack((results, parity_results))

        # calculate the expected values....
        covariance_mat = np.cov(results, ddof=1)
        sample_variance = coeff_vec.T.dot(covariance_mat).dot(coeff_vec) / (results.shape[1] - 1)

    return EstimationResult(expected_value=coeff_vec.T.dot(np.mean(results, axis=1)),
                            pauli_expectations=np.multiply(coeff_vec.flatten(), np.mean(results, axis=1).flatten()),
                            covariance=covariance_mat,
                            variance=sample_variance,
                            n_shots=results.shape[1])


#########
#
# API
#
#########
def estimate_locally_commuting_operator(program,
                                        pauli_sum,
                                        variance_bound,
                                        quantum_resource,
                                        symmetrize=True):
    """
    Estimate the expected value of a Pauli sum to fixed precision.

    :param program: state preparation program
    :param pauli_sum: pauli sum of operators to estimate expected value
    :param variance_bound: variance bound on the estimator
    :param quantum_resource: quantum abstract machine object
    :param symmetrize: flag that determines whether readout is symmetrized or not
    :return: expected value, estimator variance, total number of experiments
    """
    pauli_sum, identity_term = remove_identity(pauli_sum)

    expected_value = 0
    if isinstance(identity_term, int):
        if np.isclose(identity_term, 0):
            expected_value = 0
        else:
            expected_value = identity_term

    elif isinstance(identity_term, (PauliTerm, PauliSum)):
        if isinstance(identity_term, PauliTerm):
            expected_value = identity_term.coefficient
        else:
            expected_value = identity_term[0].coefficient
    else:
        raise TypeError("identity_term must be a PauliTerm or integer. We got {}".format(type(identity_term)))

    # check if pauli_sum didn't get killed...or we gave an identity term
    if isinstance(pauli_sum, int):
        # we have no estimation work to do...just return the identity value
        return expected_value, 0, 0

    psets = commuting_sets_by_zbasis(pauli_sum)
    variance_bound_per_set = variance_bound / len(psets)

    total_shots = 0
    estimator_variance = 0
    for qubit_op_key, pset in psets.items():
        results = estimate_pauli_sum(pset,
                                     dict(qubit_op_key),
                                     program,
                                     variance_bound_per_set,
                                     quantum_resource,
                                     commutation_check=False,
                                     symmetrize=symmetrize)
        assert results.variance < variance_bound_per_set

        expected_value += results.expected_value
        total_shots += results.n_shots
        estimator_variance += results.variance

    return expected_value, estimator_variance, total_shots


def estimate_general_psum(program, pauli_sum, variance_bound, quantum_resource,
                          sequential=False):
    """
    Estimate the expected value of a Pauli sum to fixed precision.

    :param program: state preparation program
    :param pauli_sum: pauli sum of operators to estimate expected value
    :param variance_bound: variance bound on the estimator
    :param quantum_resource: quantum abstract machine object
    :return: expected value, estimator variance, total number of experiments
    """
    if sequential:
        expected_value = 0
        estimator_variance = 0
        total_shots = 0
        variance_bound_per_term = variance_bound / len(pauli_sum)
        for term in pauli_sum:
            exp_v, exp_var, exp_shots = estimate_locally_commuting_operator(
                program, PauliSum([term]), variance_bound_per_term, quantum_resource)
            expected_value += exp_v
            estimator_variance += exp_var
            total_shots += exp_shots
        return expected_value, estimator_variance, total_shots
    else:
        return estimate_locally_commuting_operator(program, pauli_sum,
                                                   variance_bound,
                                                   quantum_resource)


def estimate_general_psum_symmeterized(program, pauli_sum, variance_bound,
                                       quantum_resource, confusion_mat_dict=None,
                                       sequential=False):
    """
    Estimate the expected value of a Pauli sum to fixed precision.

    :param program: state preparation program
    :param pauli_sum: pauli sum of operators to estimate expected value
    :param variance_bound: variance bound on the estimator
    :param quantum_resource: quantum abstract machine object
    :return: expected value, estimator variance, total number of experiments
    """
    if sequential:
        expected_value = 0
        estimator_variance = 0
        total_shots = 0
        variance_bound_per_term = variance_bound / len(pauli_sum)
        for term in pauli_sum:
            exp_v, exp_var, exp_shots = \
                estimate_locally_commuting_operator_symmeterized(
                program, PauliSum([term]), variance_bound_per_term,
                quantum_resource, confusion_mat_dict=confusion_mat_dict)
            expected_value += exp_v
            estimator_variance += exp_var
            total_shots += exp_shots

        return expected_value, estimator_variance, total_shots

    else:
        return estimate_locally_commuting_operator_symmeterized(
            program, pauli_sum, variance_bound, quantum_resource,
            confusion_mat_dict=confusion_mat_dict)


def estimate_locally_commuting_operator_symmeterized(program, pauli_sum,
                                                     variance_bound,
                                                     quantum_resource,
                                                     confusion_mat_dict=None):
    """
    Estimate the expected value of a Pauli sum to fixed precision.

    Pauli sum can be a sum of non-commuting terms.  This routine groups the
    terms into sets of locally commuting operators.  This routine uses
    symmeterized readout.

    :param program: state preparation program
    :param pauli_sum: pauli sum of operators to estimate expected value
    :param variance_bound: variance bound on the estimator
    :param quantum_resource: quantum abstract machine object
    :return: expected value, estimator variance, total number of experiments
    """
    pauli_sum, identity_term = remove_identity(pauli_sum)

    if isinstance(identity_term, int):
        if np.isclose(identity_term, 0):
            expected_value = 0
        else:
            expected_value = identity_term

    elif isinstance(identity_term, (PauliTerm, PauliSum)):
        if isinstance(identity_term, PauliTerm):
            expected_value = identity_term.coefficient
        else:
            expected_value = identity_term[0].coefficient
    else:
        raise TypeError("identity_term must be a PauliTerm or integer. We got type {}".format(type(identity_term)))

    # check if pauli_sum didn't get killed...or we gave an identity term
    if isinstance(pauli_sum, int):
        # we have no estimation work to do...just return the identity value
        return expected_value, 0, 0

    psets = commuting_sets_by_zbasis(pauli_sum)
    variance_bound_per_set = variance_bound / len(psets)
    total_shots = 0
    estimator_variance = 0

    for qubit_op_key, pset in psets.items():
        results = estimate_general_psum_symmeterized(pset, dict(qubit_op_key), program,
                                                     variance_bound_per_set,
                                                     quantum_resource,
                                                     commutation_check=False,
                                                     confusion_mat_dict=confusion_mat_dict)

        assert results.variance < variance_bound_per_set
        expected_value += results.expected_value
        total_shots += results.n_shots
        estimator_variance += results.variance

    return expected_value, estimator_variance, total_shots


def diagonal_basis_commutes(pauli_a, pauli_b):
    """
    Test if `pauli_a` and `pauli_b` share a diagonal basis
    Example:
        Check if [A, B] with the constraint that A & B must share a one-qubit
        diagonalizing basis. If the inputs were [sZ(0), sZ(0) * sZ(1)] then this
        function would return True.  If the inputs were [sX(5), sZ(4)] this
        function would return True.  If the inputs were [sX(0), sY(0) * sZ(2)]
        this function would return False.
    :param pauli_a: Pauli term to check commutation against `pauli_b`
    :param pauli_b: Pauli term to check commutation against `pauli_a`
    :return: Boolean of commutation result
    :rtype: Bool
    """
    overlapping_active_qubits = set(pauli_a.get_qubits()) & set(pauli_b.get_qubits())
    for qubit_index in overlapping_active_qubits:
        if (pauli_a[qubit_index] != 'I' and pauli_b[qubit_index] != 'I' and
           pauli_a[qubit_index] != pauli_b[qubit_index]):
            return False

    return True


def get_diagonalizing_basis(list_of_pauli_terms):
    """
    Find the Pauli Term with the most non-identity terms
    :param list_of_pauli_terms: List of Pauli terms to check
    :return: The highest weight Pauli Term
    :rtype: PauliTerm
    """
    qubit_ops = set(reduce(lambda x, y: x + y,
                       [list(term._ops.items()) for term in list_of_pauli_terms]))
    qubit_ops = sorted(list(qubit_ops), key=lambda x: x[0])

    return PauliTerm.from_list(list(map(lambda x: tuple(reversed(x)), qubit_ops)))


def _max_key_overlap(pauli_term, diagonal_sets):
    """
    Calculate the max overlap of a pauli term ID with keys of diagonal_sets
    Returns a different key if we find any collisions.  If no collisions is
    found then the pauli term is added and the key is updated so it has the
    largest weight.
    :param pauli_term:
    :param diagonal_sets:
    :return: dictionary where key value pair is tuple indicating diagonal basis
             and list of PauliTerms that share that basis
    :rtype: dict
    """
    # a lot of the ugliness comes from the fact that
    # list(PauliTerm._ops.items()) is not the appropriate input for
    # Pauliterm.from_list()
    for key in list(diagonal_sets.keys()):
        pauli_from_key = PauliTerm.from_list(
            list(map(lambda x: tuple(reversed(x)), key)))
        if diagonal_basis_commutes(pauli_term, pauli_from_key):
            updated_pauli_set = diagonal_sets[key] + [pauli_term]
            diagonalizing_term = get_diagonalizing_basis(updated_pauli_set)
            if len(diagonalizing_term) > len(key):
                del diagonal_sets[key]
                new_key = tuple(sorted(diagonalizing_term._ops.items(),
                                       key=lambda x: x[0]))
                diagonal_sets[new_key] = updated_pauli_set
            else:
                diagonal_sets[key] = updated_pauli_set
            return diagonal_sets
    # made it through all keys and sets so need to make a new set
    else:
        # always need to sort because new pauli term functionality
        new_key = tuple(sorted(pauli_term._ops.items(), key=lambda x: x[0]))
        diagonal_sets[new_key] = [pauli_term]
        return diagonal_sets


def commuting_sets_by_zbasis(pauli_sums):
    """
    Computes commuting sets based on terms having the same diagonal basis
    Following the technique outlined in the appendix of arXiv:1704.05018.
    :param pauli_sums: PauliSum object to group
    :return: dictionary where key value pair is a tuple corresponding to the
             basis and a list of PauliTerms associated with that basis.
    """
    diagonal_sets = {}
    for term in pauli_sums:
        diagonal_sets = _max_key_overlap(term, diagonal_sets)

    return diagonal_sets
