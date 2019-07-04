import warnings
from math import pi, log
from typing import List, Tuple
import itertools
from pyquil.quil import Program
import numpy as np


def _flip_array_to_prog(flip_array: Tuple[bool], qubits: List[int]) -> Program:
    """
    Generate a pre-measurement program that flips the qubit state according to the flip_array of
    bools.

    This is used, for example, in symmetrization to produce programs which flip a select subset
    of qubits immediately before measurement.

    :param flip_array: tuple of booleans specifying whether the qubit in the corresponding index
        should be flipped or not.
    :param qubits: list specifying the qubits in order corresponding to the flip_array
    :return: Program which flips each qubit (i.e. instructs RX(pi, q)) according to the flip_array.
    """
    assert len(flip_array) == len(qubits), "Mismatch of qubits and operations"
    prog = Program()
    for qubit, flip_output in zip(qubits, flip_array):
        if flip_output == 0:
            continue
        elif flip_output == 1:
            prog += Program(RX(pi, qubit))
        else:
            raise ValueError("flip_bools should only consist of 0s and/or 1s")
    return prog


def _symmetrization(program: Program, meas_qubits: List[int], symm_type: int = 3) \
        -> Tuple[List[Program], List[np.ndarray]]:
    """
    For the input program generate new programs which flip the measured qubits with an X gate in
    certain combinations in order to symmetrize readout.

    An expanded list of programs is returned along with a list of bools which indicates which
    qubits are flipped in each program.

    The symmetrization types are specified by an int; the types available are:
    -1 -- exhaustive symmetrization uses every possible combination of flips
     0 -- trivial that is no symmetrization
     1 -- symmetrization using an OA with strength 1
     2 -- symmetrization using an OA with strength 2
     3 -- symmetrization using an OA with strength 3
    In the context of readout symmetrization the strength of the orthogonal array enforces the
    symmetry of the marginal confusion matrices.

    By default a strength 3 OA is used; this ensures expectations of the form <b_k * b_j * b_i>
    for bits any bits i,j,k will have symmetric readout errors. Here expectation of a random
    variable x as is denote <x> = sum_i Pr(i) x_i. It turns out that a strength 3 OA is also a
    strength 2 and strength 1 OA it also ensures <b_j * b_i> and <b_i> have symmetric readout
    errors for any bits b_j and b_i.

    :param programs: a program which will be symmetrized.
    :param meas_qubits: the groups of measurement qubits. Only these qubits will be symmetrized
        over, even if the program acts on other qubits.
    :param sym_type: an int determining the type of symmetrization performed.
    :return: a list of symmetrized programs, the corresponding array of bools indicating which
        qubits were flipped.
    """
    if symm_type < -1 or symm_type > 3:
        raise ValueError("symm_type must be one of the following ints [-1, 0, 1, 2, 3].")
    elif symm_type == -1:
        # exhaustive = all possible binary strings
        flip_matrix = np.asarray(list(itertools.product([0, 1], repeat=len(meas_qubits))))
    elif symm_type >= 0:
        flip_matrix = _construct_orthogonal_array(len(meas_qubits), symm_type)

    # The next part is not rigorous the sense that we simply truncate to the desired
    # number of qubits. The problem is that orthogonal arrays of a certain strength for an
    # arbitrary number of qubits are not known to exist.
    num_expts, num_qubits = flip_matrix.shape
    if len(meas_qubits) != num_qubits:
        flip_matrix = flip_matrix[0:num_expts, 0:len(meas_qubits)]

    symm_programs = []
    flip_arrays = []
    for flip_array in flip_matrix:
        total_prog_symm = program.copy()
        prog_symm = _flip_array_to_prog(flip_array, meas_qubits)
        total_prog_symm += prog_symm
        symm_programs.append(total_prog_symm)
        flip_arrays.append(flip_array)

    return symm_programs, flip_arrays


def _consolidate_symmetrization_outputs(outputs: List[np.ndarray],
                                        flip_arrays: List[Tuple[bool]]) -> np.ndarray:
    """
    Given bitarray results from a series of symmetrization programs, appropriately flip output
    bits and consolidate results into new bitarrays.

    :param outputs: a list of the raw bitarrays resulting from running a list of symmetrized
        programs; for example, the results returned from _measure_bitstrings
    :param flip_arrays: a list of boolean arrays in one-to-one correspondence with the list of
        outputs indicating which qubits where flipped before each bitarray was measured.
    :return: an np.ndarray consisting of the consolidated bitarray outputs which can be treated as
        the symmetrized outputs of the original programs passed into a symmetrization method. See
        estimate_observables for example usage.
    """
    assert len(outputs) == len(flip_arrays)

    output = []
    for bitarray, flip_array in zip(outputs, flip_arrays):
        if len(flip_array) == 0:
            output.append(bitarray)
        else:
            output.append(bitarray ^ flip_array)

    return np.vstack(output)


def _measure_bitstrings(qc, programs: List[Program], meas_qubits: List[int],
                        num_shots: int = 600) -> List[np.ndarray]:
    """
    Wrapper for appending measure instructions onto each program, running the program,
    and accumulating the resulting bitarrays.

    :param qc: a quantum computer object on which to run each program
    :param programs: a list of programs to run
    :param meas_qubits: groups of qubits to measure for each program
    :param num_shots: the number of shots to run for each program
    :return: a len(programs) long list of num_shots by num_meas_qubits bit arrays of results for
        each program.
    """
    results = []
    for program in programs:
        # copy the program so the original is not mutated
        prog = program.copy()
        ro = prog.declare('ro', 'BIT', len(meas_qubits))
        for idx, q in enumerate(meas_qubits):
            prog += MEASURE(q, ro[idx])

        prog.wrap_in_numshots_loop(num_shots)
        prog = qc.compiler.quil_to_native_quil(prog)
        exe = qc.compiler.native_quil_to_executable(prog)
        shots = qc.run(exe)
        results.append(shots)
    return results


def _construct_orthogonal_array(num_qubits: int, strength: int = 3) -> np.ndarray:
    """
    Given a strength and number of qubits this function returns an Orthogonal Array (OA)
    on 'n' or more qubits. Sometimes the size of the returned array is larger than num_qubits;
    typically the next power of two relative to num_qubits. This is corrected later in the code
    flow.

    :param num_qubits: the minimum number of qubits the OA should act on.
    :param strength: the statistical "strength" of the OA
    :return: a numpy array where the rows represent the different experiments
    """
    if strength < 0 or strength > 3:
        raise ValueError("'strength' must be one of the following ints [0, 1, 2, 3].")
    if strength == 0:
        # trivial flip matrix = an array of zeros
        flip_matrix = np.zeros((1, num_qubits)).astype(int)
    elif strength == 1:
        # orthogonal array with strength equal to 1. See Example 1.4 of [OATA], referenced in the
        # `construct_strength_two_orthogonal_array` docstrings, for more details.
        zero_array = np.zeros((1, num_qubits))
        one_array = np.ones((1, num_qubits))
        flip_matrix = np.concatenate((zero_array, one_array), axis=0).astype(int)
    elif strength == 2:
        flip_matrix = _construct_strength_two_orthogonal_array(num_qubits)
    elif strength == 3:
        flip_matrix = _construct_strength_three_orthogonal_array(num_qubits)

    return flip_matrix


def _next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


# The code below is directly copied from scipy see https://bit.ly/2RjAHJz, the docstrings have
# been modified.
def hadamard(n, dtype=int):
    """
    Construct a Hadamard matrix.
    Constructs an n-by-n Hadamard matrix, using Sylvester's
    construction.  `n` must be a power of 2.
    Parameters
    ----------
    n : int
        The order of the matrix.  `n` must be a power of 2.
    dtype : numpy dtype
        The data type of the array to be constructed.
    Returns
    -------
    H : (n, n) ndarray
        The Hadamard matrix.
    Notes
    -----
    .. versionadded:: 0.8.0
    Examples
    --------
    >>> hadamard(2, dtype=complex)
    array([[ 1.+0.j,  1.+0.j],
           [ 1.+0.j, -1.-0.j]])
    >>> hadamard(4)
    array([[ 1,  1,  1,  1],
           [ 1, -1,  1, -1],
           [ 1,  1, -1, -1],
           [ 1, -1, -1,  1]])
    """
    if n < 1:
        lg2 = 0
    else:
        lg2 = int(log(n, 2))
    if 2 ** lg2 != n:
        raise ValueError("n must be an positive integer, and n must be "
                         "a power of 2")

    H = np.array([[1]], dtype=dtype)

    # Sylvester's construction
    for i in range(0, lg2):
        H = np.vstack((np.hstack((H, H)), np.hstack((H, -H))))

    return H


def _construct_strength_three_orthogonal_array(num_qubits: int) -> np.ndarray:
    r"""
    Given a number of qubits this function returns an Orthogonal Array (OA)
    on 'n' qubits where n is the next power of two relative to num_qubits.

    Specifically it returns the OA(2n, n, 2, 3).

    The parameters of the OA(N, k, s, t) are interpreted as
    N: Number of rows, level combinations or runs
    k: Number of columns, constraints or factors
    s: Number of symbols or levels
    t: Strength

    See [OATA] for more details.

    [OATA] Orthogonal Arrays: theory and applications
           Hedayat, Sloane, Stufken
           Springer Science & Business Media, 2012.
           https://dx.doi.org/10.1007/978-1-4612-1478-6

    :param num_qubits: minimum number of qubits the OA should run on.
    :return: A numpy array representing the OA with shape N by k
    """
    num_qubits_power_of_2 = _next_power_of_2(num_qubits)
    H = hadamard(num_qubits_power_of_2)
    Hfold = np.concatenate((H, -H), axis=0)
    orthogonal_array = ((Hfold + 1) / 2).astype(int)
    return orthogonal_array


def _construct_strength_two_orthogonal_array(num_qubits: int) -> np.ndarray:
    r"""
    Given a number of qubits this function returns an Orthogonal Array (OA) on 'n-1' qubits
    where n-1 is the next integer lambda so that 4*lambda -1 is larger than num_qubits.

    Specifically it returns the OA(n, n − 1, 2, 2).

    The parameters of the OA(N, k, s, t) are interpreted as
    N: Number of rows, level combinations or runs
    k: Number of columns, constraints or factors
    s: Number of symbols or levels
    t: Strength

    See [OATA] for more details.

    [OATA] Orthogonal Arrays: theory and applications
           Hedayat, Sloane, Stufken
           Springer Science & Business Media, 2012.
           https://dx.doi.org/10.1007/978-1-4612-1478-6

    :param num_qubits: minimum number of qubits the OA should run on.
    :return: A numpy array representing the OA with shape N by k
    """
    # next line will break post denali at 275 qubits
    # valid_num_qubits = 4 * lambda - 1
    valid_numbers = [4 * lam - 1 for lam in range(1, 70)]
    # 4 * lambda
    four_lam = min(x for x in valid_numbers if x >= num_qubits) + 1
    H = hadamard(_next_power_of_2(four_lam))
    # The minus sign in front of H fixes the 0 <-> 1 inversion relative to the reference [OATA]
    orthogonal_array = ((-H[1:four_lam, 0:four_lam] + 1) / 2).astype(int)
    return orthogonal_array.T


def _check_min_num_trials_for_symmetrized_readout(num_qubits: int, trials: int, symm_type: int) \
        -> int:
    """
    This function sets the minimum number of trials; it is desirable to have hundreds or
    thousands of trials more than the minimum.

    :param num_qubits: number of qubits to symmetrize
    :param trials: number of trials
    :param symm_type: symmetrization type see
    :return: possibly modified number of trials
    """
    if symm_type < -1 or symm_type > 3:
        raise ValueError("symm_type must be one of the following ints [-1, 0, 1, 2, 3].")

    if symm_type == -1:
        min_num_trials = 2 ** num_qubits
    elif symm_type == 2:
        def _f(x):
            return 4 * x - 1
        min_num_trials = min(_f(x) for x in range(1, 1024) if _f(x) >= num_qubits) + 1
    elif symm_type == 3:
        min_num_trials = _next_power_of_2(2 * num_qubits)
    else:
        # symm_type == 0 or symm_type == 1 require one and two trials respectively; ensured by:
        min_num_trials = 2

    if trials < min_num_trials:
        trials = min_num_trials
        warnings.warn(f"Number of trials was too low, it is now {trials}.")
    return trials
