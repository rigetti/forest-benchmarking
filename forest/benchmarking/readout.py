import itertools
from math import pi
from typing import Tuple, Dict, Sequence
import numpy as np
from tqdm import tqdm

from pyquil import Program
from pyquil.api import QuantumComputer
from pyquil.gates import RX, RZ, RESET, MEASURE
from pyquil.quilbase import Measurement, Pragma

from forest.benchmarking.utils import bitstring_prep, parameterized_bitstring_prep


def get_flipped_program(program: Program) -> Program:
    """For symmetrization, generate a program where X gates are added before measurement."""
    flipped_prog = Program()
    for inst in program:
        if isinstance(inst, Measurement):
            flipped_prog += Pragma('PRESERVE_BLOCK')
            flipped_prog += RX(pi, inst.qubit)
            flipped_prog += Measurement(qubit=inst.qubit, classical_reg=inst.classical_reg)
            flipped_prog += Pragma('END_PRESERVE_BLOCK')
        else:
            flipped_prog.inst(inst)

    return flipped_prog


def estimate_confusion_matrix(qc: QuantumComputer, qubit: int, num_shots: int = 10000) \
        -> np.ndarray:
    """
    Estimate the readout confusion matrix for a given qubit.

    The confusion matrix will be of the form::

        [[ p(0 | 0)     p(0 | 1) ]
         [ p(1 | 0)     p(1 | 1) ]]

    where each column sums to one. Note that the matrix need not be symmetric.

    :param qc: The quantum computer to estimate the confusion matrix.
    :param qubit: The actual physical qubit to measure
    :param num_shots: The number of shots to take. This function runs two programs, so
        the total number of shots taken will be twice this number.
    :return: a 2x2 confusion matrix for the qubit, where each row sums to one.
    """
    # prepare 0 (do nothing) and measure; repeat shots number of times
    zero_meas = Program()
    ro_zero = zero_meas.declare("ro", "BIT", 1)
    zero_meas += MEASURE(qubit, ro_zero[0])
    zero_meas.wrap_in_numshots_loop(num_shots)
    should_be_0 = qc.run(qc.compile(zero_meas)).get_register_map().get('ro')

    # prepare one and measure; repeat shots number of times
    one_meas = Program()
    one_meas += RX(pi, qubit)
    ro_one = one_meas.declare("ro", "BIT", 1)
    one_meas += MEASURE(qubit, ro_one[0])
    one_meas.wrap_in_numshots_loop(num_shots)
    should_be_1 = qc.run(qc.compile(one_meas)).get_register_map().get('ro')

    p00 = 1 - np.mean(should_be_0)
    p11 = np.mean(should_be_1)

    return np.array([[p00, 1 - p00], [1 - p11, p11]])


def estimate_joint_confusion_in_set(qc: QuantumComputer, qubits: Sequence[int] = None,
                                    num_shots: int = 1000, joint_group_size: int = 1,
                                    use_param_program: bool = True, use_active_reset=False,
                                    show_progress_bar: bool = False) \
                                    -> Dict[Tuple[int, ...], np.ndarray]:
    """
    Measures the joint readout confusion matrix for all groups of size group_size among the qubits.

    Simultaneous readout of multiple qubits may exhibit correlations in readout error.
    Measuring a single qubit confusion for each qubit of interest is therefore not sufficient to
    completely characterize readout error in general. This method estimates joint readout confusion
    matrices for groups of qubits, which involves preparing all 2^joint_group_size possible
    bitstrings on each of (len(qubits) choose joint_group_size) many groups. The joint_group_size
    specifies which order of correlation you wish to characterize, and a separate confusion
    matrix is estimated for every possible group of that size.

    For example, a joint_group_size of one will simply estimate a single-qubit 2x2 confusion
    matrix for each of the provided qubits, similar to len(qubits) calls to
    estimate_confusion_matrix. Meanwhile, a joint_group_size of two will yield estimates of the
    (len(qubits) choose 2) many 4x4 confusion matrices for each possible pair of qubits. When the
    joint_group_size is the same as len(qubits) a single confusion matrix is estimated,
    which requires 2^len(qubits) number of state preparations. Note that the maximum number of
    measurements occurs when 1 << joint_group_size << len(qubits), and this maximum is quite large.

    Because this method performs the measurement of exponentially many bitstrings on a particular
    group of qubits, use_param_program should result in an appreciable speed up for estimation
    of each matrix by looping over bitstrings at a lower level of the stack, rather than creating
    a new program for each bitstring. Use_active_reset should also speed up estimation, at the cost
    of introducing more error and potentially complicating interpretation of results.

    :param qc: a quantum computer whose readout error you wish to characterize
    :param qubits: a list of accessible qubits on the qc you wish to characterize. Defaults to
        all qubits in qc.
    :param num_shots: number of shots in measurement of each bit string on each joint group of
        qubits.
    :param joint_group_size: the size of each group; a joint confusion matrix with
        2^joint_group_size number of rows/columns will be estimated for each group of qubits of
        the given size within the qubit set.
    :param use_param_program: dictates whether to use a parameterized program to measure out
        each bitstring. If set to default of True, this routine should execute faster on a
        QPU. Note that the parameterized option does not execute a no-op when measuring 0.
    :param use_active_reset: if true, all qubits in qc (not just provided qubits) will be actively
        reset to 0 state at the start of each bitstring measurement. This option is intended as a
        potential speed up, but may complicate interpretation of the estimated confusion
        matrices. The method estimate_joint_active_reset_confusion separately characterizes
        active reset.
    :param show_progress_bar: displays a progress bar via tqdm if true.
    :return: a dictionary whose keys are all possible joint_group_sized tuples that can be
        formed from the qubits. Each value is an estimated 2^group_size square confusion matrix
        for the corresponding tuple of qubits. Each key is listed in order of increasing qubit
        number. The corresponding matrix has rows and columns indexed in increasing bitstring
        order, with most significant (leftmost) bit labeling the smallest qubit number.
    """
    # establish default as all operational qubits in qc
    if qubits is None:
        qubits = qc.qubits()

    qubits = sorted(qubits)

    groups = list(itertools.combinations(qubits, joint_group_size))
    confusion_matrices = {}

    total_num_rounds = len(groups) * 2**joint_group_size
    with tqdm(total=total_num_rounds, disable=not show_progress_bar) as pbar:
        for group in groups:

            # initialize program to optionally actively reset. Active reset will provide a speed up
            # but may make the estimate harder to interpret due to errors in the reset.
            program_start = Program()
            if use_active_reset:
                program_start += RESET()

            executable = None  # will be re-assigned to either compiled param or non-param program

            reg_name = 'bitstr'
            # generate one parameterized program now, or later create a program for each bitstring
            if use_param_program:
                # generate parameterized program for this group and append to start
                prep_and_meas = parameterized_bitstring_prep(group, reg_name, append_measure=True)
                param_program = program_start + prep_and_meas
                param_program.wrap_in_numshots_loop(shots=num_shots)
                executable = qc.compiler.native_quil_to_executable(param_program)

            matrix = np.zeros((2 ** joint_group_size, 2 ** joint_group_size))
            for row, bitstring in enumerate(itertools.product([0, 1], repeat=joint_group_size)):

                memory_map = {}
                if use_param_program:
                    # specify bitstring in parameterization at run-time
                    memory_map[reg_name] = [float(b) for b in bitstring]

                else:
                    # generate program that measures given bitstring on group, and append to start
                    bitstring_program = program_start + bitstring_prep(group, bitstring,
                                                                       append_measure=True)
                    bitstring_program.wrap_in_numshots_loop(shots=num_shots)
                    executable = qc.compiler.native_quil_to_executable(bitstring_program)

                # update confusion matrix
                results = qc.run(executable, memory_map=memory_map).get_register_map().get('ro')
                for result in results:
                    base = np.array([2 ** i for i in reversed(range(joint_group_size))])
                    observed = np.sum(base * result)
                    matrix[row, observed] += 1 / num_shots

                # update the progress bar
                pbar.update(1)

            # store completed confusion matrix in dictionary
            confusion_matrices[group] = matrix

    return confusion_matrices


def marginalize_confusion_matrix(confusion_matrix: np.ndarray, all_qubits: Sequence[int],
                                 marginal_subset: Tuple[int, ...]) -> np.ndarray:
    """
    Marginalize a confusion matrix to get a confusion matrix on only the marginal_subset.

    Taking a joint confusion matrix (see estimate_joint_confusion_in_set) on all_qubits and
    marginalizing out some qubits results in another joint confusion matrix on only the
    marginal_subset. Comparing this marginal confusion matrix to a direct estimate of the joint
    confusion matrix on only the marginal_subset provides information about correlations between
    the marginal subset and the remaining qubits in all_qubits. For example, if each qubit is
    independent, we expect that the joint matrix on all_qubits is a tensor product of the
    matrices for each individual qubit. In this case, any marginalized single qubit confusion
    matrix for a given qubit should be identical to the directly estimated single qubit confusion
    matrix for that same qubit (up to estimation error).

    :param confusion_matrix: a confusion matrix for a group of qubits.
    :param all_qubits: the sequence of qubit labels corresponding to the confusion matrix. Qubits
        should be listed such that the most significant bit labels the first qubit in the sequence.
    :param marginal_subset: a subset of labels in all_qubits. The subset may be provided in any
        order, but the returned confusion matrix order will correspond to the order of subset
        elements as they appear in all_qubits
    :return: a joint confusion matrix for the qubits in the marginal subset
    """
    # Collect the indices of the marginal_subset elements within the larger all_qubits sequence
    all_indices = np.arange(len(all_qubits))
    is_index_of_subset_elem = np.isin(all_qubits, marginal_subset)
    subset_indices = np.compress(is_index_of_subset_elem, all_indices)
    # All elements of subset should be found somewhere in the larger all_qubits sequence
    assert len(subset_indices) == len(marginal_subset)

    # Reshape the 2^len(all_qubits) x 2^len(all_qubits) matrix into a tensor with 2*len(all_qubits)
    # axes each with dim = 2 = |{0,1}|
    # Each axis corresponds to a qubit; the first len(all_qubits) axes are qubits labeling the rows
    # of the original matrix, and the last len(all_qubits) axes are the column qubits.
    # Indexing within a given axis corresponds to a choice of the value of that qubit (0 or 1).
    reshaped = confusion_matrix.reshape([2] * (2 * len(all_qubits)))

    # Keep axes corresponding to the previously collected indices; there are two axes per indexed
    # qubit--row and column. Qubit with index n in the all_qubits sequence matches the n-th (row)
    # and [length(all_qubits)+n]-th (column) axes.
    axes_labels = np.arange(2 * len(all_qubits), dtype=int)
    keep_axes = np.concatenate([subset_indices, len(all_qubits) + subset_indices])

    # Now simply sum over the axes we are not keeping. (Einsum requires native python int type)
    marginal = np.einsum(reshaped, [index.item() for index in axes_labels],
                         [index.item() for index in keep_axes])

    # Reshape back into square matrix and re-normalize.
    renormalization_factor = 2 ** (len(all_qubits) - len(marginal_subset))
    dimension = 2 ** len(marginal_subset)
    return marginal.reshape(dimension, dimension) / renormalization_factor


def estimate_joint_reset_confusion(qc: QuantumComputer, qubits: Sequence[int] = None,
                                   num_trials: int = 10, joint_group_size: int = 1,
                                   use_active_reset: bool = True, show_progress_bar: bool = False) \
                                    -> Dict[Tuple[int, ...], np.ndarray]:
    """
    Measures a reset 'confusion matrix' for all groups of size joint_group_size among the qubits.

    Specifically, for each possible joint_group_sized group among the qubits we perform a
    measurement for each bitstring on that group. The measurement proceeds as follows:

        -Repeatedly try to prepare the bitstring state on the qubits until success.
        -If use_active_reset is true (default) actively reset qubits to ground state; otherwise,
        wait the preset amount of time for qubits to decay to ground state.
        -Measure the state after the chosen reset method.

    Since reset should result in the all zeros state this 'confusion matrix' should ideally have
    all ones down the left-most column of the matrix. The entry at (row_idx, 0) thus represents
    the success probability of reset given that the pre-reset state is the binary representation
    of the number row_idx. WARNING: this method can be very slow

    :param qc: a quantum computer whose reset error you wish to characterize
    :param qubits: a list of accessible qubits on the qc you wish to characterize. Defaults to
        all qubits in qc.
    :param num_trials: number of repeated trials of reset after preparation of each bitstring on
        each joint group of qubits. Note: num_trials does not correspond to num_shots; a new
        program must be run in each trial, so running a group of n trials takes longer than would
        collecting the same number of shots output from a single program.
    :param joint_group_size: the size of each group; a square matrix with 2^joint_group_size
        number of rows/columns will be estimated for each group of qubits of the given size
        among the provided qubits.
    :param use_active_reset: dictates whether to actively reset qubits or else wait the
        pre-allotted amount of time for the qubits to decay to the ground state. Using active
        reset will allow for faster data collection.
    :param show_progress_bar: displays a progress bar via tqdm if true.
    :return: a dictionary whose keys are all possible joint_group_sized tuples that can be
        formed from the qubits. Each value is an estimated 2^group_size square matrix
        for the corresponding tuple of qubits. Each key is listed in order of increasing qubit
        number. The corresponding matrix has rows and columns indexed in increasing bitstring
        order, with most significant (leftmost) bit labeling the smallest qubit number. Each
        matrix row corresponds to the bitstring that was prepared before the reset, and each
        column corresponds to the bitstring measured after the reset.
    """
    # establish default as all operational qubits in qc
    if qubits is None:
        qubits = qc.qubits()

    qubits = sorted(qubits)

    groups = list(itertools.combinations(qubits, joint_group_size))
    confusion_matrices = {}

    total_num_rounds = len(groups) * 2**joint_group_size
    with tqdm(total=total_num_rounds, disable=not show_progress_bar) as pbar:
        for group in groups:
            reg_name = 'bitstr'
            # program prepares a bit string (specified by parameterization) and immediately measures
            prep_and_meas = parameterized_bitstring_prep(group, reg_name, append_measure=True)
            prep_executable = qc.compiler.native_quil_to_executable(prep_and_meas)

            matrix = np.zeros((2 ** joint_group_size, 2 ** joint_group_size))
            for row, bitstring in enumerate(itertools.product([0, 1], repeat=joint_group_size)):
                for _ in range(num_trials):

                    # try preparation at most 10 times.
                    for _ in range(10):
                        # prepare the given bitstring and measure
                        memory_map = {reg_name: [float(b) for b in bitstring]}
                        result = qc.run(prep_executable, memory_map).get_register_map().get('ro')

                        # if the preparation is successful, move on to reset.
                        if np.array_equal(result[0], bitstring):
                            break

                    # execute program that measures the post-reset state
                    if use_active_reset:
                        # program runs immediately after prep program and actively resets qubits
                        reset_measure_program = Program(RESET())
                    else:
                        # this program automatically waits pre-allotted time for qubits to decay
                        reset_measure_program = Program()
                    ro = reset_measure_program.declare('ro', memory_type='BIT',
                                                       memory_size=len(qubits))
                    for idx, qubit in enumerate(group):
                        reset_measure_program += MEASURE(qubit, ro[idx])
                    executable = qc.compiler.native_quil_to_executable(reset_measure_program)
                    results = qc.run(executable).get_register_map().get('ro')

                    # update confusion matrix
                    for result in results:
                        base = np.array([2 ** i for i in reversed(range(joint_group_size))])
                        observed = np.sum(base * result)
                        matrix[row, observed] += 1 / num_trials

                # update the progress bar
                pbar.update(1)

            # store completed confusion matrix in dictionary
            confusion_matrices[group] = matrix

    return confusion_matrices
