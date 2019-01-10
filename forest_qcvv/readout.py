import itertools
from math import pi
from typing import List, Tuple, Dict

import numpy as np
from pyquil import Program
from pyquil.api import QuantumComputer
from pyquil.gates import RX, RZ
from pyquil.quilbase import Measurement, Pragma


def get_flipped_program(program: Program):
    """For symmetrization, generate a program where X gates are added before measurement."""
    flipped_prog = Program()
    for inst in program:
        if isinstance(inst, Measurement):
            flipped_prog += Pragma('PRESERVE_BLOCK')
            flipped_prog.inst(RX(pi, inst.qubit))
            flipped_prog.measure(inst.qubit, inst.classical_reg)
            flipped_prog += Pragma('END_PRESERVE_BLOCK')
        else:
            flipped_prog.inst(inst)

    return flipped_prog


def estimate_confusion_matrix(qc: QuantumComputer, qubit: int, shots: int = 10000):
    """
    Estimate the readout confusion matrix for a given qubit.

    :param qc: The quantum computer to estimate the confusion matrix.
    :param qubit: The actual physical qubit to measure
    :param shots: The number of shots to take. This function runs two programs, so
        the total number of shots taken will be twice this number.
    :return: a 2x2 confusion matrix for the qubit, where each row sums to one.
    """
    # prepare 0 (do nothing) and measure; repeat shots number of times
    zero_meas = Program()
    ro_zero = zero_meas.declare("ro", "BIT", 1)
    zero_meas.measure(qubit, ro_zero[0])
    zero_meas.wrap_in_numshots_loop(shots)
    should_be_0 = qc.run(qc.compile(zero_meas))

    # prepare one and measure; repeat shots number of times
    one_meas = Program()
    one_meas += RX(pi, qubit)
    ro_one = one_meas.declare("ro", "BIT", 1)
    one_meas.measure(qubit, ro_one[0])
    one_meas.wrap_in_numshots_loop(shots)
    should_be_1 = qc.run(qc.compile(one_meas))

    p00 = 1 - np.mean(should_be_0)
    p11 = np.mean(should_be_1)

    return np.array([[p00, 1 - p00], [1 - p11, p11]])


def _readout_group_paramaterized(qubits: List[int]) -> Program:
    """
    Produces a parameterized program for the given group of qubits, where each qubit is prepared
    in the 0 or 1 state depending on the parameterization.

    :param qubits: labels of qubits in the group on which some bitstring will be measured
    :return: a parameterized program capable of measuring any bitstring on the given qubits
    """
    program = Program()
    ro = program.declare('ro', memory_type='BIT', memory_size=len(qubits))
    target = program.declare('target', memory_type='REAL', memory_size=len(qubits))
    for idx, qubit in enumerate(qubits):
        program.inst(RX(pi / 2, qubit))
        program.inst(RZ(target[idx], qubit))
        program.inst(RX(-pi / 2, qubit))
        program.measure(qubit, ro[idx])
    return program


def _readout_group_bitstring(qubits: List[int], bitstring: List[int]) -> Program:
    """
    Produces a program that prepares the given bitstring on the given qubits and measures.

    :param qubits: labels of qubits in the group on which the target bitstring will be measured
    :param bitstring: the target bitstring to be read out
    :return: a program for measuring the given bitstring on the given qubits
    """
    program = Program()
    ro = program.declare('ro', memory_type='BIT', memory_size=len(qubits))
    for idx, (qubit, bit) in enumerate(zip(qubits, bitstring)):
        program.inst(RX(pi * bit, qubit))
        program.measure(qubit, ro[idx])
    return program


def estimate_joint_confusion_in_set(qc: QuantumComputer, qubit_set: List[int] = None,
                                    num_shots: int = 1000, joint_group_size: int = 1,
                                    parameterized_program: bool = True) -> Dict[Tuple[int, ...],
                                                                                np.ndarray]:
    """
    Measures the joint confusion matrix for all groups of size group_size among the qubit_set.

    :param qc: a quantum computer whose readout error you wish to characterize
    :param qubit_set: a list of accessible qubits on the qc you wish to characterize. Defaults to
        all qubits in qc.
    :param num_shots: number of shots in measurement of each bit string on each joint group of
        qubits.
    :param joint_group_size: the size of each group; a joint confusion matrix with 
        2^joint_group_size number of rows/columns will be estimated for each group of qubits of 
        the given size within the qubit set. 
    :param parameterized_program: dictates whether to use a parameterized program to measure out
        each bitstring. If set to default of true, this routine should execute faster on a 
        QPU. Note that the parameterized option does not execute a no-op when measuring 0.
    :return: a dictionary whose keys are all possible joint_group_sized tuples of qubits formed 
        from the qubits in the qubit_set. Each value is an estimated 2^group_size square confusion 
        matrix for the corresponding tuple of qubits. Each key is listed in order of increasing  
        qubit number. The corresponding matrix has rows and columns indexed in increasing bitstring
        order, with most significant (leftmost) bit labeling the smallest qubit number.
    """
    # establish default as all operational qubits in qc
    if qubit_set is None:
        qubit_set = qc.qubits()

    qubits = sorted(qubit_set)

    groups = itertools.combinations(qubits, joint_group_size)
    confusion_matrices = {}
    for group in groups:
        if parameterized_program:
            # generate parameterized program for this group
            program = _readout_group_paramaterized(group)
        else:
            program = Program()  # new program will be created for each bitstring

        matrix = np.zeros((2 ** joint_group_size, 2 ** joint_group_size))
        for idx, bitstring in enumerate(itertools.product([0, 1], repeat=joint_group_size)):
            if not parameterized_program:
                program = _readout_group_bitstring(group, bitstring)
            program.wrap_in_numshots_loop(shots=num_shots)
            executable = qc.compiler.native_quil_to_executable(program)

            if parameterized_program:
                results = qc.run(executable, memory_map={'target': [bit * pi for bit in bitstring]})
            else:
                results = qc.run(executable)

            # update confusion matrix
            for result in results:
                base = np.array([2 ** i for i in reversed(range(joint_group_size))])
                observed = np.sum(base * result)
                matrix[idx, observed] += 1 / num_shots

        # store completed confusion matrix in dictionary
        confusion_matrices[group] = matrix
    return confusion_matrices


def marginalize_confusion_matrix(confusion_matrix: np.ndarray, all_qubits: Tuple[int, ...],
                                 marginal_subset: Tuple[int, ...]) -> np.ndarray:
    """
    Marginalize a confusion matrix to get a confusion matrix on only the marginal_subset.

    :param confusion_matrix: a confusion matrix for a group of qubits.
    :param all_qubits: the tuple of qubit labels corresponding to the confusion matrix. Qubits
    should be listed such that the most significant bit labels the first qubit in the tuple.
    :param marginal_subset: a subset of labels in all_qubits. The subset must be listed in the same
        ordering as the elements appear in the group.
    :return: a joint confusion matrix for the qubits in the marginal subset
    """
    indices = []
    subset_index = 0
    # Collect the indices of the marginal_subset elements within the larger group by iterating
    # once through the set of all_qubits.
    for idx, qubit in enumerate(all_qubits):
        if subset_index == len(marginal_subset):
            break  # have found all elements in the subset
        if marginal_subset[subset_index] == qubit:
            indices.append(idx)
            subset_index += 1
    # All elements of the subset should have been found somewhere in the group (assuming proper
    # subset ordering)
    assert len(indices) == len(marginal_subset)

    # Reshape the 2^len(all_qubits) x 2^len(all_qubits) matrix into a tensor with 2*len(all_qubits)
    # axes each with dim = 2 = |{0,1}|
    # Each axis corresponds to a qubit; the first len(all_qubits) axes are qubits labeling the rows
    # of the original matrix, and the last len(all_qubits) axes are the column qubits.
    # The axis index corresponds to the value of the qubit (0 or 1).
    reshaped = confusion_matrix.reshape(np.tile([2] * len(all_qubits), 2))

    # Keep axes corresponding to the previously collected indices; there are two axes per indexed
    # qubit--row and column.
    axes_labels = [i for i in range(2 * len(all_qubits))]
    keep_axes = indices + [len(all_qubits) + i for i in indices]

    # Now simply sum over the axes we are not keeping.
    marginal = np.einsum(reshaped, axes_labels, keep_axes)

    # Reshape back into square matrix and re-normalize
    renormalization_factor = 2 ** (len(all_qubits) - len(marginal_subset))
    dimension = 2 ** len(marginal_subset)
    return marginal.reshape(dimension, dimension) / renormalization_factor
