from math import pi
from typing import List, Tuple, Dict
import numpy as np
import itertools
from pyquil.gates import I, MEASURE, RX, RZ
from pyquil import Program
from pyquil.api import QuantumComputer
from pyquil.quil import address_qubits
from pyquil.quilatom import QubitPlaceholder
from pyquil.quilbase import Measurement, Pragma


def get_flipped_program(program: Program):
    """For symmetrization, generate a program where X gates are added before measurement."""

    flipped_prog = Program()
    for inst in program:
        if isinstance(inst, Measurement):
            flipped_prog += Pragma('PRESERVE_BLOCK')
            flipped_prog += RX(pi, inst.qubit)
            flipped_prog += Measurement(qubit=inst.qubit, classical_reg=inst.classical_reg)
            flipped_prog += Pragma('END_PRESERVE_BLOCK')
        else:
            flipped_prog += inst

    return flipped_prog


def get_flipped_protoquil_program(program: Program):
    """
    For symmetrization, generate a program where X gates are added before measurement.

    Forest 2 ProtoQuil requires that measurement instructions occur at the end of the program, so all X gates must be
    inserted before the first measurement. 
    """
    program = Program(program.instructions)  # Copy
    to_measure = []
    while True:
        inst = program.instructions[-1]
        if isinstance(inst, Measurement):
            program.pop()
            to_measure.append((inst.qubit, inst.classical_reg))
        else:
            break

    program += Pragma('PRESERVE_BLOCK')
    for qu, addr in to_measure[::-1]:
        program += RX(pi, qu)
    program += Pragma('END_PRESERVE_BLOCK')

    for qu, addr in to_measure[::-1]:
        program += Measurement(qubit=qu, classical_reg=addr)

    return program


def get_confusion_matrix_programs(qubit: int):
    """
    Construct programs for measuring a confusion matrix.

    This is a fancy way of saying "measure |0>"  and "measure |1>".

    :returns: program that should measure |0>, program that should measure |1>.
    """
    zero_meas = Program()
    zero_meas += I(qubit)
    zero_meas += I(qubit)

    # prepare one and get statistics
    one_meas = Program()
    one_meas += I(qubit)
    one_meas += RX(pi, qubit)

    return zero_meas, one_meas


def estimate_confusion_matrix(qc: QuantumComputer, qubit: int, samples=10000):
    """
    Estimate the readout confusion matrix for a given qubit.

    :param qc: The quantum computer to estimate the confusion matrix.
    :param qubit: The actual physical qubit to measure
    :param samples: The number of shots to take. This function runs two programs, so
        the total number of shots taken will be twice this number.
    :return: a 2x2 confusion matrix for the qubit, where each row sums to one.
    """
    zero_meas, one_meas = get_confusion_matrix_programs(qubit)
    should_be_0 = qc.run_and_measure(zero_meas, samples)[qubit]
    should_be_1 = qc.run_and_measure(one_meas, samples)[qubit]
    p00 = 1 - np.mean(should_be_0)
    p11 = np.mean(should_be_1)

    return np.array([[p00, 1 - p00],
                     [1 - p11, p11]])


def readout_group_param(num_qubits: int) -> Tuple[Program, List[QubitPlaceholder]]:
    """
    Produces an unaddressed program of the given number of qubits where each qubit has a paramaterized RX followed by
    a measurement.

    :param num_qubits: number of qubits in the group for which the readout error will be measured
    :return: an unaddressed program along with the placeholders necessary for addressing
    """
    qubit_placeholders = QubitPlaceholder.register(num_qubits)
    program = Program()
    ro = program.declare('ro', memory_type='BIT', memory_size=num_qubits)
    target = program.declare('target', memory_type='REAL', memory_size=num_qubits)
    for idx, qubit in enumerate(qubit_placeholders):
        program.inst(RX(pi / 2, qubit))
        program.inst(RZ(target[idx], qubit))
        program.inst(RX(-pi / 2, qubit))
        program += MEASURE(qubit, ro[idx])
    return program, qubit_placeholders


def measure_grouped_readout_error_param(qc: QuantumComputer,
                                        qubit_labels: List[int] = [],
                                        num_shots: int = 1000,
                                        group_size: int = 1) -> Dict[Tuple[int, ...], np.ndarray]:
    """
    Measures the confusion matrix for all groups of qubits in qubit_labels of the given group size.

    :param qc: a quantum computer whose readout error you wish to characterize
    :param qubit_labels: a list of accessible qubits on the qc you wish to characterize. Defaults to all qubits in qc.
    :param num_shots: number of shots for each group of qubits
    :param group_size: the size of the groups; there will be a group for each subset of qubits_labels with this size
    :return: a dictionary whose keys are all possible groups of the group_size formed from the qubits in qubit_labels
        and whose values are the confusion matrix for the corresponding group, each a 2**group_size square matrix.
    """
    # establish default as all operational qubits
    if not qubit_labels:
        qubit_labels = qc.qubits()

    qubit_labels = sorted(qubit_labels)

    program, placeholders = readout_group_param(group_size)

    groups = itertools.combinations(qubit_labels, group_size)
    confusion_matrices = {}
    for group in groups:
        prog = address_qubits(program, qubit_mapping={ph: qubit for ph, qubit in zip(placeholders, qubit_labels)})
        prog.wrap_in_numshots_loop(shots=num_shots)
        executable = qc.compiler.native_quil_to_executable(prog)

        matrix = np.zeros((2 ** group_size, 2 ** group_size))
        for idx, bitstring in enumerate(itertools.product([0, 1], repeat=group_size)):
            results = qc.run(executable, memory_map={'target': [bit * pi for bit in bitstring]})
            _update_confusion_matrix(matrix, idx, results)

        confusion_matrices[group] = matrix
    return confusion_matrices


def readout_group_bitstring(qubits: Tuple[int, ...], bitstring: Tuple[int, ...]) -> Program:
    """
    Produces a program that prepares the given bitstring on the given qubits and measures.

    :param qubits: qubits on which the target bitstring will be measured
    :param bitstring: the target bitstring to be read out
    :return: a program for measuring the readout of the given bitstring on the given qubits
    """
    program = Program()
    ro = program.declare('ro', memory_type='BIT', memory_size=len(qubits))
    for idx, (qubit, bit) in enumerate(zip(qubits, bitstring)):
        program.inst(RX(pi * bit, qubit))
        program += MEASURE(qubit, ro[idx])
    return program


def measure_grouped_readout_error(qc: QuantumComputer,
                                  qubits: Tuple[int, ...] = [],
                                  num_shots: int = 1000,
                                  group_size: int = 1) -> Dict[Tuple[int, ...], np.ndarray]:
    """
    Measures the confusion matrix for all groups of qubits in qubit_labels of the given group size.

    :param qc: a quantum computer whose readout error you wish to characterize
    :param qubits: a list of accessible qubits on the qc you wish to characterize. Defaults to all qubits in qc.
    :param num_shots: number of shots for each group of qubits
    :param group_size: the size of the groups; there will be a group for each subset of qubits_labels with this size
    :return: a dictionary whose keys are all possible groups of the group_size formed from the qubits in qubit_labels
    and whose values are the confusion matrix for the corresponding group, each a 2**group_size square matrix.
    """
    # establish default as all operational qubits
    if not qubits:
        qubits = qc.qubits()

    qubits = sorted(qubits)

    groups = itertools.combinations(qubits, group_size)
    confusion_matrices = {}
    for group in groups:
        matrix = np.zeros((2 ** group_size, 2 ** group_size))
        for idx, bitstring in enumerate(itertools.product([0, 1], repeat=group_size)):
            program = readout_group_bitstring(group, bitstring)
            program.wrap_in_numshots_loop(shots=num_shots)
            executable = qc.compiler.native_quil_to_executable(program)
            results = qc.run(executable)
            _update_confusion_matrix(matrix, idx, results)

        confusion_matrices[group] = matrix
    return confusion_matrices


def _update_confusion_matrix(matrix: np.ndarray, target_index: int, results: np.ndarray):
    """
    Updates the input confusion matrix based on the target bitstring index and the measured results.

    :param matrix: confusion matrix to be updated. The input matrix will be mutated.
    :param target_index: index corresponding to the target bitstring.
    :param results: an num_shots by num_qubits array of results where each result (row) is a measured bitstring
    :return: None. Mutates input matrix
    """
    num_shots, num_qubits = results.shape
    for result in results:
        base = np.array([2 ** i for i in reversed(range(num_qubits))])
        observed = np.sum(base * result)
        matrix[target_index][observed] += 1 / num_shots


def marginal_for_qubits(confusion_matrix: np.ndarray,
                        group: Tuple[int, ...],
                        marginal_subset: Tuple[int, ...]) -> np.ndarray:
    """
    Marginalize a confusion matrix to get a confusion matrix on only the marginal_subset.

    :param confusion_matrix: a confusion matrix for a group of qubits
    :param group: the group of qubits in order corresponding to the input confusion matrix
    :param marginal_subset: a subset of the group of qubits. The subset must be in the same order as the group.
    :return: a confusion matrix for the marginal subset
    """
    indices = []
    subset_index = 0
    # collect the indices of the marginal_subset elements within the larger group by iterating once through the group.
    for idx, qubit in enumerate(group):
        if subset_index == len(marginal_subset):
            break  # have found all elements in the subset
        if marginal_subset[subset_index] == qubit:
            indices.append(idx)
            subset_index += 1
    # all elements of the subset should have been found somewhere in the group (assuming proper subset ordering)
    assert len(indices) == len(marginal_subset)

    # reshape the 2^len(group) x 2^len(group) matrix into a tensor with 2*len(group) axes each with dim = 2 = |{0,1}|
    # Each axis corresponds to a qubit; the first len(group) axes are qubits labeling the rows of the original matrix,
    # and the last len(group) axes are the column qubits. The axis index corresponds to the value of the qubit (0 or 1).
    reshaped = confusion_matrix.reshape(np.tile([2] * len(group), 2))
    # keep axes corresponding to the previously collected indices; there are two axes per indexed qubit--row and column.
    axes_labels = [i for i in range(2 * len(group))]
    keep_axes = indices + [len(group) + i for i in indices]
    # now simply sum over the axes we are not keeping.
    marginal = np.einsum(reshaped, axes_labels, keep_axes)
    # reshape back into matrix and re-normalize
    return marginal.reshape(2 ** len(indices), 2 ** len(indices)) / 2 ** (len(group) - len(marginal_subset))
