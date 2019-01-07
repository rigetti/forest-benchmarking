from math import pi

import numpy as np
from pyquil.gates import I, MEASURE, RX
from pyquil.quil import Program
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
    """For symmetrization, generate a program where X gates are added before measurement.

    Forest 1.3 is really picky about where the measure instructions happen. It has to be
    at the end!
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


def get_confusion_matrix_programs(qubit):
    """Construct programs for measuring a confusion matrix.

    This is a fancy way of saying "measure |0>"  and "measure |1>".

    :returns: program that should measure |0>, program that should measure |1>.
    """
    zero_meas = Program()
    zero_meas += I(qubit)
    zero_meas += I(qubit)
    zero_meas += MEASURE(qubit, 0)

    # prepare one and get statistics
    one_meas = Program()
    one_meas += I(qubit)
    one_meas += RX(pi, qubit)
    one_meas += MEASURE(qubit, 0)

    return zero_meas, one_meas


def estimate_confusion_matrix(qam: 'QuantumComputer', qubit: int, samples=10000):
    """Estimate the readout confusion matrix for a given qubit.

    :param qam: The quantum computer to estimate the confusion matrix.
    :param qubit: The actual physical qubit to measure
    :param samples: The number of shots to take. This function runs two programs, so
        the total number of shots taken will be twice this number.
    """
    zero_meas, one_meas = get_confusion_matrix_programs(qubit)
    should_be_0 = qam.run(zero_meas, [0], samples)
    should_be_1 = qam.run(one_meas, [0], samples)
    p00 = 1 - np.mean(should_be_0)
    p11 = np.mean(should_be_1)

    return np.array([[p00, 1 - p00],
                     [1 - p11, p11]])
