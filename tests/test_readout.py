import re
import numpy as np
from pyquil import Program
from pyquil.device import gates_in_isa
from pyquil.gates import I, RX, CNOT, MEASURE
from pyquil.noise import decoherence_noise_with_asymmetric_ro

from forest_qcvv.readout import get_flipped_program, measure_grouped_readout_error, marginal_for_qubits, \
    estimate_confusion_matrix, measure_grouped_readout_error_param


def test_get_flipped_program():
    program = Program()
    ro = program.declare('ro', memory_type='BIT', memory_size=2)
    program += Program([
        I(0),
        RX(2.3, 1),
        CNOT(0, 1),
        MEASURE(0, ro[0]),
        MEASURE(1, ro[1]),
    ])

    flipped_program = get_flipped_program(program)

    lines = flipped_program.out().splitlines()
    matched = 0
    for l1, l2 in zip(lines, lines[1:]):
        ma = re.match(r'MEASURE (\d) ro\[(\d)\]', l2)
        if ma is not None:
            matched += 1
            assert int(ma.group(1)) == int(ma.group(2))
            assert l1 == 'RX(pi) {}'.format(int(ma.group(1)))

    assert matched == 2


def test_consistency(qvm):
    noise_model = decoherence_noise_with_asymmetric_ro(gates=gates_in_isa(qvm.device.get_isa()))
    qvm.qam.noise_model = noise_model
    qvm.qam.random_seed = 1
    num_shots = 10000
    qubits = (0, 1, 2)
    qubit = (0,)
    cm1_3q = measure_grouped_readout_error(qvm, qubits=qubits, num_shots=num_shots, group_size=len(qubits))[qubits]
    cm1 = marginal_for_qubits(cm1_3q, qubits, qubit)
    cm2 = measure_grouped_readout_error(qvm, qubits=qubit, num_shots=num_shots, group_size=1)[qubit]
    cm3 = estimate_confusion_matrix(qvm, qubit[0], num_shots)
    cm4 = measure_grouped_readout_error_param(qvm, qubits=qubit, num_shots=num_shots, group_size=1)[qubit]
    atol = .01
    assert np.allclose(cm1, cm2, atol=atol)
    assert np.allclose(cm2, cm3, atol=atol)
    assert np.allclose(cm1, cm3, atol=atol)
    assert np.allclose(cm2, cm4, atol=atol)
