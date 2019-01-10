import re

import numpy as np
from pyquil import Program
from pyquil.device import gates_in_isa
from pyquil.gates import I, RX, CNOT, MEASURE
from pyquil.noise import decoherence_noise_with_asymmetric_ro

from forest_qcvv.readout import get_flipped_program, estimate_confusion_matrix, \
    estimate_joint_confusion_in_set, marginalize_confusion_matrix


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


def test_confusion_matrix_consistency(qvm):
    noise_model = decoherence_noise_with_asymmetric_ro(gates=gates_in_isa(qvm.device.get_isa()))
    qvm.qam.noise_model = noise_model
    qvm.qam.random_seed = 1
    num_shots = 500
    qubits = (0, 1, 2)
    qubit = (0,)

    # parameterized confusion matrices
    cm_3q_param = estimate_joint_confusion_in_set(qvm, qubit_set=qubits, num_shots=num_shots,
                                                  joint_group_size=len(qubits))[qubits]
    cm_1q_param = estimate_joint_confusion_in_set(qvm, qubit_set=qubit, num_shots=num_shots,
                                                  joint_group_size=1)[qubit]

    # non-parameterized confusion matrices
    cm_3q = estimate_joint_confusion_in_set(qvm, qubit_set=qubits, num_shots=num_shots,
                                            joint_group_size=len(qubits),
                                            parameterized_program=False)[qubits]
    cm_1q = estimate_joint_confusion_in_set(qvm, qubit_set=qubit, num_shots=num_shots,
                                            joint_group_size=1,
                                            parameterized_program=False)[qubit]
    # single qubit cm
    single_q = estimate_confusion_matrix(qvm, qubit[0], num_shots)

    # marginals from 3q above
    marginal_1q_param = marginalize_confusion_matrix(cm_3q_param, qubits, qubit)
    marginal_1q = marginalize_confusion_matrix(cm_3q, qubits, qubit)

    atol = .03
    assert np.allclose(cm_3q_param, cm_3q, atol=atol)
    assert np.allclose(cm_1q_param, single_q, atol=atol)
    assert np.allclose(cm_1q, single_q, atol=atol)
    assert np.allclose(cm_1q_param, marginal_1q_param, atol=atol)
    assert np.allclose(cm_1q, marginal_1q, atol=atol)
    assert np.allclose(marginal_1q_param, single_q, atol=atol)
