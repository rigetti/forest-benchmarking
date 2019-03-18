"""
Test over all inputs
"""
from pyquil.quil import Program
from pyquil.gates import X, I, CNOT, CCNOT, H, MEASURE
from forest.benchmarking.classical_logic.primitives import *


def test_majority_gate(qvm):
    """
    Testing the majority gate with a truth table
    """
    #                    a, b, c    a, b, c
    true_truth_table = {(0, 0, 0): (0, 0, 0),
                        (0, 0, 1): (0, 0, 1),
                        (0, 1, 0): (0, 1, 0),
                        (0, 1, 1): (1, 1, 1),
                        (1, 0, 0): (0, 1, 1),
                        (1, 0, 1): (1, 1, 0),
                        (1, 1, 0): (1, 0, 1),
                        (1, 1, 1): (1, 0, 0)}

    maj_gate_program = majority_gate(0, 1, 2)
    for key, value in true_truth_table.items():
        state_prep_prog = Program()
        for qbit_idx, bit in enumerate(key):
            if bit == 1:
                state_prep_prog += X(qbit_idx)
        prog = state_prep_prog + maj_gate_program
        ro = prog.declare('ro', 'BIT', 3)
        for q in range(3):
            prog += MEASURE(q, ro[q])
        exe = qvm.compiler.native_quil_to_executable(prog)
        result = qvm.run(exe)
        assert tuple(result[0]) == true_truth_table[key]


def test_unmajority_add_gate(qvm):
    """
    Testing the Unmajority add gate with a truth table
    """
    true_truth_table = {(0, 0, 0): (0, 0, 0),
                        (0, 0, 1): (0, 1, 1),
                        (0, 1, 0): (0, 1, 0),
                        (0, 1, 1): (1, 1, 0),
                        (1, 0, 0): (1, 1, 1),
                        (1, 0, 1): (1, 0, 0),
                        (1, 1, 0): (1, 0, 1),
                        (1, 1, 1): (0, 0, 1)}

    unmaj_add_gate_program = unmajority_add_gate(0, 1, 2)
    for key, value in true_truth_table.items():
        state_prep_prog = Program()
        for qbit_idx, bit in enumerate(key):
            if bit == 1:
                state_prep_prog += X(qbit_idx)
        prog = state_prep_prog + unmaj_add_gate_program
        ro = prog.declare('ro', 'BIT', 3)
        for q in range(3):
            prog += MEASURE(q, ro[q])
        exe = qvm.compiler.native_quil_to_executable(prog)
        result = qvm.run(exe)
        assert tuple(result[0]) == true_truth_table[key]


def test_composition_of_majority_and_unmajority_gates(qvm):
    """
    Testing the composition of the majority gate with the unmajority add gate with a truth table
    """
    true_truth_table = {(0, 0, 0): (0, 0, 0),
                        (0, 0, 1): (0, 1, 1),
                        (0, 1, 0): (0, 1, 0),
                        (0, 1, 1): (0, 0, 1),
                        (1, 0, 0): (1, 1, 0),
                        (1, 0, 1): (1, 0, 1),
                        (1, 1, 0): (1, 0, 0),
                        (1, 1, 1): (1, 1, 1)}

    compose_maj_and_unmaj_gate_program = majority_gate(0, 1, 2) + unmajority_add_gate(0, 1, 2)
    for key, value in true_truth_table.items():
        state_prep_prog = Program()
        for qbit_idx, bit in enumerate(key):
            if bit == 1:
                state_prep_prog += X(qbit_idx)
        prog = state_prep_prog + compose_maj_and_unmaj_gate_program
        ro = prog.declare('ro', 'BIT', 3)
        for q in range(3):
            prog += MEASURE(q, ro[q])
        exe = qvm.compiler.native_quil_to_executable(prog)
        result = qvm.run(exe)
        assert tuple(result[0]) == true_truth_table[key]


def test_CNOT_in_X_basis(qvm):
    """
    Testing the definition of CNOT in the X basis.
    """
    # CNOT truth table
    true_truth_table = {(0, 0): (0, 0),
                        (0, 1): (0, 1),
                        (1, 0): (1, 1),
                        (1, 1): (1, 0)}

    CNOTX = CNOT_X_basis(0, 1)
    for key, value in true_truth_table.items():
        state_prep_prog = Program()
        meas_prog = Program()
        for qbit_idx, bit in enumerate(key):
            if bit == 1:
                state_prep_prog += X(qbit_idx)
            # Hadamard to get to the X basis
            state_prep_prog += H(qbit_idx)
            # Hadamard to get back to the Z basis before measurement
            meas_prog += H(qbit_idx)

        prog = state_prep_prog + CNOTX + meas_prog
        ro = prog.declare('ro', 'BIT', 3)
        for q in range(2):
            prog += MEASURE(q, ro[q])
        exe = qvm.compiler.native_quil_to_executable(prog)
        result = qvm.run(exe)
        assert tuple(result[0]) == true_truth_table[key]


def test_CCNOT_in_X_basis(qvm):
    """
    Testing the definition of Toffoli / CCNOT in the X basis.
    """
    # Toffoli truth table
    true_truth_table = {(0, 0, 0): (0, 0, 0),
                        (0, 0, 1): (0, 0, 1),
                        (0, 1, 0): (0, 1, 0),
                        (0, 1, 1): (0, 1, 1),
                        (1, 0, 0): (1, 0, 0),
                        (1, 0, 1): (1, 0, 1),
                        (1, 1, 0): (1, 1, 1),
                        (1, 1, 1): (1, 1, 0)}

    CCNOTX = CCNOT_X_basis(0, 1, 2)
    for key, value in true_truth_table.items():
        state_prep_prog = Program().inst(I(2))
        meas_prog = Program()
        for qbit_idx, bit in enumerate(key):
            if bit == 1:
                state_prep_prog += X(qbit_idx)
            # Hadamard to get to the X basis
            state_prep_prog += H(qbit_idx)
            # Hadamard to get back to the Z basis before measurement
            meas_prog += H(qbit_idx)

        prog=state_prep_prog + CCNOTX + meas_prog
        ro = prog.declare('ro', 'BIT', 3)
        for q in range(3):
            prog += MEASURE(q, ro[q])
        exe = qvm.compiler.native_quil_to_executable(prog)
        result = qvm.run(exe)
        assert tuple(result[0]) == true_truth_table[key]
