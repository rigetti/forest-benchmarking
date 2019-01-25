"""
Test one bit addition over all inputs
"""
from pyquil.api import QVMConnection
from pyquil.quil import Program
from pyquil.gates import X, I, CNOT, CCNOT, H

from forest_benchmarking.benchmarks.classical_logic_circuits\
    .classical_reversible_logic_primitives import adder, CNOT_X_basis, CCNOT_X_basis

qvm = QVMConnection()



def test_one_bit_addition():
    """
    Testing the ripple carry adder with one bit addition in the computational  (Z) basis.
    """

    # q0 = prior carry ancilla   i.e. c0
    # q1 = b0 bit
    # q2 = a0 bit
    # q3 = carry forward bit ie. z bit

    #                    q0,q1,q2,q3: 0, sum,a0,z
    true_truth_table = {(0, 0, 0, 0): (0, 0, 0, 0),
                        (0, 0, 1, 0): (0, 1, 1, 0),
                        (0, 1, 0, 0): (0, 1, 0, 0),
                        (0, 1, 1, 0): (0, 0, 1, 1)}

    adder_prog = adder([2], [1], 0, 3)

    for key, value in true_truth_table.items():
        state_prep_prog = Program().inst(I(2))
        for qbit_idx, bit in enumerate(key):
            if bit == 1:
                state_prep_prog += X(qbit_idx)

        result = qvm.run_and_measure(state_prep_prog + adder_prog,
                                     list(range(4)), trials=1)
        assert tuple(result[0]) == true_truth_table[key]


def test_one_bit_addition_X_basis():
    """
    Testing the ripple carry adder with one bit addition in the X basis.
    """

    # q0 = prior carry ancilla   i.e. c0
    # q1 = b0 bit
    # q2 = a0 bit
    # q3 = carry forward bit ie. z bit

    #                    q0,q1,q2,q3: 0, sum,a0,z
    true_truth_table = {(0, 0, 0, 0): (0, 0, 0, 0),
                        (0, 0, 1, 0): (0, 1, 1, 0),
                        (0, 1, 0, 0): (0, 1, 0, 0),
                        (0, 1, 1, 0): (0, 0, 1, 1)}

    adder_prog = adder([2], [1], 0, 3, CNOT_X_basis, CCNOT_X_basis)

    for key, value in true_truth_table.items():
        state_prep_prog = Program().inst(I(2))
        meas_prog = Program().inst(I(2))
        for qbit_idx, bit in enumerate(key):
            if bit == 1:
                state_prep_prog += X(qbit_idx)
            # Hadamard to get to the X basis
            state_prep_prog += H(qbit_idx)
            # Hadamard to get back to the Z basis before measurement
            meas_prog += H(qbit_idx)
        result = qvm.run_and_measure(state_prep_prog + adder_prog + meas_prog,
                                     list(range(4)), trials=1)
        assert tuple(result[0]) == true_truth_table[key]