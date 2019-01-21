"""
Test one bit addition over all inputs
"""
from pyquil.api import QVMConnection
from pyquil.quil import Program
from pyquil.gates import X, I, CNOT, CCNOT

from forest_qcvv.benchmarks.classical_logic_circuits.classical_reversible_logic_primitives import\
    adder

qvm = QVMConnection()

def test_one_bit_addition():
    """
    Testing the ripple carry adder with one bit addition
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
