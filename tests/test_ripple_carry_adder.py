"""
Test one bit addition over all inputs
"""
from pyquil.quil import Program
from pyquil.gates import X, I, CNOT, CCNOT, H

from forest_benchmarking.benchmarks.classical_logic_circuits\
    .classical_reversible_logic_primitives import adder, CNOT_X_basis, CCNOT_X_basis


def test_one_bit_addition(qvm):
    """
    Testing the ripple carry adder with one bit addition in the computational  (Z) basis.
    """
    qvm.qam.random_seed = 1

    # q0 = prior carry ancilla   i.e. c0
    # q1 = b0 bit
    # q2 = a0 bit
    # q3 = carry forward bit ie. z bit

    #                    a, b : z, sum
    true_truth_table = {(0, 0): (0, 0),
                        (0, 1): (0, 1),
                        (1, 0): (0, 1),
                        (1, 1): (1, 0)}

    for key, value in true_truth_table.items():
        adder_prog = adder([key[0]], [key[1]], [2], [1], 0, 3)
        exe = qvm.compile(adder_prog)
        result = qvm.run(exe)
        print(key)
        assert tuple(result[0]) == value


def test_one_bit_addition_X_basis(qvm):
    """
    Testing the ripple carry adder with one bit addition in the X basis.
    """
    true_truth_table = {(0, 0): (0, 0),
                        (0, 1): (0, 1),
                        (1, 0): (0, 1),
                        (1, 1): (1, 0)}

    for key, value in true_truth_table.items():
        adder_prog = adder([key[0]], [key[1]], [2], [1], 0, 3, in_x_basis=True)
        exe = qvm.compile(adder_prog)
        result = qvm.run(exe)
        print(key)
        assert tuple(result[0]) == value