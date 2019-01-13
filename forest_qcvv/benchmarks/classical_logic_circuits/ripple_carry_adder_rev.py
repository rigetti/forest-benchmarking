"""
adder programs
"""
from pyquil.quil import Program
from pyquil.gates import CNOT
from pyrethrum.benchmarking.classical_logic_circuits.classical_logic_primitives import majority_gate, unmajority_add_gate


def adder(register_a, register_b, carry_ancilla=None, z_ancilla=None):
    """
    Reversable adding on a quantum computer.

    This implementation is based on:

    "A new quantum ripple-carry addition circuit"
    S. Cuccaro, T. Draper, s. Kutin, D. Moulton
    https://arxiv.org/abs/quant-ph/0410184

    It is not the most efficent but it is easy to implement.

    This method requires two extra ancilla, one for a carry bit 
    and one for fully reversable computing.

    :param register_a: list of qubit labels for register a
    :param register_b: list of qubit labels for register b
    :return: pyQuil program of adder
    :rtype: Program
    """
    if len(register_b) != len(register_a):
        raise ValueError("Registers must be equal length")

    if carry_ancilla is None:
        carry_ancilla = max(register_a + register_b) + 1
    if z_ancilla is None:
        z_ancilla = max(register_a + register_b) + 2

    prog = Program()
    prog_to_rev = Program()
    carry_ancilla_temp = carry_ancilla
    for (a, b) in zip(register_a, register_b):
        prog += majority_gate(carry_ancilla_temp, b, a)
        prog_to_rev += unmajority_add_gate(carry_ancilla_temp, b, a).dagger()
        carry_ancilla_temp = a

    prog += CNOT(register_a[-1], z_ancilla)
    prog += prog_to_rev.dagger()

    return prog


if __name__ == "__main__":
    register_a = list(range(4))
    register_b = list(range(5, 9))
    adder_prog = adder(register_a, register_b)
    print(adder_prog)



