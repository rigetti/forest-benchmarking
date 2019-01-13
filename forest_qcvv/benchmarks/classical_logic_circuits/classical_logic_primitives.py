"""
Circuit primitives for classical reversible logic
"""
from pyquil.quil import Program
from pyquil.gates import CNOT, CCNOT, X


def majority_gate(a, b, c):
    """
    The majority gate.

    Computes (a * b) xor (a * c) xor  (b * c)
    where * is multiplication mod 2

    See https://arxiv.org/abs/quant-ph/0410184 .
    
    :param a: qubit label
    :param b: qubit label
    :param c: qubit label
    :return: program
    """
    prog = Program()
    prog += CNOT(c, b)
    prog += CNOT(c, a)
    prog += CCNOT(a, b, c)
    return prog


def unmajority_add_gate(a, b, c):
    """
    The UnMajority and Add or UMA gate

    See https://arxiv.org/abs/quant-ph/0410184 .

    :param a: qubit label
    :param b: qubit label
    :param c: qubit label
    :return: program
    """
    prog = Program()
    prog += CCNOT(a, b, c)
    prog += CNOT(c, a)
    prog += CNOT(a, b)
    return prog


def unmajority_add_parallel_gate(a, b, c):
    """
    The UnMajority and Add or UMA gate

    3-CNOT version but admits greater parallelism
    
    See https://arxiv.org/abs/quant-ph/0410184 .

    Computes
    :param a: qubit label
    :param b: qubit label
    :param c: qubit label
    :return: program
    """
    prog = Program()
    prog += X(b)
    prog += CNOT(a, b)
    prog += CCNOT(a, b, c)
    prog += X(b)
    prog += CNOT(c, a)
    prog += CNOT(c, b)
    return prog

