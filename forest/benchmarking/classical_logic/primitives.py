from pyquil.gates import CNOT, CCNOT, X, I, H, CZ
from pyquil import Program


def CNOT_X_basis(control, target) -> Program:
    """
    The CNOT in the X basis, i.e.

    ::

        CNOTX = |+X+| * I + |-X-| * Z

    where ``|+>`` and ``|->`` are the +/- eigenstate of the Pauli X operator, and ``*`` denotes a
    tensor product.

    :param control: qubit label
    :param target: qubit label
    :return: program
    """
    prog = Program()
    prog += H(control)
    prog += CZ(control, target)
    prog += H(control)
    return prog


def CCNOT_X_basis(control1, control2, target) -> Program:
    """
    The CCNOT (Toffoli) in the X basis, i.e.

    ::

        CCNOTX = |+X+| * |+X+| * I +
                 |+X+| * |-X-| * I +
                 |-X-| * |+X+| * I +
                 |-X-| * |-X-| * Z

    where ``|+>`` and ``|->`` are the +/- eigenstate of the Pauli X operator, and ``*`` denotes a
    tensor product.

    :param control1: qubit label
    :param control2: qubit label
    :param target: qubit label
    :return: program
    """
    prog = Program()
    prog += H(control1)
    prog += H(control2)
    prog += H(target)
    prog += CCNOT(control1, control2, target)
    prog += H(control1)
    prog += H(control2)
    prog += H(target)
    return prog


def majority_gate(a: int, b: int, c: int, in_x_basis: bool = False) -> Program:
    """
    The majority gate.

    Computes (a * b) xor (a * c) xor  (b * c)
    where * is multiplication mod 2.

    The default option is to compute this in the computational (aka Z) basis. If in_x_basis is
    true then the computation is instead in the X basis, i.e.
    CNOT is replaced by CNOT_X_basis and CCNOT is replaced by CCNOT_X_basis

    See [CDKM96]_ reference in ripple_carry_adder.adder() docstring

    :param a: qubit label
    :param b: qubit label
    :param c: qubit label
    :param in_x_basis: if true, the returned program performs the equivalent logic in the X basis.
    :return: program which results in (c xor a) on the c line, (b xor a) on the b line,
        and the output (majority of the inputs) on the a line.
    """
    if in_x_basis:
        cnot_gate = CNOT_X_basis
        ccnot_gate = CCNOT_X_basis
    else:
        cnot_gate = CNOT
        ccnot_gate = CCNOT

    prog = Program()
    prog += cnot_gate(a, b)
    prog += cnot_gate(a, c)
    prog += ccnot_gate(c, b, a)
    return prog


def unmajority_add_gate(a: int, b: int, c: int, in_x_basis: bool = False) -> Program:
    """
    The UnMajority and Add gate, or UMA for short.

    The default option is to compute this in the computational (aka Z) basis. If in_x_basis is
    true then the computation is instead in the X basis, i.e.
    CNOT is replaced by CNOT_X_basis and CCNOT is replaced by CCNOT_X_basis

    See [CDKM96]_ reference in ripple_carry_adder.adder() docstring

    .. seealso:: :func:`unmajority_add_parallel_gate`

    :param a: qubit label
    :param b: qubit label
    :param c: qubit label
    :param in_x_basis: if true, the returned program performs the equivalent logic in the X basis.
    :return: program which when run on the output of majority_gate(a,b,c) returns the input to
        majority_gate on the c and a lines, and outputs the sum of a+b+c (mod 2) on the b line.
    """
    if in_x_basis:
        cnot_gate = CNOT_X_basis
        ccnot_gate = CCNOT_X_basis
    else:
        cnot_gate = CNOT
        ccnot_gate = CCNOT

    prog = Program()
    prog += ccnot_gate(c, b, a)
    prog += cnot_gate(a, c)
    prog += cnot_gate(c, b)
    return prog


def unmajority_add_parallel_gate(a: int, b: int, c: int, in_x_basis: bool = False) -> Program:
    """
    An alternative form of the UnMajority and Add gate, or UMA for short.

    This implementation of UMA has 3-CNOTs, rather than 2, but admits greater parallelism

    See [CDKM96]_ reference in ripple_carry_adder.adder() docstring

    .. seealso:: :func:`unmajority_add_gate`

    :param a: qubit label
    :param b: qubit label
    :param c: qubit label
    :param in_x_basis: if true, the returned program performs the equivalent logic in the X basis.
    :return: program that executes the same logic as unmajority_add_gate but with different gates
    """
    if in_x_basis:
        cnot_gate = CNOT_X_basis
        ccnot_gate = CCNOT_X_basis
    else:
        cnot_gate = CNOT
        ccnot_gate = CCNOT

    prog = Program()
    prog += X(b)
    prog += cnot_gate(a, b)
    prog += ccnot_gate(a, b, c)
    prog += X(b)
    prog += cnot_gate(c, a)
    prog += cnot_gate(c, b)
    return prog
