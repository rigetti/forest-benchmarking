from pyquil.paulis import PauliTerm

from forest.benchmarking.utils import *


def test_trivial_pauli_str_to_pauli_term():
    assert str_to_pauli_term('I') == PauliTerm('I', 0)
    assert str_to_pauli_term('I', set([10])) == PauliTerm('I', 0)
    return


def test_1q_pauli_str_to_pauli_term():
    for pauli_str in ['X', 'Y', 'Z']:
        assert str_to_pauli_term(pauli_str) == PauliTerm(pauli_str, 0)
        assert str_to_pauli_term(pauli_str, set([10])) == PauliTerm(pauli_str, 10)
    return


def test_2q_pauli_str_to_pauli_term():
    for pauli_str1 in ['I', 'X', 'Y', 'Z']:
        for pauli_str2 in ['I', 'X', 'Y', 'Z']:
            assert str_to_pauli_term(pauli_str1 + pauli_str2) == \
                   PauliTerm(pauli_str1, 0) * PauliTerm(pauli_str2, 1)
            assert str_to_pauli_term(pauli_str1 + pauli_str2, [21, 10]) == \
                   PauliTerm(pauli_str1, 21) * PauliTerm(pauli_str2, 10)
    return


def test_all_pauli_terms():
    a1 = all_traceless_pauli_terms([0])
    a2 = all_traceless_pauli_terms([0, 1])
    assert len(a1) == 3
    assert len(a2) == 15
    for pauli_str1 in ['X', 'Y', 'Z']:
        assert PauliTerm(pauli_str1, 0) in a1
    for pauli_str1 in ['I', 'X', 'Y', 'Z']:
        for pauli_str2 in ['I', 'X', 'Y', 'Z']:
            if pauli_str1 == pauli_str2 == 'I':
                continue
            assert PauliTerm(pauli_str1, 0) * PauliTerm(pauli_str2, 1) in a2


def test_all_pauli_z_terms():
    a1 = all_traceless_pauli_z_terms([0])
    a2 = all_traceless_pauli_z_terms([0, 1])
    assert len(a1) == 1
    assert len(a2) == 3
    assert PauliTerm('Z', 0) in a1
    for pauli_str1 in ['I', 'Z']:
        for pauli_str2 in ['I', 'Z']:
            if pauli_str1 == pauli_str2 == 'I':
                continue
            assert PauliTerm(pauli_str1, 0) * PauliTerm(pauli_str2, 1) in a2


def test_partial_trace():
    I = np.asarray([[1, 0], [0, 1]])
    rho = np.kron(I, I) / 4
    np.testing.assert_array_equal(I / 2, partial_trace(rho, [1], [2, 2]))
    np.testing.assert_array_equal(I / 2, partial_trace(rho, [0], [2, 2]))


def test_bitstring_prep():
    # no flips
    flip_prog = bitstring_prep([0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0])
    assert flip_prog.out().splitlines() == ['RX(0) 0',
                                            'RX(0) 1',
                                            'RX(0) 2',
                                            'RX(0) 3',
                                            'RX(0) 4',
                                            'RX(0) 5']
    # mixed flips
    flip_prog = bitstring_prep([0, 1, 2, 3, 4, 5], [1, 1, 0, 1, 0, 1])
    assert flip_prog.out().splitlines() == ['RX(pi) 0',
                                            'RX(pi) 1',
                                            'RX(0) 2',
                                            'RX(pi) 3',
                                            'RX(0) 4',
                                            'RX(pi) 5']
    # flip all
    flip_prog = bitstring_prep([0, 1, 2, 3, 4, 5], [1, 1, 1, 1, 1, 1])
    assert flip_prog.out().splitlines() == ['RX(pi) 0',
                                            'RX(pi) 1',
                                            'RX(pi) 2',
                                            'RX(pi) 3',
                                            'RX(pi) 4',
                                            'RX(pi) 5']
