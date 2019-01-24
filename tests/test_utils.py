from pyquil.paulis import PauliTerm

from forest_benchmarking.utils import *


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
                   PauliTerm(pauli_str1, 1) * PauliTerm(pauli_str2, 0)
            assert str_to_pauli_term(pauli_str1 + pauli_str2, set([10, 21])) == \
                   PauliTerm(pauli_str1, 21) * PauliTerm(pauli_str2, 10)
    return


def test_all_pauli_terms():
    a1 = all_pauli_terms(1)
    a2 = all_pauli_terms(2)
    assert len(a1) == 3
    assert len(a2) == 15
    for pauli_str1 in ['X', 'Y', 'Z']:
        assert PauliTerm(pauli_str1, 0) in a1
    for pauli_str1 in ['I', 'X', 'Y', 'Z']:
        for pauli_str2 in ['I', 'X', 'Y', 'Z']:
            if not (pauli_str1 == 'I' and pauli_str2 == 'I'):
                assert PauliTerm(pauli_str1, 0) * PauliTerm(pauli_str2, 1) in a2


def test_all_pauli_z_terms():
    a1 = all_pauli_z_terms(1)
    a2 = all_pauli_z_terms(2)
    assert len(a1) == 1
    assert len(a2) == 3
    assert PauliTerm('Z', 0) in a1
    for pauli_str1 in ['I', 'Z']:
        for pauli_str2 in ['I', 'Z']:
            if not (pauli_str1 == 'I' and pauli_str2 == 'I'):
                assert PauliTerm(pauli_str1, 0) * PauliTerm(pauli_str2, 1) in a2


def test_partial_trace():
    I = np.asarray([[1, 0], [0, 1]])
    rho = np.kron(I, I) / 4
    np.testing.assert_array_equal(I / 2, partial_trace(rho, [1], [2, 2]))
    np.testing.assert_array_equal(I / 2, partial_trace(rho, [0], [2, 2]))
