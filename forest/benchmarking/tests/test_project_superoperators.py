from pyquil.simulation.matrices import I, X, Y, Z, H, CNOT

from forest.benchmarking.operator_tools.project_superoperators import *
from forest.benchmarking.operator_tools.validate_superoperator import *


def test_proj_to_cp():
    state = np.eye(2)
    assert np.allclose(state, proj_choi_to_completely_positive(state))

    state = np.array([[1.5, 0], [0, 10]])
    assert choi_is_completely_positive(state)
    assert np.allclose(state, proj_choi_to_completely_positive(state))

    state = -Z
    cp_state = proj_choi_to_completely_positive(state)
    target = np.array([[0, 0], [0, 1.]])
    assert choi_is_completely_positive(cp_state)
    assert np.allclose(target, cp_state)

    state = X
    cp_state = proj_choi_to_completely_positive(state)
    target = np.array([[.5, .5], [.5, .5]])
    assert choi_is_completely_positive(cp_state)
    assert np.allclose(target, cp_state)

    state = Y
    cp_state = proj_choi_to_completely_positive(state)
    target = np.array([[.5, -.5j], [.5j, .5]])
    assert choi_is_completely_positive(cp_state)
    assert np.allclose(target, cp_state)

    choi = kraus2choi(np.kron(X, Z))
    assert choi_is_completely_positive(choi)
    assert np.allclose(choi, proj_choi_to_completely_positive(choi))


def test_proj_to_tp():
    # Identity process is trace preserving, so no change
    choi = kraus2choi(np.eye(2))
    assert np.allclose(choi, proj_choi_to_trace_preserving(choi))

    # Bit flip process is trace preserving, so no change
    choi = kraus2choi(X)
    assert np.allclose(choi, proj_choi_to_trace_preserving(choi))

    # start with a non-trace-preserving choi.
    choi = kraus2choi(X - np.eye(2)*.01)
    assert choi_is_trace_preserving(proj_choi_to_trace_preserving(choi))


def test_proj_to_tni():
    # Bit flip process is trace preserving, so no change
    choi = kraus2choi(X)
    assert np.allclose(choi, proj_choi_to_trace_non_increasing(choi))

    choi = np.array(
        [[0., 0., 0., 0.], [0., 1.01, 1.01, 0.], [0., 1., 1., 0.], [0., 0., 0., 0.]])
    assert choi_is_trace_preserving(proj_choi_to_trace_non_increasing(choi))

    # start with a non-trace-preserving choi.
    choi = kraus2choi(np.kron(X - np.eye(2) * .01, np.eye(2)))
    choi_tni = proj_choi_to_trace_non_increasing(choi)
    plusplus = np.array([[1, 1, 1, 1]]).T / 2
    rho_pp = plusplus @ plusplus.T
    output = apply_choi_matrix_2_state(choi_tni, rho_pp)
    assert 0 < np.trace(output) <= 1


def test_proj_to_cptp():
    # Identity process is cptp, so no change
    choi = kraus2choi(np.eye(2))
    assert np.allclose(choi, proj_choi_to_physical(choi))

    # Bit flip process is cptp, so no change
    choi = kraus2choi(X)
    assert np.allclose(choi, proj_choi_to_physical(choi))

    # Small perturbation shouldn't change too much
    choi = np.array([[1.001, 0., 0., .99], [0., 0., 0., 0.], [0., 0., 0., 0.],
                     [1.004, 0., 0., 1.01]])
    assert np.allclose(choi, proj_choi_to_physical(choi), atol=1e-2)

    # Ensure final product is cptp with arbitrary perturbation
    choi = np.array([[1.1, 0.2, -0.4, .9], [.5, 0., 0., 0.], [0., 0., 0., 0.],
                     [1.4, 0., 0., .8]])
    physical_choi = proj_choi_to_physical(choi)
    assert choi_is_trace_preserving(physical_choi)
    assert choi_is_completely_positive(physical_choi, atol=1e-1)
    assert choi_is_cptp(physical_choi, atol=1e-1)

def test_choi_to_unitary():
    choi_CNOT = kraus2choi(CNOT)
    ans = proj_choi_to_unitary(choi_CNOT)
    assert np.allclose(choi_CNOT, ans)
    choi_Z = kraus2choi(Z)
    ans = proj_choi_to_unitary(choi_Z)
    assert np.allclose(choi_Z, ans)
    choi_H = kraus2choi(H)
    ans = proj_choi_to_unitary(choi_H)
    assert np.allclose(choi_H, ans)
    def bit_flip_kraus(p):
        M0 = np.sqrt(1 - p) * I
        M1 = np.sqrt(p) * X
        return [M0, M1]
    # close to identity
    choi_I = kraus2choi(I)
    ans = proj_choi_to_unitary(kraus2choi(bit_flip_kraus(0.1)))
    assert np.allclose(choi_I, ans)
    # close to Pauli X
    choi_X = kraus2choi(X)
    ans = proj_choi_to_unitary(kraus2choi(bit_flip_kraus(0.9)))
    assert np.allclose(choi_X, ans)
