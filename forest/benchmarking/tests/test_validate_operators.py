import pytest
import numpy as np
from pyquil.simulation.matrices import X, Y, Z, H
from forest.benchmarking.operator_tools.random_operators import haar_rand_unitary
from forest.benchmarking.operator_tools.validate_operator import *


# Matrix below is from https://en.wikipedia.org/wiki/Normal_matrix
# it is normal but NOT unitary or Hermitian
NORMAL = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])

# not symmetric
SIGMA_MINUS = (X + 1j*Y)/2 # np.array([[0, 1], [0, 0]])

# idempotent
PROJ_ZERO = np.array([[1, 0], [0, 0]])


def test_is_square_matrix():
    assert is_square_matrix(np.eye(3))
    with pytest.raises(ValueError):
        is_square_matrix(np.ndarray(shape=(2, 2, 2)))
    assert not is_square_matrix(np.array([[1, 0]]))


def test_is_symmetric_matrix():
    assert is_symmetric_matrix(X)
    assert not is_symmetric_matrix(SIGMA_MINUS)
    with pytest.raises(ValueError):
        is_symmetric_matrix(np.ndarray(shape=(2, 2, 2)))
    with pytest.raises(ValueError):
        is_symmetric_matrix(np.array([[1, 0]]))


def test_is_identity_matrix():
    assert not is_identity_matrix(Z)
    assert is_identity_matrix(np.eye(3))
    with pytest.raises(ValueError):
        is_identity_matrix(np.ndarray(shape=(2, 2, 2)))
    with pytest.raises(ValueError):
        is_identity_matrix(np.array([[1, 0]]))


def test_is_idempotent_matrix():
    assert not is_idempotent_matrix(SIGMA_MINUS)
    assert is_idempotent_matrix(PROJ_ZERO)
    assert is_idempotent_matrix(np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))


def test_is_normal_matrix():
    assert is_normal_matrix(NORMAL)
    assert not is_normal_matrix(SIGMA_MINUS)


def test_is_hermitian_matrix():
    assert not is_hermitian_matrix(NORMAL)
    assert is_hermitian_matrix(X)
    assert is_hermitian_matrix(Y)


def test_is_unitary_matrix():
    assert not is_unitary_matrix(NORMAL)
    assert is_unitary_matrix(Y)
    assert is_unitary_matrix(haar_rand_unitary(4))


def test_is_positive_definite_matrix():
    # not atol is = 1e-08
    assert not is_positive_definite_matrix(np.array([[-1e-08, 0], [0, 0.1]]))
    assert is_positive_definite_matrix(np.array([[0.5e-08, 0], [0, 0.1]]))


def test_is_positive_semidefinite_matrix():
    # not atol is = 1e-08
    assert not is_positive_semidefinite_matrix(np.array([[-1e-07, 0], [0, 0.1]]))
    assert is_positive_semidefinite_matrix(np.array([[-1e-08, 0], [0, 0.1]]))
    assert is_positive_semidefinite_matrix(np.array([[0.5e-08, 0], [0, 0.1]]))
