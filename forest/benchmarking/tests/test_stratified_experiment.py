from numpy import random

from forest.benchmarking.stratified_experiment import *
from pyquil import Program
from pyquil.gates import X, Y


def test_merge_sequences():
    random.seed(0)
    seq0 = [Program(X(0)), Program(Y(0)), Program(X(0))]
    seq1 = [Program(X(1)), Program(Y(1)), Program(Y(1))]
    assert merge_sequences([seq0, seq1]) == [Program(X(0), X(1)),
                                             Program(Y(0), Y(1)),
                                             Program(X(0), Y(1))]
