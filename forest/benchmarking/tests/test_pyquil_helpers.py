import itertools

import networkx as nx
import numpy as np
import pytest

from pyquil import Program, get_qc, list_quantum_computers
from pyquil.api import QVM, QuantumComputer, local_qvm
from pyquil.device import NxDevice, gates_in_isa
from pyquil.gates import *
from pyquil.noise import decoherence_noise_with_asymmetric_ro
from pyquil.pyqvm import PyQVM

from forest.benchmarking.pyquil_helpers import (_symmetrization,
                                                _flip_array_to_prog,
                                                _construct_orthogonal_array,
                                                _construct_strength_two_orthogonal_array,
                                                _construct_strength_three_orthogonal_array,
                                                _measure_bitstrings, _consolidate_symmetrization_outputs,
                                                _check_min_num_trials_for_symmetrized_readout)


class DummyCompiler(AbstractCompiler):
    def get_version_info(self):
        return {}

    def quil_to_native_quil(self, program: Program):
        return program

    def native_quil_to_executable(self, nq_program: Program):
        return nq_program


def test_flip_array_to_prog():
    # no flips
    flip_prog = _flip_array_to_prog((0, 0, 0, 0, 0, 0), [0, 1, 2, 3, 4, 5])
    assert flip_prog.out().splitlines() == []
    # mixed flips
    flip_prog = _flip_array_to_prog((1, 0, 1, 0, 1, 1), [0, 1, 2, 3, 4, 5])
    assert flip_prog.out().splitlines() == [
        'RX(pi) 0',
        'RX(pi) 2',
        'RX(pi) 4',
        'RX(pi) 5'
    ]
    # flip all
    flip_prog = _flip_array_to_prog((1, 1, 1, 1, 1, 1), [0, 1, 2, 3, 4, 5])
    assert flip_prog.out().splitlines() == [
        'RX(pi) 0',
        'RX(pi) 1',
        'RX(pi) 2',
        'RX(pi) 3',
        'RX(pi) 4',
        'RX(pi) 5'
    ]


def test_symmetrization():
    prog = Program(I(0), I(1))
    meas_qubits = [0, 1]
    # invalid input if symm_type < -1 or > 3
    with pytest.raises(ValueError):
        _, _ = _symmetrization(prog, meas_qubits, symm_type=-2)
    with pytest.raises(ValueError):
        _, _ = _symmetrization(prog, meas_qubits, symm_type=4)
    # exhaustive symm
    sym_progs, flip_array = _symmetrization(prog, meas_qubits, symm_type=-1)
    assert sym_progs[0].out().splitlines() == ['I 0', 'I 1']
    assert sym_progs[1].out().splitlines() == ['I 0', 'I 1', 'RX(pi) 1']
    assert sym_progs[2].out().splitlines() == ['I 0', 'I 1', 'RX(pi) 0']
    assert sym_progs[3].out().splitlines() == ['I 0', 'I 1', 'RX(pi) 0', 'RX(pi) 1']
    right = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]
    assert all([np.allclose(x, y) for x, y in zip(flip_array, right)])
    # strength 0 i.e. no symm
    sym_progs, flip_array = _symmetrization(prog, meas_qubits, symm_type=-1)
    assert sym_progs[0].out().splitlines() == ['I 0', 'I 1']
    right = [np.array([0, 0])]
    assert all([np.allclose(x, y) for x, y in zip(flip_array, right)])
    # strength 1
    sym_progs, flip_array = _symmetrization(prog, meas_qubits, symm_type=1)
    assert sym_progs[0].out().splitlines() == ['I 0', 'I 1']
    assert sym_progs[1].out().splitlines() == ['I 0', 'I 1', 'RX(pi) 0', 'RX(pi) 1']
    right = [np.array([0, 0]), np.array([1, 1])]
    assert all([np.allclose(x, y) for x, y in zip(flip_array, right)])
    # strength 2
    sym_progs, flip_array = _symmetrization(prog, meas_qubits, symm_type=2)
    assert sym_progs[0].out().splitlines() == ['I 0', 'I 1']
    assert sym_progs[1].out().splitlines() == ['I 0', 'I 1', 'RX(pi) 0']
    assert sym_progs[2].out().splitlines() == ['I 0', 'I 1', 'RX(pi) 1']
    assert sym_progs[3].out().splitlines() == ['I 0', 'I 1', 'RX(pi) 0', 'RX(pi) 1']
    right = [np.array([0, 0]), np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]
    assert all([np.allclose(x, y) for x, y in zip(flip_array, right)])
    # strength 3
    sym_progs, flip_array = _symmetrization(prog, meas_qubits, symm_type=3)
    assert sym_progs[0].out().splitlines() == ['I 0', 'I 1', 'RX(pi) 0', 'RX(pi) 1']
    assert sym_progs[1].out().splitlines() == ['I 0', 'I 1', 'RX(pi) 0']
    assert sym_progs[2].out().splitlines() == ['I 0', 'I 1']
    assert sym_progs[3].out().splitlines() == ['I 0', 'I 1', 'RX(pi) 1']
    right = [np.array([1, 1]), np.array([1, 0]), np.array([0, 0]), np.array([0, 1])]
    assert all([np.allclose(x, y) for x, y in zip(flip_array, right)])


def test_construct_orthogonal_array():
    # check for valid inputs
    with pytest.raises(ValueError):
        _construct_orthogonal_array(3, strength=-1)
    with pytest.raises(ValueError):
        _construct_orthogonal_array(3, strength=4)
    with pytest.raises(ValueError):
        _construct_orthogonal_array(3, strength=100)


def test_construct_strength_three_orthogonal_array():
    # This is test is table 1.3 in [OATA]. Next to the np.array below the "line" number refers to
    # the row in table 1.3. It is not important that the rows are switched! Specifically
    #  "A permutation of the runs or factors in an orthogonal array results in an orthogonal
    #  array with the same parameters." page 27 of [OATA].
    #
    # [OATA] Orthogonal Arrays Theory and Applications
    #        Hedayat, Sloane, Stufken
    #        Springer, 1999
    answer = np.array([[1, 1, 1, 1],   # line 8
                       [1, 0, 1, 0],   # line 6
                       [1, 1, 0, 0],   # line 7
                       [1, 0, 0, 1],   # line 5
                       [0, 0, 0, 0],   # line 1
                       [0, 1, 0, 1],   # line 3
                       [0, 0, 1, 1],   # line 2
                       [0, 1, 1, 0]])  # line 4
    assert np.allclose(_construct_strength_three_orthogonal_array(4), answer)


def test_construct_strength_two_orthogonal_array():
    # This is example 1.5 in [OATA]. Next to the np.array below the "line" number refers to
    # the row in example 1.5.
    answer = np.array([[0, 0, 0],   # line 1
                       [1, 0, 1],   # line 3
                       [0, 1, 1],   # line 2
                       [1, 1, 0]])  # line 4
    assert np.allclose(_construct_strength_two_orthogonal_array(3), answer)


def test_measure_bitstrings(forest):
    device = NxDevice(nx.complete_graph(2))
    qc_pyqvm = QuantumComputer(
        name='testy!',
        qam=PyQVM(n_qubits=2),
        device=device,
        compiler=DummyCompiler()
    )
    qc_forest = QuantumComputer(
        name='testy!',
        qam=QVM(connection=forest, gate_noise=[0.00] * 3),
        device=device,
        compiler=DummyCompiler()
    )
    prog = Program(I(0), I(1))
    meas_qubits = [0, 1]
    sym_progs, flip_array = _symmetrization(prog, meas_qubits, symm_type=-1)
    results = _measure_bitstrings(qc_pyqvm, sym_progs, meas_qubits, num_shots=1)
    # test with pyQVM
    answer = [np.array([[0, 0]]), np.array([[0, 1]]), np.array([[1, 0]]), np.array([[1, 1]])]
    assert all([np.allclose(x, y) for x, y in zip(results, answer)])
    # test with regular QVM
    results = _measure_bitstrings(qc_forest, sym_progs, meas_qubits, num_shots=1)
    assert all([np.allclose(x, y) for x, y in zip(results, answer)])


def test_consolidate_symmetrization_outputs():
    flip_arrays = [np.array([[0, 0]]), np.array([[0, 1]]), np.array([[1, 0]]), np.array([[1, 1]])]
    # if results = flip_arrays should be a matrix of zeros
    ans1 = np.array([[0, 0],
                     [0, 0],
                     [0, 0],
                     [0, 0]])
    assert np.allclose(_consolidate_symmetrization_outputs(flip_arrays, flip_arrays), ans1)
    results = [np.array([[0, 0]]), np.array([[0, 0]]), np.array([[0, 0]]), np.array([[0, 0]])]
    # results are all zero then output should be
    ans2 = np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]])
    assert np.allclose(_consolidate_symmetrization_outputs(results, flip_arrays), ans2)


def test_check_min_num_trials_for_symmetrized_readout():
    # trials = -2 should get bumped up to 4 trials
    assert _check_min_num_trials_for_symmetrized_readout(num_qubits=2, trials=-2, symm_type=-1) == 4
    # can't have symm_type < -2 or > 3
    with pytest.raises(ValueError):
        _check_min_num_trials_for_symmetrized_readout(num_qubits=2, trials=-2, symm_type=-2)
    with pytest.raises(ValueError):
        _check_min_num_trials_for_symmetrized_readout(num_qubits=2, trials=-2, symm_type=4)


def test_readout_symmetrization(forest):
    device = NxDevice(nx.complete_graph(3))
    noise_model = decoherence_noise_with_asymmetric_ro(gates=gates_in_isa(device.get_isa()))
    qc = QuantumComputer(
        name='testy!',
        qam=QVM(connection=forest, noise_model=noise_model),
        device=device,
        compiler=DummyCompiler()
    )

    prog = Program(I(0), X(1),
                   MEASURE(0, 0),
                   MEASURE(1, 1))
    prog.wrap_in_numshots_loop(1000)

    bs1 = qc.run(prog)
    avg0_us = np.mean(bs1[:, 0])
    avg1_us = 1 - np.mean(bs1[:, 1])
    diff_us = avg1_us - avg0_us
    assert diff_us > 0.03

    bs2 = qc.run_symmetrized_readout(prog, 1000)
    avg0_s = np.mean(bs2[:, 0])
    avg1_s = 1 - np.mean(bs2[:, 1])
    diff_s = avg1_s - avg0_s
    assert diff_s < 0.05


def test_run_symmetrized_readout_error():
    # This test checks if the function runs for any possible input on a small number of qubits.
    # Locally this test was run on all 8 qubits, but it was slow.
    qc = get_qc("8q-qvm")
    sym_type_vec = [-1, 0, 1, 2, 3]
    prog_vec = [Program(I(x) for x in range(0, 3))[0:n] for n in range(0, 4)]
    trials_vec = list(range(0, 5))
    for prog, trials, sym_type in itertools.product(prog_vec, trials_vec, sym_type_vec):
        print(qc.run_symmetrized_readout(prog, trials, sym_type))