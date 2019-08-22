import numpy as np
from forest.benchmarking.qubit_spectroscopy import MICROSECOND, do_t1_or_t2


def test_do_t2(qvm):
    qubits = [0, 1]
    stop_time = 60 * MICROSECOND
    num_points = 15
    times = np.linspace(0, stop_time, num_points)
    t2_star_by_qubit = do_t1_or_t2(qvm, qubits, times, kind='t2_star', show_progress_bar=True)[0]
    assert qubits[0] in t2_star_by_qubit.keys()
    assert qubits[1] in t2_star_by_qubit.keys()
    assert t2_star_by_qubit[qubits[0]] > 0

    t2_echo_by_qubit = do_t1_or_t2(qvm, qubits, times, kind='t2_echo', show_progress_bar=True)[0]
    assert qubits[0] in t2_echo_by_qubit.keys()
    assert qubits[1] in t2_echo_by_qubit.keys()
    assert t2_echo_by_qubit[qubits[0]] > 0