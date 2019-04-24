from numpy import random
from pyquil import Program
from pyquil.gates import X, Y
from pyquil.operator_estimation import ExperimentSetting, zeros_state

from forest.benchmarking.stratified_experiment import *
from forest.benchmarking.stratified_experiment import _group_allowed_types, _group_by_depth, \
    _get_simultaneous_groups, _partition_settings
from forest.benchmarking.utils import all_pauli_z_terms, all_pauli_terms, str_to_pauli_term


def test_merge_sequences():
    random.seed(0)
    seq0 = [Program(X(0)), Program(Y(0)), Program(X(0))]
    seq1 = [Program(X(1)), Program(Y(1)), Program(Y(1))]
    assert merge_sequences([seq0, seq1]) == [Program(X(0), X(1)),
                                             Program(Y(0), Y(1)),
                                             Program(X(0), Y(1))]


def test_group_allowed_types():
    depth = 0
    sequence = (Program(), )
    qubits = [0]
    z_ops = all_pauli_z_terms(qubits)
    settings = [ExperimentSetting(zeros_state(qubits), op) for op in z_ops]
    lyr = Layer(depth, sequence, settings, qubits)
    expt1 = StratifiedExperiment([lyr], settings, 'RB')
    expt2 = StratifiedExperiment([lyr], settings, 'URB')
    expt3 = StratifiedExperiment([lyr], settings, 'T1')
    allowed_parallel_types = [['RB', 'URB']]
    expts = _group_allowed_types([expt1, expt2, expt3],
                                 allowed_parallel_types=allowed_parallel_types)
    for group in expts:
        if len(group) == 2:
            assert group[0].expt_type in ['RB', 'URB']
            assert group[1].expt_type in ['RB', 'URB']
        else:
            assert len(group) == 1
            assert group[0].expt_type == 'T1'


def test_group_by_depth():
    sequence = (Program(),)
    qubits = [0, 1]
    z_ops = all_pauli_z_terms(qubits)
    settings = [ExperimentSetting(zeros_state(qubits), op) for op in z_ops]
    lyr0 = Layer(0, sequence, settings, qubits)
    lyr0_2 = Layer(0, sequence, settings, qubits)
    lyr1 = Layer(1, sequence, settings, qubits)
    lyr2 = Layer(2, sequence, settings, qubits)
    expt1 = StratifiedExperiment([lyr0, lyr1], qubits, 'RB')
    expt2 = StratifiedExperiment([lyr0_2, lyr2], qubits, 'URB')
    expt3 = StratifiedExperiment([lyr0], qubits, 'T1')
    expt4 = StratifiedExperiment([lyr1], qubits, 'T1')
    expts = [[expt1, expt2], [expt3, expt4]]
    depth_groups = _group_by_depth(expts)

    target = [[[lyr0, lyr0_2], [lyr0]], [[lyr1], [lyr1]], [[lyr2]]]
    assert depth_groups == target


def test_get_simultaneous_groups():
    qubits = [0, 1]
    sequence = (Program(),)
    seq2 = (Program(X(3)), )
    ops = all_pauli_z_terms(qubits)
    settings = [ExperimentSetting(zeros_state(qubits), op) for op in ops]
    lyr01 = Layer(0, sequence, settings, qubits)
    lyr01_same = Layer(0, sequence, settings[2], qubits)
    lyr01_diff = Layer(0, seq2, settings, qubits)
    settings3 = [ExperimentSetting(zeros_state([3]), op) for op in ops]
    lyr3 = Layer(0, seq2, settings3, (3,))
    layers = [lyr01, lyr01_diff, lyr01_same, lyr3]
    settings_groups = _get_simultaneous_groups(layers)
    target = [[lyr01, lyr01_same, lyr3], [lyr01_diff]]
    assert settings_groups == target


def test_partition_settings():
    qubits = [0]
    sequence = (Program(),)
    z_ops = all_pauli_z_terms(qubits)
    z_settings = [ExperimentSetting(zeros_state(qubits), op) for op in z_ops]
    pauli_ops = all_pauli_terms(qubits)
    p_settings = [ExperimentSetting(zeros_state(qubits), op) for op in pauli_ops]
    lyrz = Layer(0, sequence, z_settings, qubits)
    lyrp = Layer(0, sequence, p_settings, qubits)
    settings_groups, assoc_layers = _partition_settings([lyrz, lyrp])
    xop = str_to_pauli_term('X', qubits)
    yop = str_to_pauli_term('Y', qubits)
    zop = str_to_pauli_term('Z', qubits)
    iop = str_to_pauli_term('I', qubits)
    z_setting = ExperimentSetting(zeros_state(qubits), zop)
    i_setting = ExperimentSetting(zeros_state(qubits), iop)
    target_groups = [[i_setting, i_setting, z_setting, z_setting],
                     [ExperimentSetting(zeros_state(qubits), xop)],
                     [ExperimentSetting(zeros_state(qubits), yop)]]
    target_layers = [[lyrz, lyrp]*2, [lyrp], [lyrp]]
    assert settings_groups == target_groups
    assert assoc_layers == target_layers
