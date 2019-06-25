import functools
import itertools
import json
import logging
import re
import sys
import warnings
from json import JSONEncoder
from operator import mul
from typing import List, Union, Iterable, Tuple, Dict, Callable
from copy import copy

import numpy as np
from math import pi
from scipy.stats import beta
import networkx as nx
from networkx.algorithms.approximation.clique import clique_removal
from pyquil import Program
from pyquil.api import QuantumComputer
from pyquil.gates import RX, RZ, MEASURE
from pyquil.paulis import PauliTerm, sI, is_identity

from forest.benchmarking.compilation import basic_compile, _RY
from forest.benchmarking.utils import transform_bit_moments_to_pauli

if sys.version_info < (3, 7):
    from pyquil.external.dataclasses import dataclass
else:
    from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _OneQState:
    """
    A description of a named one-qubit quantum state.
    This can be used to generate pre-rotations for quantum process tomography. For example,
    X0_14 will generate the +1 eigenstate of the X operator on qubit 14. X1_14 will generate the
    -1 eigenstate. SIC0_14 will generate the 0th SIC-basis state on qubit 14.
    """
    label: str
    index: int
    qubit: int

    def __str__(self):
        return f'{self.label}{self.index}_{self.qubit}'

    @classmethod
    def from_str(cls, s):
        ma = re.match(r'\s*(\w+)(\d+)_(\d+)\s*', s)
        if ma is None:
            raise ValueError(f"Couldn't parse '{s}'")
        return _OneQState(
            label=ma.group(1),
            index=int(ma.group(2)),
            qubit=int(ma.group(3)),
        )


@dataclass(frozen=True)
class TensorProductState:
    """
    A description of a multi-qubit quantum state that is a tensor product of many _OneQStates
    states.
    """
    states: Tuple[_OneQState]

    def __init__(self, states=None):
        if states is None:
            states = tuple()
        object.__setattr__(self, 'states', tuple(states))

    def __mul__(self, other):
        return TensorProductState(self.states + other.states)

    def __str__(self):
        return ' * '.join(str(s) for s in self.states)

    def __repr__(self):
        return f'TensorProductState[{self}]'

    def __getitem__(self, qubit):
        """Return the _OneQState at the given qubit."""
        for oneq_state in self.states:
            if oneq_state.qubit == qubit:
                return oneq_state
        raise IndexError()

    def __iter__(self):
        yield from self.states

    def __len__(self):
        return len(self.states)

    def states_as_set(self):
        return frozenset(self.states)

    def __eq__(self, other):
        if not isinstance(other, TensorProductState):
            return False

        return self.states_as_set() == other.states_as_set()

    def __hash__(self):
        return hash(self.states_as_set())

    @classmethod
    def from_str(cls, s):
        if s == '':
            return TensorProductState()
        return TensorProductState(tuple(_OneQState.from_str(x) for x in s.split('*')))


def SIC0(q):
    return TensorProductState((_OneQState('SIC', 0, q),))


def SIC1(q):
    return TensorProductState((_OneQState('SIC', 1, q),))


def SIC2(q):
    return TensorProductState((_OneQState('SIC', 2, q),))


def SIC3(q):
    return TensorProductState((_OneQState('SIC', 3, q),))


def plusX(q):
    return TensorProductState((_OneQState('X', 0, q),))


def minusX(q):
    return TensorProductState((_OneQState('X', 1, q),))


def plusY(q):
    return TensorProductState((_OneQState('Y', 0, q),))


def minusY(q):
    return TensorProductState((_OneQState('Y', 1, q),))


def plusZ(q):
    return TensorProductState((_OneQState('Z', 0, q),))


def minusZ(q):
    return TensorProductState((_OneQState('Z', 1, q),))


def zeros_state(qubits: Iterable[int]):
    return TensorProductState(_OneQState('Z', 0, q) for q in qubits)


@dataclass(frozen=True, init=False)
class ExperimentSetting:
    """
    Input and output settings for an ObservablesExperiment.
    Many near-term quantum algorithms and QCVV protocols take the following form:
     - Start in a pauli state
     - Do some interesting quantum circuit (e.g. prepare some ansatz)
     - Measure the output of the circuit w.r.t. expectations of Pauli observables.
    Where we typically use a large number of (start, measure) pairs but keep the quantum circuit
    program consistent. This class represents the (start, measure) pairs. Typically a large
    number of these :py:class:`ExperimentSetting` objects will be created and grouped into
    a :py:class:`ObservablesExperiment`.
    """
    in_state: TensorProductState
    observable: PauliTerm

    def __init__(self, in_state: TensorProductState, observable: PauliTerm):

        object.__setattr__(self, 'in_state', in_state)
        object.__setattr__(self, 'observable', observable)

    def __str__(self):
        return f'{self.in_state}→{self.observable.compact_str()}'

    def __repr__(self):
        return f'ExperimentSetting[{self}]'

    def serializable(self):
        return str(self)

    @classmethod
    def from_str(cls, s: str):
        """The opposite of str(expt)"""
        instr, outstr = s.split('→')
        return ExperimentSetting(in_state=TensorProductState.from_str(instr),
                                 observable=PauliTerm.from_compact_str(outstr))


def _abbrev_program(program: Program, max_len=10):
    """Create an abbreviated string representation of a Program.
    This will join all instructions onto a single line joined by '; '. If the number of
    instructions exceeds ``max_len``, some will be excluded from the string representation.
    """
    program_lines = program.out().splitlines()
    if max_len is not None and len(program_lines) > max_len:
        first_n = max_len // 2
        last_n = max_len - first_n
        excluded = len(program_lines) - max_len
        program_lines = (program_lines[:first_n] + [f'... {excluded} instrs not shown ...']
                         + program_lines[-last_n:])

    return '; '.join(program_lines)


class ObservablesExperiment:
    """
    Many near-term quantum algorithms involve:
     - some limited state preparation, e.g. prepare a Pauli eigenstate
     - enacting a quantum process (like in tomography) or preparing a variational ansatz state
       (like in VQE) with some circuit.
     - Measure the output of the circuit w.r.t. expectations of Pauli observables
    Where we typically use a large number of (state_prep, measure_observable) pairs but keep the
    quantum circuit program consistent. This class stores the circuit program as a
    :py:class:`~pyquil.Program` and maintains a list of :py:class:`ExperimentSetting` objects
    which each represent a (state_prep, measure_observable) pair.
    Settings diagonalized by a shared tensor product basis (TPB) can (optionally) be estimated
    simultaneously. Therefore, this class is backed by a list of list of ExperimentSettings.
    Settings sharing an inner list will be estimated simultaneously. If you don't want this,
    provide a list of length-1-lists. As a convenience, if you pass a 1D list to the constructor
    will expand it to a list of length-1-lists.
    This class will not group settings for you. Please see :py:func:`group_settings` for
    a function that will automatically process a ObservablesExperiment to group Experiments sharing
    a TPB.
    """

    def __init__(self,
                 settings: Union[List[ExperimentSetting], List[List[ExperimentSetting]]],
                 program: Program,
                 qubits: List[int] = None):
        if len(settings) == 0:
            settings = []
        else:
            if isinstance(settings[0], ExperimentSetting):
                # convenience wrapping in lists of length 1
                settings = [[expt] for expt in settings]

        self._settings = settings  # type: List[List[ExperimentSetting]]
        self.program = program
        if qubits is not None:
            warnings.warn("The 'qubits' parameter has been deprecated and will be removed"
                          "in a future release of pyquil")
        self.qubits = qubits

    def __len__(self):
        return len(self._settings)

    def __getitem__(self, item):
        return self._settings[item]

    def __setitem__(self, key, value):
        self._settings[key] = value

    def __delitem__(self, key):
        self._settings.__delitem__(key)

    def __iter__(self):
        yield from self._settings

    def __reversed__(self):
        yield from reversed(self._settings)

    def __contains__(self, item):
        return item in self._settings

    def append(self, expts):
        if not isinstance(expts, list):
            expts = [expts]
        return self._settings.append(expts)

    def count(self, expt):
        return self._settings.count(expt)

    def index(self, expt, start=None, stop=None):
        return self._settings.index(expt, start, stop)

    def extend(self, expts):
        return self._settings.extend(expts)

    def insert(self, index, expt):
        return self._settings.insert(index, expt)

    def pop(self, index=None):
        return self._settings.pop(index)

    def remove(self, expt):
        return self._settings.remove(expt)

    def reverse(self):
        return self._settings.reverse()

    def sort(self, key=None, reverse=False):
        return self._settings.sort(key, reverse)

    def setting_strings(self):
        yield from ('{i}: {st_str}'.format(i=i, st_str=', '.join(str(setting)
                                                                 for setting in settings))
                    for i, settings in enumerate(self._settings))

    def settings_string(self, abbrev_after=None):
        setting_strs = list(self.setting_strings())
        if abbrev_after is not None and len(setting_strs) > abbrev_after:
            first_n = abbrev_after // 2
            last_n = abbrev_after - first_n
            excluded = len(setting_strs) - abbrev_after
            setting_strs = (setting_strs[:first_n] + [f'... {excluded} not shown ...',
                                                      '... use e.settings_string() for all ...']
                            + setting_strs[-last_n:])
        return '\n'.join(setting_strs)

    def __str__(self):
        return _abbrev_program(self.program) + '\n' + self.settings_string(abbrev_after=20)

    def serializable(self):
        return {
            'type': 'ObservablesExperiment',
            'settings': self._settings,
            'program': self.program.out(),
        }

    def __eq__(self, other):
        if not isinstance(other, ObservablesExperiment):
            return False
        return self.serializable() == other.serializable()


class OperatorEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, ExperimentSetting):
            return o.serializable()
        if isinstance(o, ObservablesExperiment):
            return o.serializable()
        if isinstance(o, ExperimentResult):
            return o.serializable()
        return o


def to_json(fn, obj):
    """Convenience method to save forest.benchmarking.observable_estimation objects as a JSON file.
    See :py:func:`read_json`.
    """
    with open(fn, 'w') as f:
        json.dump(obj, f, cls=OperatorEncoder, indent=2, ensure_ascii=False)
    return fn


def _operator_object_hook(obj):
    if 'type' in obj and obj['type'] == 'ObservablesExperiment':
        return ObservablesExperiment([[ExperimentSetting.from_str(s) for s in settings]
                                     for settings in obj['settings']],
                                    program=Program(obj['program']))
    return obj


def read_json(fn):
    """Convenience method to read forest.benchmarking.observable_estimation objects from a JSON file.
    See :py:func:`to_json`.
    """
    with open(fn) as f:
        return json.load(f, object_hook=_operator_object_hook)


def _one_q_sic_prep(index, qubit):
    """Prepare the index-th SIC basis state."""
    if index == 0:
        return Program()

    theta = 2 * np.arccos(1 / np.sqrt(3))
    zx_plane_rotation = Program([
        RX(-pi / 2, qubit),
        RZ(theta - pi, qubit),
        RX(-pi / 2, qubit),
    ])

    if index == 1:
        return zx_plane_rotation

    elif index == 2:
        return zx_plane_rotation + RZ(-2 * pi / 3, qubit)

    elif index == 3:
        return zx_plane_rotation + RZ(2 * pi / 3, qubit)

    raise ValueError(f'Bad SIC index: {index}')


def _one_q_pauli_prep(label, index, qubit):
    """Prepare the index-th eigenstate of the pauli operator given by label."""
    if index not in [0, 1]:
        raise ValueError(f'Bad Pauli index: {index}')

    if label == 'X':
        if index == 0:
            return Program(_RY(pi / 2, qubit))
        else:
            return Program(_RY(-pi / 2, qubit))

    elif label == 'Y':
        if index == 0:
            return Program(RX(-pi / 2, qubit))
        else:
            return Program(RX(pi / 2, qubit))

    elif label == 'Z':
        if index == 0:
            return Program()
        else:
            return Program(RX(pi, qubit))

    raise ValueError(f'Bad Pauli label: {label}')


def _one_q_state_prep(oneq_state: _OneQState):
    """Prepare a one qubit state.
    Either SIC[0-3], X[0-1], Y[0-1], or Z[0-1].
    """
    label = oneq_state.label
    if label == 'SIC':
        return _one_q_sic_prep(oneq_state.index, oneq_state.qubit)
    elif label in ['X', 'Y', 'Z']:
        return _one_q_pauli_prep(label, oneq_state.index, oneq_state.qubit)
    else:
        raise ValueError(f"Bad state label: {label}")


def _local_pauli_eig_meas(op, idx):
    """
    Generate gate sequence to measure in the eigenbasis of a Pauli operator, assuming
    we are only able to measure in the Z eigenbasis. (Note: The unitary operations of this
    Program are essentially the Hermitian conjugates of those in :py:func:`_one_q_pauli_prep`)
    """
    if op == 'X':
        return Program(_RY(-pi / 2, idx))
    elif op == 'Y':
        return Program(RX(pi / 2, idx))
    elif op == 'Z':
        return Program()
    raise ValueError(f'Unknown operation {op}')


def construct_tpb_graph(obs_expt: ObservablesExperiment):
    """
    Construct a graph where an edge signifies two settings are diagonal in a TPB.
    """
    g = nx.Graph()
    for groups in obs_expt:
        assert len(groups) == 1, 'already grouped?'
        setting = groups[0]

        if setting not in g:
            g.add_node(setting, count=1)
        else:
            g.nodes[setting]['count'] += 1

    for group1, group2 in itertools.combinations(obs_expt, r=2):
        sett1 = group1[0]
        sett2 = group2[0]

        if sett1 == sett2:
            continue

        max_weight_in = _max_weight_state([sett1.in_state, sett2.in_state])
        max_weight_out = _max_weight_operator([sett1.observable, sett2.observable])
        if max_weight_in is not None and max_weight_out is not None:
            g.add_edge(sett1, sett2)

    return g


def group_settings_clique_removal(experiment: ObservablesExperiment) -> ObservablesExperiment:
    """
    Group settings that are diagonal in a shared tensor product basis (TPB) to minimize number
    of QPU runs, using a graph clique removal algorithm.

    :param experiment: an ObservablesExperiment
    :return: a ObservablesExperiment with all the same settings, just grouped according to shared
        TPBs.
    """
    g = construct_tpb_graph(experiment)
    _, cliqs = clique_removal(g)
    new_cliqs = []
    for cliq in cliqs:
        new_cliq = []
        for sett in cliq:
            # duplicate `count` times
            new_cliq += [sett] * g.nodes[sett]['count']

        new_cliqs += [new_cliq]

    return ObservablesExperiment(new_cliqs, program=experiment.program)


def _max_weight_operator(ops: Iterable[PauliTerm]) -> Union[None, PauliTerm]:
    """
    Construct a PauliTerm operator by taking the non-identity single-qubit operator at each
    qubit position.

    This function will return ``None`` if the input operators do not share a natural tensor
    product basis.
    For example, the max_weight_operator of ["XI", "IZ"] is "XZ". Asking for the max weight
    operator of something like ["XI", "ZI"] will return None.
    """
    mapping = dict()  # type: Dict[int, str]
    for op in ops:
        for idx, op_str in op:
            if idx in mapping:
                if mapping[idx] != op_str:
                    return None
            else:
                mapping[idx] = op_str
    op = functools.reduce(mul, (PauliTerm(op, q) for q, op in mapping.items()), sI())
    return op


def _max_weight_state(states: Iterable[TensorProductState]) -> Union[None, TensorProductState]:
    """
    Construct a TensorProductState by taking the single-qubit state at each
    qubit position.

    This function will return ``None`` if the input states are not compatible
    For example, the max_weight_state of ["(+X, q0)", "(-Z, q1)"] is "(+X, q0; -Z q1)". Asking for
    the max weight state of something like ["(+X, q0)", "(+Z, q0)"] will return None.
    """
    mapping = dict()  # type: Dict[int, _OneQState]
    for state in states:
        for oneq_state in state.states:
            if oneq_state.qubit in mapping:
                if mapping[oneq_state.qubit] != oneq_state:
                    return None
            else:
                mapping[oneq_state.qubit] = oneq_state
    return TensorProductState(list(mapping.values()))


def _max_tpb_overlap(obs_expt: ObservablesExperiment):
    """
    Given an input ObservablesExperiment, provide a dictionary indicating which ExperimentSettings
    share a tensor product basis

    :param obs_expt: ObservablesExperiment, from which to group ExperimentSettings that share a tpb
        and can be run together
    :return: dictionary keyed with ExperimentSetting (specifying a tpb), and with each value being a
            list of ExperimentSettings (diagonal in that tpb)
    """
    # initialize empty dictionary
    diagonal_sets = {}
    # loop through ExperimentSettings of the ObservablesExperiment
    for expt_setting in obs_expt:
        # no need to group already grouped ObservablesExperiment
        assert len(expt_setting) == 1, 'already grouped?'
        expt_setting = expt_setting[0]
        # calculate max overlap of expt_setting with keys of diagonal_sets
        # keep track of whether a shared tpb was found
        found_tpb = False
        # loop through dict items
        for es, es_list in diagonal_sets.items():
            trial_es_list = es_list + [expt_setting]
            diag_in_term = _max_weight_state(expst.in_state for expst in trial_es_list)
            diag_out_term = _max_weight_operator(expst.observable for expst in trial_es_list)
            # max_weight_xxx returns None if the set of xxx's don't share a TPB, so the following
            # conditional is True if expt_setting can be inserted into the current es_list.
            if diag_in_term is not None and diag_out_term is not None:
                found_tpb = True
                assert len(diag_in_term) >= len(es.in_state), \
                    "Highest weight in-state can't be smaller than the given in-state"
                assert len(diag_out_term) >= len(es.observable), \
                    "Highest weight out-PauliTerm can't be smaller than the given out-PauliTerm"

                # update the diagonalizing basis (key of dict) if necessary
                if len(diag_in_term) > len(es.in_state) or len(diag_out_term) > len(es.observable):
                    del diagonal_sets[es]
                    new_es = ExperimentSetting(diag_in_term, diag_out_term)
                    diagonal_sets[new_es] = trial_es_list
                else:
                    diagonal_sets[es] = trial_es_list
                break

        if not found_tpb:
            # made it through entire dict without finding any ExperimentSetting with shared tpb,
            # so need to make a new item
            diagonal_sets[expt_setting] = [expt_setting]

    return diagonal_sets


def group_settings_greedy(obs_expt: ObservablesExperiment):
    """
    Greedy method to group ExperimentSettings in a given ObservablesExperiment

    :param obs_expt: ObservablesExperiment to group ExperimentSettings within
    :return: ObservablesExperiment, with grouped ExperimentSettings according to whether
        it consists of PauliTerms diagonal in the same tensor product basis
    """
    diag_sets = _max_tpb_overlap(obs_expt)
    grouped_expt_settings_list = list(diag_sets.values())
    grouped_obs_expt = ObservablesExperiment(grouped_expt_settings_list, program=obs_expt.program)
    return grouped_obs_expt


def group_settings(obs_expt: ObservablesExperiment,
                   method: str = 'greedy') -> ObservablesExperiment:
    """
    Group settings that are diagonal in a shared tensor product basis (TPB) to minimize number
    of QPU runs.
    Background
    ----------
    Given some PauliTerm operator, the 'natural' tensor product basis to
    diagonalize this term is the one which diagonalizes each Pauli operator in the
    product term-by-term.
    For example, X(1) * Z(0) would be diagonal in the 'natural' tensor product basis
    {(|0> +/- |1>)/Sqrt[2]} * {|0>, |1>}, whereas Z(1) * X(0) would be diagonal
    in the 'natural' tpb {|0>, |1>} * {(|0> +/- |1>)/Sqrt[2]}. The two operators
    commute but are not diagonal in each others 'natural' tpb (in fact, they are
    anti-diagonal in each others 'natural' tpb). This function tests whether two
    operators given as PauliTerms are both diagonal in each others 'natural' tpb.
    Note that for the given example of X(1) * Z(0) and Z(1) * X(0), we can construct
    the following basis which simultaneously diagonalizes both operators:
      -- |0>' = |0> (|+>) + |1> (|->)
      -- |1>' = |0> (|+>) - |1> (|->)
      -- |2>' = |0> (|->) + |1> (|+>)
      -- |3>' = |0> (-|->) + |1> (|+>)
    In this basis, X Z looks like diag(1, -1, 1, -1), and Z X looks like diag(1, 1, -1, -1).
    Notice however that this basis cannot be constructed with single-qubit operations, as each
    of the basis vectors are entangled states.
    Methods
    -------
    The "greedy" method will keep a running set of 'buckets' into which grouped ExperimentSettings
    will be placed. Each new ExperimentSetting considered is assigned to the first applicable
    bucket and a new bucket is created if there are no applicable buckets.
    The "clique-removal" method maps the term grouping problem onto Max Clique graph problem.
    This method constructs a NetworkX graph where an edge exists between two settings that
    share an nTPB and then uses networkx's algorithm for clique removal. This method can give
    you marginally better groupings in certain circumstances, but constructing the
    graph is pretty slow so "greedy" is the default.

    :param obs_expt: an ObservablesExperiment
    :param method: method used for grouping; the allowed methods are one of
        ['greedy', 'clique-removal']
    :return: an ObservablesExperiment with all the same settings, just grouped according to shared
        TPBs.
    """
    allowed_methods = ['greedy', 'clique-removal']
    assert method in allowed_methods, f"'method' should be one of {allowed_methods}."
    if method == 'greedy':
        return group_settings_greedy(obs_expt)
    elif method == 'clique-removal':
        return group_settings_clique_removal(obs_expt)


@dataclass(frozen=True)
class ExperimentResult:
    """
    An expectation and standard deviation for the measurement of one experiment setting
    in an ObservablesExperiment.

    In the case of readout error calibration, we also include
    expectation, standard deviation and count for the calibration results, as well as the
    expectation and standard deviation for the corrected results.
    """

    setting: ExperimentSetting
    expectation: Union[float, complex]
    total_counts: int
    std_err: Union[float, complex] = None
    raw_expectation: Union[float, complex] = None
    raw_std_err: float = None
    calibration_expectation: Union[float, complex] = None
    calibration_std_err: Union[float, complex] = None
    calibration_counts: int = None

    def __init__(self, setting: ExperimentSetting,
                 expectation: Union[float, complex],
                 total_counts: int,
                 stddev: Union[float, complex] = None,
                 std_err: Union[float, complex] = None,
                 raw_expectation: Union[float, complex] = None,
                 raw_stddev: float = None,
                 raw_std_err: float = None,
                 calibration_expectation: Union[float, complex] = None,
                 calibration_stddev: Union[float, complex] = None,
                 calibration_std_err: Union[float, complex] = None,
                 calibration_counts: int = None):

        object.__setattr__(self, 'setting', setting)
        object.__setattr__(self, 'expectation', expectation)
        object.__setattr__(self, 'total_counts', total_counts)
        object.__setattr__(self, 'raw_expectation', raw_expectation)
        object.__setattr__(self, 'calibration_expectation', calibration_expectation)
        object.__setattr__(self, 'calibration_counts', calibration_counts)

        if stddev is not None:
            warnings.warn("'stddev' has been renamed to 'std_err'")
            std_err = stddev
        object.__setattr__(self, 'std_err', std_err)

        if raw_stddev is not None:
            warnings.warn("'raw_stddev' has been renamed to 'raw_std_err'")
            raw_std_err = raw_stddev
        object.__setattr__(self, 'raw_std_err', raw_std_err)

        if calibration_stddev is not None:
            warnings.warn("'calibration_stddev' has been renamed to 'calibration_std_err'")
            calibration_std_err = calibration_stddev
        object.__setattr__(self, 'calibration_std_err', calibration_std_err)

    def get_stddev(self) -> Union[float, complex]:
        warnings.warn("'stddev' has been renamed to 'std_err'")
        return self.std_err

    def set_stddev(self, value: Union[float, complex]):
        warnings.warn("'stddev' has been renamed to 'std_err'")
        object.__setattr__(self, 'std_err', value)

    stddev = property(get_stddev, set_stddev)

    def get_raw_stddev(self) -> float:
        warnings.warn("'raw_stddev' has been renamed to 'raw_std_err'")
        return self.raw_std_err

    def set_raw_stddev(self, value: float):
        warnings.warn("'raw_stddev' has been renamed to 'raw_std_err'")
        object.__setattr__(self, 'raw_std_err', value)

    raw_stddev = property(get_raw_stddev, set_raw_stddev)

    def get_calibration_stddev(self) -> Union[float, complex]:
        warnings.warn("'calibration_stddev' has been renamed to 'calibration_std_err'")
        return self.calibration_std_err

    def set_calibration_stddev(self, value: Union[float, complex]):
        warnings.warn("'calibration_stddev' has been renamed to 'calibration_std_err'")
        object.__setattr__(self, 'calibration_std_err', value)

    calibration_stddev = property(get_calibration_stddev, set_calibration_stddev)

    def __str__(self):
        return f'{self.setting}: {self.expectation} +- {self.std_err}'

    def __repr__(self):
        return f'ExperimentResult[{self}]'

    def serializable(self):
        return {
            'type': 'ExperimentResult',
            'setting': self.setting,
            'expectation': self.expectation,
            'std_err': self.std_err,
            'total_counts': self.total_counts,
            'raw_expectation': self.raw_expectation,
            'raw_std_err': self.raw_std_err,
            'calibration_expectation': self.calibration_expectation,
            'calibration_std_err': self.calibration_std_err,
            'calibration_counts': self.calibration_counts,
        }


def generate_experiment_programs(obs_expt: ObservablesExperiment, active_reset: bool = False,
                                 use_basic_compile: bool = True) \
        -> Tuple[List[Program], List[List[int]]]:
    """
    Generate the programs necessary to estimate the observables in an ObservablesExperiment.

    Grouping of settings to be run in parallel, e.g. by a call to group_settings, should be
    done before this method is called.

    By default the program field of the input obs_expt is assumed to hold a program composed of
    gates which are either native quil gates or else can be compiled to native quil by the method
    basic_compile. If this is not the case, then use_basic_compile should be set to False and
    each returned program must be compiled before being executed on a QPU; NOTE however that
    compiling a program may change the qubit indices so meas_qubits might need to be re-ordered
    as well if the MEASURE instructions were not compiled with the program. For this reason it is
    recommended that obs_expt.program be compiled before this method is called. If one is careful
    then it is still possible to add the measurement instructions first and subsequently compile
    the programs before running, e.g. by setting use_compilation = True in the call to
    _measure_bitstrings (but compiling in this way could interfere with symmetrization).

    :param obs_expt: a single ObservablesExperiment to be translated to a series of programs that
        when run serially can be used to estimate each of obs_expt's observables.
    :param active_reset: whether or not to begin the program by actively resetting. If true,
        execution of each of the returned programs in a loop on the QPU will generally be faster.
    :param use_basic_compile: whether or not to call basic_compile on the programs after they are
        created. To run on a QPU it is necessary that programs use only native quil gates. See
        the warning above about setting use_basic_compile to false.
    :return: a list of programs along with a corresponding list of the groups of qubits that are
        measured by that program. The returned programs may be run on a qc after measurement
        instructions are added for the corresponding group of qubits in meas_qubits -- see
        estimate_observables and _measure_bitstrings for possible usage.
    """
    # Outer loop over a collection of grouped settings for which we can simultaneously estimate.
    programs = []
    meas_qubits = []
    for settings in obs_expt:

        # Prepare a state according to the amalgam of all setting.in_state
        total_prog = Program()
        if active_reset:
            total_prog += RESET()
        max_weight_in_state = _max_weight_state(setting.in_state for setting in settings)
        if max_weight_in_state is None:
            raise ValueError('Input states are not compatible. Re-group the experiment settings '
                             'so that groups of parallel settings have compatible input states.')
        for oneq_state in max_weight_in_state.states:
            total_prog += _one_q_state_prep(oneq_state)

        # Add in the program
        total_prog += obs_expt.program

        # Prepare for measurement state according to setting.observable
        max_weight_out_op = _max_weight_operator(setting.observable for setting in settings)
        if max_weight_out_op is None:
            raise ValueError('Observables not compatible. Re-group the experiment settings '
                             'so that groups of parallel settings have compatible observables.')
        for qubit, op_str in max_weight_out_op:
            total_prog += _local_pauli_eig_meas(op_str, qubit)

        if use_basic_compile:
            programs.append(basic_compile(total_prog))
        else:
            programs.append(total_prog)

        meas_qubits.append(max_weight_out_op.get_qubits())
    return programs, meas_qubits


def _flip_array_to_prog(flip_array: Tuple[bool], qubits: List[int]) -> Program:
    """
    Generate a pre-measurement program that flips the qubit state according to the flip_array of
    bools.

    This is used, for example, in exhaustive_symmetrization to produce programs which flip a
    select subset of qubits immediately before measurement.

    :param flip_array: tuple of booleans specifying whether the qubit in the corresponding index
        should be flipped or not.
    :param qubits: list specifying the qubits in order corresponding to the flip_array
    :return: Program which flips each qubit (i.e. instructs RX(pi, q)) according to the flip_array.
    """
    assert len(flip_array) == len(qubits), "Mismatch of qubits and operations"
    prog = Program()
    for qubit, flip_output in zip(qubits, flip_array):
        if flip_output == 0:
            continue
        elif flip_output == 1:
            prog += Program(RX(pi, qubit))
        else:
            raise ValueError("flip_bools should only consist of 0s and/or 1s")
    return prog


def exhaustive_symmetrization(programs: List[Program], meas_qubits: List[List[int]]) \
        -> Tuple[List[Program], List[List[int]], List[Tuple[bool]], List[int]]:
    """
    For each program in the input programs generate new programs which flip the measured qubits
    in every possible combination in order to symmetrize readout.

    The expanded list of programs is returned along with a correspondingly expanded list of
    meas_qubits, a list of bools which indicates which qubits are flipped in each program,
    and a list of indices which records from which program in the input programs list each
    symmetrized program originated.

    :param programs: a list of programs each of which will be symmetrized.
    :param meas_qubits: the corresponding groups of measurement qubits for the input list of
        programs. Only these qubits will be symmetrized over, even if the program acts on other
        qubits.
    :return: a list of symmetrized programs, the corresponding measurement qubits,
        the corresponding array of bools indicating which qubits were flipped,
        and a corresponding list of indices specifying the generating `program'
    """
    assert len(programs) == len(meas_qubits), 'mismatch of programs and qubits; must know which ' \
                                              'qubits are being measured to symmetrize them.'

    symm_programs = []
    symm_meas_qs = []
    prog_groups = []
    flip_arrays = []

    for idx, (prog, meas_qs) in enumerate(zip(programs, meas_qubits)):
        for flip_array in itertools.product([0, 1], repeat=len(meas_qs)):
            total_prog_symm = prog.copy()
            prog_symm = _flip_array_to_prog(flip_array, meas_qs)
            total_prog_symm += prog_symm
            symm_programs.append(total_prog_symm)
            symm_meas_qs.append(meas_qs)
            prog_groups.append(idx)
            flip_arrays.append(flip_array)

    return symm_programs, symm_meas_qs, flip_arrays, prog_groups


def _measure_bitstrings(qc: QuantumComputer, programs: List[Program], meas_qubits: List[List[int]],
                        num_shots = 500, use_compiler = False) -> List[np.ndarray]:
    """
    Wrapper for appending measure instructions onto each program, running the program,
    and accumulating the resulting bitarrays.

    By default each program is assumed to be native quil.

    :param qc: a quantum computer object on which to run each program
    :param programs: a list of programs to run
    :param meas_qubits: groups of qubits to measure for each program
    :param num_shots: the number of shots to run for each program
    :return: a len(programs) long list of num_shots by num_meas_qubits bit arrays of results for
        each program.
    """
    assert len(programs) == len(meas_qubits), 'The qubits to measure must be specified for each ' \
                                              'program, one list of qubits per program.'

    results = []
    for program, qubits in zip(programs, meas_qubits):
        if len(qubits) == 0:
            # corresponds to measuring identity; no program needs to be run.
            results.append(np.array([[]]))
            continue
        # copy the program so the original is not mutated
        prog = program.copy()
        ro = prog.declare('ro', 'BIT', len(qubits))
        for idx, q in enumerate(qubits):
            prog += MEASURE(q, ro[idx])

        prog.wrap_in_numshots_loop(num_shots)
        if use_compiler:
            prog = qc.quil_to_native_quil(prog)
        exe = qc.compiler.native_quil_to_executable(prog)
        shots = qc.run(exe)
        results.append(shots)
    return results


def shots_to_obs_moments(bitarray: np.ndarray, qubits: List[int], observable: PauliTerm,
                         use_beta_dist_unbiased_prior: bool = False) -> Tuple[float, float]:
    """
    Calculate the mean and variance of the given observable based on the bitarray of results.

    :param bitarray: results from running `qc.run`, a 2D num_shots by num_qubits array.
    :param qubits: list of qubits in order corresponding to the bitarray results.
    :param observable: the observable whose moments are calculated from the shot data
    :param use_beta_dist_unbiased_prior: if true then the mean and variance are estimated from a
        beta distribution that incorporates an unbiased Bayes prior. This precludes var = 0.
    :return: tuple specifying (mean, variance)
    """
    coeff = complex(observable.coefficient)
    if not np.isclose(coeff.imag, 0):
        raise ValueError(f"The coefficient of an observable should not be complex.")
    coeff = coeff.real

    obs_qubits = [q for q, _ in observable]
    # Identify classical register indices to select
    idxs = [idx for idx, q in enumerate(qubits) if q in obs_qubits]

    if len(idxs) == 0: # identity term
        return coeff, 0

    assert bitarray.shape[1] == len(qubits), 'qubits should label each column of the bitarray'

    # Pick columns corresponding to qubits with a non-identity out_operation
    obs_strings = bitarray[:, idxs]
    # Transform bits to eigenvalues; ie (+1, -1)
    my_obs_strings = 1 - 2 * obs_strings
    # Multiply row-wise to get operator values.
    obs_vals = np.prod(my_obs_strings, axis=1)

    if use_beta_dist_unbiased_prior:
        # For binary classified data with N counts of + and M counts of -, these can be estimated
        # using the mean and variance of the beta distribution beta(N+1, M+1) where the +1 is used
        # to incorporate an unbiased Bayes prior.
        plus_array = obs_vals == 1
        n_minus, n_plus = np.bincount(plus_array,  minlength=2)
        bernoulli_mean = beta.mean(n_plus + 1, n_minus + 1)
        bernoulli_var = beta.var(n_plus + 1, n_minus + 1)
        obs_mean, obs_var = transform_bit_moments_to_pauli(bernoulli_mean, bernoulli_var)
        obs_mean *= coeff
        obs_var *= coeff**2
    else:
        obs_vals = coeff * obs_vals
        obs_mean = np.mean(obs_vals).item()
        obs_var = np.var(obs_vals).item() / len(bitarray)

    return obs_mean, obs_var


def consolidate_symmetrization_outputs(outputs: List[np.ndarray], flip_arrays: List[Tuple[bool]],
                                       groups: List[int]) -> List[np.ndarray]:
    """
    Given bitarray results from a series of symmetrization programs, appropriately flip output
    bits and consolidate results into new bitarrays.

    :param outputs: a list of the raw bitarrays resulting from running a list of symmetrized
        programs; for example, the results returned from _measure_bitstrings
    :param flip_arrays: a list of boolean arrays in one-to-one correspondence with the list of
        outputs indicating which qubits where flipped before each bitarray was measured.
    :param groups: the group from which each symmetrized program was generated. E.g. if only one
        program was symmetrized then groups is simply [0] * len(outputs). The length of the
        returned consolidated outputs is exactly the number of distinct integers in groups.
    :return: a list of the consolidated bitarray outputs which can be treated as the symmetrized
        outputs of the original programs passed into a symmetrization method. See
        estimate_observables for example usage.
    """
    assert len(outputs) == len(groups) == len(flip_arrays)

    output = {group: [] for group in set(groups)}
    for bitarray, group, flip_array in zip(outputs, groups, flip_arrays):
        if len(flip_array) == 0:
            # happens when measuring identity.
            # TODO: better way of handling identity measurement? (in _measure_bitstrings too)
            output[group].append(bitarray)
        else:
            output[group].append(bitarray ^ flip_array)

    return [np.vstack(output[group]) for group in sorted(list(set(groups)))]


def estimate_observables(qc: QuantumComputer, obs_expt: ObservablesExperiment,
                         num_shots: int = 500, symmetrization_method: Callable = None,
                         active_reset: bool = False) -> Iterable[ExperimentResult]:
    """
    Standard wrapper for estimating the observables in an ObservableExperiment.

    Because of the use of default parameters for _measure_bitstrings, this method assumes the
    program in obs_expt can be compiled to native_quil using only basic_compile; the qc
    object's compiler is only used to translate native quil to an executable.

    A symmetrization_method can be specified which will be used to generate the necessary
    symmetrization results. This method should match the api of exhaustive_symmetrization; there,
    a list of symmetrized programs, the qubits to be measured for each program, the qubits that
    were flipped for each program, and the original pre-symmetrized program index are returned
    so that the bitarray results of the symmetrized programs and be processed via
    consolidate_symmetrization_outputs which returns 'symmetrized' results for the original
    pre-symmetrized programs.

    :param qc: a quantum computer object on which to run the programs necessary to estimate each
        observable of obs_expt.
    :param obs_expt: a single ObservablesExperiment with settings pre-grouped as desired.
    :param num_shots: the number of shots to run each program or each symmetrized program.
    :param symmetrization_method: if not None, this can be used to symmetrize each program to
        remove unwanted classical correlations in the qubit measurement readout.
    :param active_reset: whether or not to begin the program by actively resetting. If true,
        execution of each of the returned programs in a loop on the QPU will generally be faster.
    :return: all of the ExperimentResults which hold an estimate of each observable of obs_expt
    """
    programs, meas_qubits = generate_experiment_programs(obs_expt, active_reset)

    if symmetrization_method is not None:
        programs, symm_qubits, flip_arrays, prog_groups = symmetrization_method(programs,
                                                                                meas_qubits)
        symm_outputs = _measure_bitstrings(qc, programs, symm_qubits, num_shots)
        results = consolidate_symmetrization_outputs(symm_outputs, flip_arrays, prog_groups)
    else:
        results = _measure_bitstrings(qc, programs, meas_qubits, num_shots)

    assert len(results) == len(meas_qubits) == len(obs_expt)
    for bitarray, meas_qs, settings in zip(results, meas_qubits, obs_expt):

        for setting in settings:
            observable = setting.observable

            # Obtain statistics from result of experiment
            obs_mean, obs_var = shots_to_obs_moments(bitarray, meas_qs, observable)

            yield ExperimentResult(
                setting=setting,
                expectation=obs_mean,
                std_err=np.sqrt(obs_var),
                total_counts=len(bitarray),
            )


def get_calibration_program(observable: PauliTerm, noisy_program: Program = None) -> Program:
    """
    Program required for calibrating the given observable.

    :param observable: observable to calibrate
    :param noisy_program: a program with readout and gate noise defined; only useful for QVM
    :return: Program performing the calibration
    """
    calibr_prog = Program()

    # Inherit any noisy attributes from noisy_program, including gate definitions
    # and applications which can be handy in simulating noisy channels
    if noisy_program is not None:
        # Inherit readout error instructions from main Program
        readout_povm_instruction = [i for i in noisy_program.out().split('\n')
                                    if 'PRAGMA READOUT-POVM' in i]
        calibr_prog += readout_povm_instruction
        # Inherit any definitions of noisy gates from main Program
        kraus_instructions = [i for i in noisy_program.out().split('\n') if 'PRAGMA ADD-KRAUS' in i]
        calibr_prog += kraus_instructions

    # Prepare the +1 eigenstate for the out operator
    for q, op in observable.operations_as_set():
        calibr_prog += _one_q_pauli_prep(label=op, index=0, qubit=q)
    # Measure the out operator in this state
    for q, op in observable.operations_as_set():
        calibr_prog += _local_pauli_eig_meas(op, q)

    return calibr_prog


def calibrate_observable_estimates(qc: QuantumComputer, expt_results: List[ExperimentResult],
                                   symmetrization_method: Callable = exhaustive_symmetrization,
                                   num_shots: int = 500, noisy_program: Program = None) \
        -> Iterable[ExperimentResult]:
    """
    Calibrates the expectation and std_err of the input expt_results and updates those estimates.

    The old estimates are moved to raw_* and the calibration results themselves for the given
    observable are recorded as calibration_*

    Calibration is done by measuring expectation values of eigenstates of the obervable which
    ideally should yield either +/- 1 but in practice will have magnitude less than 1.

    :param qc: a quantum computer object on which to run the programs necessary to calibrate each
        result.
    :param expt_results: a list of results, each of which will be separately calibrated.
    :param symmetrization_method: by default every eigenvector of each observable is measured.
    :param num_shots: the number of shots to run for each eigenvector
    :param noisy_program: an optional program from which to inherit a noise model; only relevant
        for running on a QVM
    :return: a copy of the input results with updated estimates and calibration results.
    """
    observables = [copy(res.setting.observable) for res in expt_results]
    for obs in observables:
        obs.coefficient = 1.
    observables = list(set(observables))  # get unique observables that will need to be calibrated

    programs = [get_calibration_program(obs, noisy_program) for obs in observables]
    meas_qubits = [obs.get_qubits() for obs in observables]

    symm_progs, symm_qubits, flip_array, prog_groups = symmetrization_method(programs, meas_qubits)
    symm_outputs = _measure_bitstrings(qc, symm_progs, symm_qubits, num_shots)
    results = consolidate_symmetrization_outputs(symm_outputs, flip_array, prog_groups)

    assert len(results) == len(meas_qubits) == len(observables)
    calibrations = {}
    for bitarray, meas_qs, obs in zip(results, meas_qubits, observables):
        # Obtain statistics from result of experiment
        obs_mean, obs_var = shots_to_obs_moments(bitarray, meas_qs, obs)
        calibrations[obs.operations_as_set()] = (obs_mean, obs_var, len(bitarray))

    for expt_result in expt_results:
        # get the calibration data for this observable
        cal_data = calibrations[expt_result.setting.observable.operations_as_set()]
        obs_mean, obs_var, counts = cal_data

        # Use the calibration to correct the mean and var
        result_mean = expt_result.expectation
        result_var = expt_result.std_err**2
        corrected_mean = result_mean / obs_mean
        corrected_var = ratio_variance(result_mean, result_var, obs_mean, obs_var)

        yield ExperimentResult(
            setting=expt_result.setting,
            expectation=corrected_mean,
            std_err=np.sqrt(corrected_var),
            total_counts=expt_result.total_counts,
            raw_expectation=result_mean,
            raw_std_err=expt_result.std_err,
            calibration_expectation=obs_mean,
            calibration_std_err=np.sqrt(obs_var),
            calibration_counts=counts
        )


def ratio_variance(a: Union[float, np.ndarray],
                   var_a: Union[float, np.ndarray],
                   b: Union[float, np.ndarray],
                   var_b: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    r"""
    Given random variables 'A' and 'B', compute the variance on the ratio Y = A/B. Denote the
    mean of the random variables as a = E[A] and b = E[B] while the variances are var_a = Var[A]
    and var_b = Var[B] and the covariance as Cov[A,B]. The following expression approximates the
    variance of Y
    Var[Y] \approx (a/b) ^2 * ( var_a /a^2 + var_b / b^2 - 2 * Cov[A,B]/(a*b) )
    We assume the covariance of A and B is negligible, resting on the assumption that A and B
    are independently measured. The expression above rests on the assumption that B is non-zero,
    an assumption which we expect to hold true in most cases, but makes no such assumptions
    about A. If we allow E[A] = 0, then calculating the expression above via numpy would complain
    about dividing by zero. Instead, we can re-write the above expression as
    Var[Y] \approx var_a /b^2 + (a^2 * var_b) / b^4
    where we have dropped the covariance term as noted above.
    See the following for more details:
      - https://doi.org/10.1002/(SICI)1097-0320(20000401)39:4<300::AID-CYTO8>3.0.CO;2-O
      - http://www.stat.cmu.edu/~hseltman/files/ratio.pdf
      - https://en.wikipedia.org/wiki/Taylor_expansions_for_the_moments_of_functions_of_random_variables
    :param a: Mean of 'A', to be used as the numerator in a ratio.
    :param var_a: Variance in 'A'
    :param b: Mean of 'B', to be used as the numerator in a ratio.
    :param var_b: Variance in 'B'
    """
    return var_a / b**2 + (a**2 * var_b) / b**4
