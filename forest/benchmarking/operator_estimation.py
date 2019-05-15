import functools
from functools import partial
import itertools
import json
import logging
import re
import sys
import warnings
from json import JSONEncoder
from operator import mul
from typing import List, Union, Iterable, Tuple, Dict, Callable

import numpy as np
from math import pi
import networkx as nx
from networkx.algorithms.approximation.clique import clique_removal
from pyquil import Program
from pyquil.api import QuantumComputer
from pyquil.gates import *
from pyquil.paulis import PauliTerm, sI, is_identity

from forest.benchmarking.compilation import basic_compile

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
    Input and output settings for a tomography-like experiment.
    Many near-term quantum algorithms take the following form:
     - Start in a pauli state
     - Prepare some ansatz
     - Measure it w.r.t. pauli operators
    Where we typically use a large number of (start, measure) pairs but keep the ansatz preparation
    program consistent. This class represents the (start, measure) pairs. Typically a large
    number of these :py:class:`ExperimentSetting` objects will be created and grouped into
    a :py:class:`ObservablesExperiment`.
    """
    in_state: TensorProductState
    observable: PauliTerm

    def __init__(self, in_state: TensorProductState, observable: PauliTerm):
        # For backwards compatibility, handle in_state specified by PauliTerm.
        if isinstance(in_state, PauliTerm):
            warnings.warn("Please specify in_state as a TensorProductState",
                          DeprecationWarning, stacklevel=2)

            if is_identity(in_state):
                in_state = TensorProductState()
            else:
                in_state = TensorProductState([
                    _OneQState(label=pauli_label, index=0, qubit=qubit)
                    for qubit, pauli_label in in_state._ops.items()
                ])

        object.__setattr__(self, 'in_state', in_state)
        object.__setattr__(self, 'observable', observable)

    @property
    def in_operator(self):
        warnings.warn("ExperimentSetting.in_operator is deprecated in favor of in_state",
                      stacklevel=2)

        # Backwards compat
        pt = sI()
        for oneq_state in self.in_state.states:
            if oneq_state.label not in ['X', 'Y', 'Z']:
                raise ValueError(f"Can't shim {oneq_state.label} into a pauli term. Use in_state.")
            if oneq_state.index != 0:
                raise ValueError(f"Can't shim {oneq_state} into a pauli term. Use in_state.")

            pt *= PauliTerm(op=oneq_state.label, index=oneq_state.qubit)

        return pt

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
    A tomography-like experiment.
    Many near-term quantum algorithms involve:
     - some limited state preparation
     - enacting a quantum process (like in tomography) or preparing a variational ansatz state
       (like in VQE)
     - measuring observables of the state.
    Where we typically use a large number of (state_prep, measure) pairs but keep the ansatz
    program consistent. This class stores the ansatz program as a :py:class:`~pyquil.Program`
    and maintains a list of :py:class:`ExperimentSetting` objects which each represent a
    (state_prep, measure) pair.
    Settings diagonalized by a shared tensor product basis (TPB) can (optionally) be estimated
    simultaneously. Therefore, this class is backed by a list of list of ExperimentSettings.
    Settings sharing an inner list will be estimated simultaneously. If you don't want this,
    provide a list of length-1-lists. As a convenience, if you pass a 1D list to the constructor
    will expand it to a list of length-1-lists.
    This class will not group settings for you. Please see :py:func:`group_experiments` for
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
    """Convenience method to save pyquil.operator_estimation objects as a JSON file.
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
    """Convenience method to read pyquil.operator_estimation objects from a JSON file.
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
            return Program(RY(pi / 2, qubit))
        else:
            return Program(RY(-pi / 2, qubit))

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
        return Program(RY(-pi / 2, idx))
    elif op == 'Y':
        return Program(RX(pi / 2, idx))
    elif op == 'Z':
        return Program()
    raise ValueError(f'Unknown operation {op}')


def construct_tpb_graph(experiments: ObservablesExperiment):
    """
    Construct a graph where an edge signifies two experiments are diagonal in a TPB.
    """
    g = nx.Graph()
    for expt in experiments:
        assert len(expt) == 1, 'already grouped?'
        expt = expt[0]

        if expt not in g:
            g.add_node(expt, count=1)
        else:
            g.nodes[expt]['count'] += 1

    for expt1, expt2 in itertools.combinations(experiments, r=2):
        expt1 = expt1[0]
        expt2 = expt2[0]

        if expt1 == expt2:
            continue

        max_weight_in = _max_weight_state([expt1.in_state, expt2.in_state])
        max_weight_out = _max_weight_operator([expt1.observable, expt2.observable])
        if max_weight_in is not None and max_weight_out is not None:
            g.add_edge(expt1, expt2)

    return g


def group_experiments_clique_removal(experiments: ObservablesExperiment) -> ObservablesExperiment:
    """
    Group experiments that are diagonal in a shared tensor product basis (TPB) to minimize number
    of QPU runs, using a graph clique removal algorithm.
    :param experiments: a tomography experiment
    :return: a tomography experiment with all the same settings, just grouped according to shared
        TPBs.
    """
    g = construct_tpb_graph(experiments)
    _, cliqs = clique_removal(g)
    new_cliqs = []
    for cliq in cliqs:
        new_cliq = []
        for expt in cliq:
            # duplicate `count` times
            new_cliq += [expt] * g.nodes[expt]['count']

        new_cliqs += [new_cliq]

    return ObservablesExperiment(new_cliqs, program=experiments.program)


def _max_weight_operator(ops: Iterable[PauliTerm]) -> Union[None, PauliTerm]:
    """Construct a PauliTerm operator by taking the non-identity single-qubit operator at each
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
    """Construct a TensorProductState by taking the single-qubit state at each
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


def _max_tpb_overlap(tomo_expt: ObservablesExperiment):
    """
    Given an input ObservablesExperiment, provide a dictionary indicating which ExperimentSettings
    share a tensor product basis
    :param tomo_expt: ObservablesExperiment, from which to group ExperimentSettings that share a tpb
        and can be run together
    :return: dictionary keyed with ExperimentSetting (specifying a tpb), and with each value being a
            list of ExperimentSettings (diagonal in that tpb)
    """
    # initialize empty dictionary
    diagonal_sets = {}
    # loop through ExperimentSettings of the ObservablesExperiment
    for expt_setting in tomo_expt:
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


def group_experiments_greedy(tomo_expt: ObservablesExperiment):
    """
    Greedy method to group ExperimentSettings in a given ObservablesExperiment
    :param tomo_expt: ObservablesExperiment to group ExperimentSettings within
    :return: ObservablesExperiment, with grouped ExperimentSettings according to whether
        it consists of PauliTerms diagonal in the same tensor product basis
    """
    diag_sets = _max_tpb_overlap(tomo_expt)
    grouped_expt_settings_list = list(diag_sets.values())
    grouped_tomo_expt = ObservablesExperiment(grouped_expt_settings_list, program=tomo_expt.program)
    return grouped_tomo_expt


def group_experiments(experiments: ObservablesExperiment,
                      method: str = 'greedy') -> ObservablesExperiment:
    """
    Group experiments that are diagonal in a shared tensor product basis (TPB) to minimize number
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
    :param experiments: a tomography experiment
    :param method: method used for grouping; the allowed methods are one of
        ['greedy', 'clique-removal']
    :return: a tomography experiment with all the same settings, just grouped according to shared
        TPBs.
    """
    allowed_methods = ['greedy', 'clique-removal']
    assert method in allowed_methods, f"'method' should be one of {allowed_methods}."
    if method == 'greedy':
        return group_experiments_greedy(experiments)
    elif method == 'clique-removal':
        return group_experiments_clique_removal(experiments)


@dataclass(frozen=True)
class ExperimentResult:
    """An expectation and standard deviation for the measurement of one experiment setting
    in a tomographic experiment.
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


def generate_experiment_programs(obs_expt: ObservablesExperiment, active_reset: bool) \
        -> Tuple[List[Program], List[List[int]]]:
    """
    Generate the programs necessary to estimate the observables in an ObservablesExperiment.

    :param obs_expt:
    :param active_reset:
    :return:
    """
    # Outer loop over a collection of grouped settings for which we can simultaneously estimate.
    programs = []
    meas_qubits = []
    for i, settings in enumerate(obs_expt):

        # Prepare a state according to the amalgam of all setting.in_state
        total_prog = Program()
        if active_reset:
            total_prog += RESET()
        max_weight_in_state = _max_weight_state(setting.in_state for setting in settings)
        for oneq_state in max_weight_in_state.states:
            total_prog += _one_q_state_prep(oneq_state)

        # Add in the program
        total_prog += obs_expt.program

        # Prepare for measurement state according to setting.observable
        max_weight_out_op = _max_weight_operator(setting.observable for setting in settings)
        for qubit, op_str in max_weight_out_op:
            total_prog += _local_pauli_eig_meas(op_str, qubit)

        programs.append(total_prog)
        meas_qubits.append(max_weight_out_op.get_qubits())
    return programs, meas_qubits


def consolidate_exhaustive_outputs(outputs: List[np.ndarray], groups: List[int],
                                   flip_arrays: List[Tuple[bool]]) -> List[np.ndarray]:
    """
    Given outputs from exhaustive_symmetrization programs, appropriately flip output bits and
    consolidate results into groups determined by the programs passed into
    exhaustive_symmetrization.

    :param outputs:
    :param groups:
    :param flip_arrays:
    :return:
    """
    output = []
    for bitarray, group, flip_array in zip(outputs, groups, flip_arrays):
        if len(output) <= group:
            output[group] = bitarray ^ flip_array
            continue
        output[group] = np.vstack(output[group], bitarray ^ flip_array)
    return output


def _flip_array_to_prog(flip_array: Tuple[bool], qubits: List[int]) -> Program:
    """
    Generate a pre-measurement program that flips the qubit state according to the flip_array of
    bools.

    :param ops_bool: tuple of booleans specifying the operation to be carried out on `qubits`
    :param qubits: list specifying the qubits to be carried operations on
    :return: Program with the operations specified in `ops_bool` on the qubits specified in
        `qubits`
    """
    assert len(flip_array) == len(qubits), "Mismatch of qubits and operations"
    prog = Program()
    for i, flip_output in enumerate(flip_array):
        if flip_output == 0:
            continue
        elif flip_output == 1:
            prog += Program(X(qubits[i]))
        else:
            raise ValueError("flip_bools should only consist of 0s and/or 1s")
    return prog


def exhaustive_symmetrization(programs: List[Program], meas_qubits: List[List[int]]) \
        -> Tuple[List[Program], Callable]:
    """
    For each program in the input programs generate new programs which flip the measured qubits
    in every possible combination in order to symmetrize readout.

    :param programs:
    :param meas_qubits:
    :return:
    """
    symm_programs = []
    prog_groups = []
    flip_arrays = []

    for idx, (prog, meas_qs) in enumerate(zip(programs, meas_qubits)):
        for flip_array in itertools.product([0, 1], repeat=len(meas_qs)):
            total_prog_symm = prog.copy()
            prog_symm = _flip_array_to_prog(flip_array, meas_qs)
            total_prog_symm += prog_symm
            symm_programs.append(total_prog_symm)
            prog_groups.append(idx)
            flip_arrays.append(flip_array)

    return symm_programs, partial(consolidate_exhaustive_outputs, groups=prog_groups,
                                  flip_arrays=flip_arrays)


def _measure_bitstrings(qc: QuantumComputer, programs: List[Program], prog_qubits: List[List[int]],
                        num_shots = 500) -> List[np.ndarray]:
    """
    Wrapper for appending measure instructions onto each program, running the program,
    and accumulating the resulting bitarrays.

    Each program is assumed to be composed of gates that can be compiled into native quil by
    basic_compile.

    :param qc:
    :param programs:
    :param prog_qubits:
    :param num_shots:
    :return:
    """
    results = []
    for program, qubits in zip(programs, prog_qubits):

        ro = program.declare('ro', 'BIT', len(qubits))
        for idx, q in enumerate(qubits):
            program += MEASURE(q, ro[idx])

        program.wrap_in_numshots_loop(num_shots)
        native_quil = basic_compile(program)
        exe = qc.compiler.native_quil_to_executable(native_quil)
        shots = qc.run(exe)
        results.append(shots)
    return results


def shots_to_obs_moments(bitarray: np.ndarray, qubits: List[int], observable: PauliTerm) \
        -> Tuple[float, float]:
    """
    Calculate the mean and variance of the given observable based on the bitarray of results.

    :param bitarray: results from running `qc.run`
    :param qubits: list of qubits in order corresponding to the bitarray results.
    :param observable: the observable whose moments are calculated from the shot data
    :return: tuple specifying (mean, variance)
    """
    coeff = complex(observable.coefficient)
    if not np.isclose(coeff.imag, 0):
        raise ValueError(f"The coefficient of an observable should not be complex.")
    coeff = coeff.real

    obs_qubits = [q for q, _ in observable]
    # Identify classical register indices to select
    idxs = [idx for idx, q in enumerate(qubits) if q in obs_qubits]
    # Pick columns corresponding to qubits with a non-identity out_operation
    obs_strings = bitarray[:, idxs]
    # Transform bits to eigenvalues; ie (+1, -1)
    my_obs_strings = 1 - 2 * obs_strings
    # Multiply row-wise to get operator values. Do statistics. Return result.
    obs_vals = coeff * np.prod(my_obs_strings, axis=1)
    obs_mean = np.mean(obs_vals).item()
    obs_var = np.var(obs_vals).item() / len(bitarray)

    return obs_mean, obs_var


def estimate_observables(qc: QuantumComputer, obs_expt: ObservablesExperiment,
                         num_shots: int = 500, symmetrization_method: Callable = None,
                         active_reset: bool = False) -> Iterable[ExperimentResult]:
    """
    Standard wrapper for estimating the observables in an ObservableExperiment.

    Because of the use of standard _measure_bitstrings, this method assumes the program in
    obs_expt can be compiled to native_quil using only basic_compile.

    A symmetrization_method can be specified which will be used to generate the necessary
    symmetrization results. This method should return a callable 'output_transform' which takes
    in the full list of bitarrays from the symmetrization programs and consolidates the outputs
    into the appropriate results for the initial un-symmetrized programs. For example,
    in exhaustive_symmetrization the results of the symmetrization programs have appropriate bits
    flipped and are re-grouped into bitarrays so that the final list of bitarrays corresponds to
    the initial programs output by generate_experiment_programs.

    :param qc:
    :param obs_expt:
    :param num_shots:
    :param symmetrization_method:
    :param active_reset:
    :return:
    """
    programs, meas_qubits = generate_experiment_programs(obs_expt, active_reset)

    if symmetrization_method is not None:
        programs, output_transform = symmetrization_method(programs, meas_qubits)

    results = _measure_bitstrings(qc, programs, meas_qubits, num_shots)

    if symmetrization_method is not None and output_transform is not None:
        results = output_transform(results)

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
    for q, op in observable.observable.operations_as_set():
        calibr_prog += _local_pauli_eig_meas(op, q)

    return calibr_prog


def calibrate_observable_estimates(qc: QuantumComputer, expt_results: List[ExperimentResult],
                                   symmetrization_method: Callable = exhaustive_symmetrization,
                                   num_shots: int = 500, noisy_program: Program = None) \
        -> Iterable[ExperimentResult]:
    """
    Calibrates the expectation and std_err of the input expt_results.

    :param qc:
    :param expt_results:
    :param symmetrization_method:
    :param num_shots:
    :param noisy_program:
    :return:
    """
    programs = [get_calibration_program(expt_result.setting.observable, noisy_program) for
                    expt_result in expt_results]

    meas_qubits = [expt_result.setting.observable.get_qubits() for expt_result in expt_results]

    programs, output_transform = symmetrization_method(programs, meas_qubits)

    results = _measure_bitstrings(qc, programs, meas_qubits, num_shots)
    results = output_transform(results)

    for bitarray, meas_qs, expt_result in zip(results, meas_qubits, expt_results):

        observable = expt_result.setting.observable

        # Obtain statistics from result of experiment
        obs_mean, obs_var = shots_to_obs_moments(bitarray, meas_qs, observable)

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
            calibration_counts=len(bitarray)
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
