from typing import Tuple, Sequence, List, Set, Dict
import numpy as np
from numpy.random import permutation
import networkx as nx
from networkx.algorithms.approximation.clique import clique_removal
from dataclasses import dataclass

from pyquil.operator_estimation import ExperimentSetting, ExperimentResult, measure_observables, \
    TomographyExperiment as PyQuilTomographyExperiment
from pyquil import Program
from pyquil.quil import merge_programs


@dataclass()
class Component:
    """
    A component is the low-level structure of a StratifiedExperiment that stores a sequence of
    gates that will be run on a qc.

    In the case of standard rb, a component is simply the rb Clifford sequence whose end
    result will be measured in the computational basis. For unitarity, a component sequence will
    be run with 4^n different measurements.

    Note that in standard rb the result of interest for a given sequence is the "survival
    probability" which is essentially just the shifted z expectation of the sequence end-state.
    In unitarity the result of interest for a sequence is a purity--this is calculated from the
    expectations of the 4^n expectations that are stored in the 4^n ExperimentResults individual
    expectations.
    """
    sequence: Tuple[Program]
    experiments: Tuple[ExperimentSetting]
    qubits: Tuple[int]
    label: str = None
    num_shots: int = None
    results: Tuple[ExperimentResult] = None
    estimates: Dict[str, Tuple[float, float]] = None

    # these probably aren't very useful to keep track of
    # mean: float = None
    # stderr: float = None


    def __str__(self):
        output = self.label + '['

        abbreviate_sequence = len([inst for gate in self.sequence for inst in gate]) > 5

        if not abbreviate_sequence:
            output += '[' + '], ['.join([', '.join([str(instr) for instr in gate]) for gate in
                                   self.sequence]) + ']'
        else:
            first_gate = self.sequence[0]
            abbreviate_first_gate = len(first_gate) >= 3

            first_gate_output = '['
            if abbreviate_first_gate:
                first_gate_output += str(first_gate[0]) + ',...,  ' + str(first_gate[-1])
            else:
                first_gate_output += ', '.join([str(instr) for instr in first_gate])
            first_gate_output += ']'

            output += first_gate_output

            if len(self.sequence) >=3:
                output += ',...'

            if len(self.sequence) >= 2:
                last_gate = self.sequence[-1]
                abbreviate_last_gate = len(last_gate) >= 3

                last_gate_output = ', ['
                if abbreviate_last_gate:
                    last_gate_output += str(last_gate[0]) + ',...,  ' + str(last_gate[-1])
                else:
                    last_gate_output += ', '.join([str(instr) for instr in last_gate])
                last_gate_output += ']'

                output += last_gate_output

        return output + ']'


@dataclass(order=True)
class Layer:
    """
    A Layer is the mid-level structure of a StratifiedExperiment that collects all of the
    individual qc-runnable components of a particular depth.

    Each component may operate on the same qubits, and so in general may not be parallelizable.
    For both rb and unitarity a particular layer will contain num_sequences_per_depth many
    components, a component just acting as a container for each sequence; in the case of
    unitarity a component consists of 4^n different programs that will need to be run,
    one for each pauli operator.

    """
    depth: int
    components: Tuple[Component]
    estimates: Dict[str, Tuple[float, float]] = None

    def __str__(self):
        return f'Depth {self.depth}:\n' + '\n'.join([str(comp) for comp in self.components]) + '\n'


@dataclass
class StratifiedExperiment:
    """
    This is the high-level structure that captures everything about an experiment on a particular
    qubit or pair of qubits.

    A StratifiedExperiment is composed of several layers of increasing depth. Simultaneous RB would
    involve making a StratifiedExperiment for each of the qubits you want to characterize and
    passing all of these experiments to the acquire_data method which sorts out which can be run
    together.
    """
    layers: Tuple[Layer]
    qubits: Tuple[int]
    exp_type: str

    def __str__(self):
        return '\n'.join([str(lyr) for lyr in self.layers]) + '\n'


def determine_simultaneous_expt_grouping(experiments: Sequence[StratifiedExperiment]) \
        -> List[Set[int]]:
    """
    Determines a grouping of experiments acting on disjoint sets of qubits that can be run
    simultaneously.

    :param experiments:
    :return: a list of the simultaneous groups, each specified by a set of indices of each grouped
        experiment in experiments
    """
    g = nx.Graph()
    nodes = np.arange(len(experiments))
    g.add_nodes_from(nodes)
    qubits = [set(expt.qubits) for expt in experiments]

    for node1 in nodes:
        qbs1 = qubits[node1]
        for node2 in nodes[node1+1:]:
            if len(qbs1.intersection(qubits[node2])) == 0:
                # no shared qubits so add edge
                g.add_edge(node1, node2)

    # get the largest groups of nodes with shared edges, as each can be run simultaneously
    _, cliqs = clique_removal(g)

    return cliqs


def merge_sequences(sequences: list) -> list:
    """
    Takes a list of equal-length "sequences" (lists of Programs) and merges them element-wise,
    returning the merged outcome.

    :param sequences: List of equal-length Lists of Programs
    :return: A single List of Programs
    """
    depth = len(sequences[0])
    assert all([len(s) == depth for s in sequences])
    return [merge_programs([seq[idx] for seq in sequences]) for idx in range(depth)]


def _group_by_depth(experiments: Sequence[StratifiedExperiment]):
    groups = {}
    for expt in experiments:
        for layer in expt.layers:
            if layer.depth in groups.keys():
                groups[layer.depth].append(layer)
            else:
                groups[layer.depth] = [layer]
    return groups


def _get_simultaneous_components(layers: Sequence[Layer]):
    #TODO: consider e.g. running two short components serially in parallel with one longer component
    components = [component for layer in layers for component in layer.components]

    g = nx.Graph()
    nodes = np.arange(len(components))
    g.add_nodes_from(nodes)
    qubits = [set(component.qubits) for component in components]

    for node1 in nodes:
        qbs1 = qubits[node1]
        for node2 in nodes[node1+1:]:
            if len(qbs1.intersection(qubits[node2])) == 0:
                # no shared qubits so add edge
                g.add_edge(node1, node2)

    # get the largest groups of nodes with shared edges, as each can be run simultaneously
    _, cliques = clique_removal(g)

    return [[components[idx] for idx in clique] for clique in cliques]


def _partition_settings(components: Sequence[Component]):
    # assume components act on separate qubits
    settings_pool = [permutation(component.experiments) for component in components]
    max_num_settings = max([len(component.experiments) for component in components])
    groups = []
    idx_maps = []
    for idx in range(max_num_settings):
        group = [settings[idx] for settings in settings_pool if len(settings) > idx ]
        groups.append(group)
        idx_map = [j for j, settings in enumerate(settings_pool) if len(settings) > idx]
        idx_maps.append(idx_map)

    return groups, idx_maps


def acquire_stratified_data(qc, experiments: Sequence[StratifiedExperiment], num_shots: int = 500):
    """
    Takes in StratifiedExperiments and simultaneously runs individual Components of separate
    experiments that are in Layers of equal depth.

    First, we group all the Layers that share a depth. Within one such group, we then group all
    of the Components that act on disjoint sets of qubits and so can be run simultaneously; this
    is done irrespective of experiment type
    TODO: add support for requiring same experiment type or even same component label.
    For each of these groups we form a single simultaneous program that is run on the qc
    and randomly partition measurement settings to be measured simultaneously on this program
    until all of the required measurements for each component are taken (note this might mean
    that the sequences of components with fewer measurements are simply run without being
    measured for some runs).

    :param qc:
    :param experiments:
    :param num_shots:
    :return: Currently mutates the input StratifiedExperiments.
    """
    # make a copy of each experiment, or return a separate results dataclass?
    # copies = [copy.deepcopy(expt) for expt in experiments]

    if not isinstance(experiments, Sequence):
        experiments = [experiments]

    depth_groups = _group_by_depth(experiments)

    # could shuffle depths too, permutation(depth_groups.items())
    for depth, dgroup in depth_groups.items():
        # currently, get back groups of components in random order
        # e.g. if we are passed unitarity, irb, and t1 experiments acting on the *same* qubit then
        # the sequences for each will be comingled in random order and run serially. If multiple
        # experiments of the same type act on different qubits, then there is currently no
        # guarantee that experiments of the same type will be run simultaneously or that other
        # experiments will be uniformly randomly comingled into simultaneous groups...
        component_groups = permutation(_get_simultaneous_components(dgroup))
        for cgroup in component_groups:

            sequence = merge_sequences([component.sequence for component in cgroup])
            qubits = [q for component in cgroup for q in component.qubits]

            # basic compile probably not necessary here, but in general may be for other modules.
            program = merge_programs(sequence)

            all_results = [[] for _ in cgroup]
            for settings, idx_map  in zip(*_partition_settings(cgroup)):
                single_expt = PyQuilTomographyExperiment([settings], program, qubits)
                group_results = list(measure_observables(qc, single_expt, num_shots))

                for idx, result in enumerate(group_results):
                    all_results[idx_map[idx]].append(result)

            for component, results in zip(cgroup, all_results):
                component.results = results
                component.num_shots = num_shots

                # defer to component.estimates for now
                # component.mean = np.mean([result.expectation for result in results])
                # # TODO: check this... stderr? covariance for IX, XI, XX?
                # var = np.sum([result.stddev**2 for result in results]) / len(results)**2
                # component.stderr = np.sqrt(var)
