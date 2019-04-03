from typing import Tuple, Sequence, List, Set, Dict, Any
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
    settings: Tuple[ExperimentSetting]
    qubits: Tuple[int]
    label: str = None
    num_shots: int = None
    results: Tuple[ExperimentResult] = None
    estimates: Dict[str, Tuple[float, float]] = None

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
    expt_type: str
    estimates: Dict[str, Tuple[float, float]] = None
    meta_data: Dict[str, Any] = None

    def __str__(self):
        return '\n'.join([str(lyr) for lyr in self.layers]) + '\n'


def _group_allowed_types(expts, allowed_parallel_types = None):
    parallelizable_groups = []
    expt_type_group_idx = {}
    for expt in expts:
        if expt.expt_type not in expt_type_group_idx.keys():
            group_idx = len(parallelizable_groups)
            parallelizable_groups.append([expt])
            expt_type_group_idx[expt.expt_type] = group_idx

            if allowed_parallel_types is not None:
                # form allowed groups of experiments based on experiment type
                for allowed_groups in allowed_parallel_types:  # search for group
                    if expt.expt_type in allowed_groups:
                        for other_type in allowed_groups:  # set group index for all members
                            expt_type_group_idx[other_type] = group_idx
                    break  # found group, so stop looking
        else:
            parallelizable_groups[expt_type_group_idx[expt.expt_type]].append(expt)

    return parallelizable_groups


def _group_by_depth(experiment_groups: Sequence[Sequence[StratifiedExperiment]]):

    depth_groups = {}
    for expt_group in experiment_groups:
        # first go through all of the experiments in allowed type groups and make subgroups by depth
        type_groups = {}
        for expt in expt_group:
            for layer in expt.layers:
                if layer.depth in type_groups.keys():
                    type_groups[layer.depth].append(layer)
                else:
                    type_groups[layer.depth] = [layer]

        # now aggregate the subgroups by depth
        for depth, type_group in type_groups.items():
            if depth in depth_groups.keys():
                depth_groups[depth].append(type_group)
            else:
                depth_groups[depth] = [type_group]

    return [group for group in depth_groups.values()]


def _get_simultaneous_components(layers: Sequence[Layer]):
    #TODO: consider e.g. running two short components serially in parallel with one longer component
    #TODO: any reason this should or shouldn't be randomized?
    components = permutation([component for layer in layers for component in layer.components])

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
    settings_pool = [permutation(component.settings) for component in components]
    max_num_settings = max([len(component.settings) for component in components])
    groups = []
    idx_maps = []
    for idx in range(max_num_settings):
        group = [settings[idx] for settings in settings_pool if len(settings) > idx ]
        groups.append(group)
        idx_map = [j for j, settings in enumerate(settings_pool) if len(settings) > idx]
        idx_maps.append(idx_map)

    return groups, idx_maps


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


def _run_component_group(qc, group, num_shots):
    sequence = merge_sequences([component.sequence for component in group])
    qubits = [q for component in group for q in component.qubits]

    # basic compile probably not necessary here, but in general may be for other modules.
    program = merge_programs(sequence)

    all_results = [[] for _ in group]
    for settings, idx_map  in zip(*_partition_settings(group)):
        single_expt = PyQuilTomographyExperiment([settings], program, qubits)
        group_results = list(measure_observables(qc, single_expt, num_shots))

        for idx, result in enumerate(group_results):
            all_results[idx_map[idx]].append(result)

    for component, results in zip(group, all_results):
        component.results = results
        component.num_shots = num_shots

        # defer to component.estimates for now
        # component.mean = np.mean([result.expectation for result in results])
        # # TODO: check this... stderr? covariance for IX, XI, XX?
        # var = np.sum([result.stddev**2 for result in results]) / len(results)**2
        # component.stderr = np.sqrt(var)


def acquire_stratified_data(qc, experiments: Sequence[StratifiedExperiment], num_shots: int = 500,
                            parallelize_components: bool = True,
                            allowed_parallel_types: Sequence[Sequence[str]] = None,
                            randomize_components_within_depth = False,
                            randomize_layers_within_depth = False,
                            randomize_all_components = True):
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
    :param parallelize_components: if true, components of equal depth will be run in parallel
        wherever possible. Components may be run in parallel if they act on disjoint sets of
        qubits, and if they do not rely on a DELAY pragma. You may require that grouping of
        experiments only occur for certain types; see param allowed_parallel_types and method
        _group_by_depth
    :param allowed_parallel_types: by default, when parallelize_components is set to true only
        components in experiments of the same type will ever possibly be run in parallel. If
        allowed_parallel_types is specified then components whose experiment types share an inner
        list may be run in parallel. Note that running a T1/T2 experiment in parallel with any other
        type may affect results since measurement of all qubits occurs at the end of each program;
        if the non T1/T2 experiment sequences take more time than the DELAY time specified,
        the T1/T2 results will not match the intended delay time.
    :param randomize_components_within_depth:
    :param randomize_layers_within_depth:
    :param randomize_all_components:
    :return: Currently mutates the input StratifiedExperiments.
    """
    # make a copy of each experiment, or return a separate results dataclass?
    # copies = [copy.deepcopy(expt) for expt in experiments]

    if not isinstance(experiments, Sequence):
        experiments = [experiments]

    if parallelize_components:
        expts_by_type_group = _group_allowed_types(experiments, allowed_parallel_types)
    else:
        expts_by_type_group = [[expt] for expt in experiments]

    # TODO: any reason to suppply option for not randomizing order of depths?
    # for each depth get a list of groups of layers from experiment types that can be parallelized
    parallelizable_groups_by_depth = permutation(_group_by_depth(expts_by_type_group))

    all_simultaneous_component_groups = []
    for parallelizable_groups in parallelizable_groups_by_depth:  # iterate over each depth

        if randomize_layers_within_depth:  # randomize (groups) of layers within this depth
            parallelizable_groups = permutation(parallelizable_groups)

        component_groups_by_depth = []
        # iterate over each parallelizable group of layers
        for layers in parallelizable_groups:

            if parallelize_components:
                # form groups of components acting on separate qubits that can be run in parallel
                component_groups = _get_simultaneous_components(layers)
            else:
                # put each component in its own component group. All components are run serially.
                component_groups = [[component] for layer in layers
                                    for component in layer.components]
            # form a flat list of all component groups at this depth
            component_groups_by_depth += component_groups

        if randomize_components_within_depth:
            # randomize components within this depth; this overrides randomize_layers_within_depth
            component_groups_by_depth = permutation(component_groups_by_depth)
        # form a flat list of each group of simultaneous components that will be run.
        all_simultaneous_component_groups += component_groups_by_depth

    if randomize_all_components:
        # shuffle all of the components; any randomization before this is moot since components
        # will be run in random order irrespective of depth or layer.
        all_simultaneous_component_groups = permutation(all_simultaneous_component_groups)

    for cgroup in all_simultaneous_component_groups:
        _run_component_group(qc, cgroup, num_shots)
