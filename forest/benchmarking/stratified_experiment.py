from typing import Tuple, Sequence, Dict, Any
import itertools
from copy import deepcopy
import numpy as np
from numpy.random import shuffle
import networkx as nx
from networkx.algorithms.approximation.clique import clique_removal
from dataclasses import dataclass

from pyquil.operator_estimation import ExperimentSetting, ExperimentResult, measure_observables, \
    TomographyExperiment as PyQuilTomographyExperiment, _max_weight_state, _max_weight_operator
from pyquil import Program
from pyquil.quil import merge_programs

from forest.benchmarking.compilation import basic_compile

@dataclass(order=True)
class Layer:
    """
    A Layer is the low-level structure of a StratifiedExperiment that stores a sequence of
    gates that will be run on a qc.
    """
    depth: int
    sequence: Tuple[Program]
    settings: Tuple[ExperimentSetting]
    qubits: Tuple[int]
    label: str = None
    num_shots: int = None
    results: Tuple[ExperimentResult] = None
    continuous_param: float = None
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
                first_gate_output += str(first_gate[0]) + ', ...,  ' + str(first_gate[-1])
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
                    last_gate_output += str(last_gate[0]) + ', ...,  ' + str(last_gate[-1])
                else:
                    last_gate_output += ', '.join([str(instr) for instr in last_gate])
                last_gate_output += ']'

                output += last_gate_output

        return output + ']'


@dataclass
class StratifiedExperiment:
    """
    This is the high-level structure intended to organize an isolated experiment on a particular
    group of qubits where there is a notion of iterated data collection for 'layers' of
    increasing depth.

    A StratifiedExperiment is composed of several layers of increasing depth. Simultaneous RB would
    involve making a StratifiedExperiment for each of the qubits you want to characterize and
    passing all of these experiments to the acquire_data method which sorts out which can be run
    together.
    """
    layers: Tuple[Layer]
    qubits: Tuple[int]
    expt_type: str
    estimates: Dict[str, Tuple[float, float]] = None
    metadata: Dict[str, Any] = None

    def __str__(self):
        return self.expt_type + '[' + '\n'.join([str(lyr) for lyr in self.layers]) + ']\n'


def _group_allowed_types(expts: Sequence[StratifiedExperiment], allowed_parallel_types = None):
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


def _get_simultaneous_groups(layers: Sequence[Layer]):
    #TODO: consider e.g. running two short sequences serially in parallel with one longer sequence
    #TODO: any reason this should or shouldn't be randomized?
    g = nx.Graph()
    nodes = np.arange(len(layers))
    g.add_nodes_from(nodes)

    for node1 in nodes:
        qbs1 = set(layers[node1].qubits)
        for node2 in nodes[node1+1:]:
            if len(qbs1.intersection( set(layers[node2].qubits))) == 0:
                # no shared qubits so add edge
                g.add_edge(node1, node2)
            elif layers[node1].sequence == layers[node2].sequence:
                # programs are the same, so experiment settings may be parallelizable
                # TODO: check behavior... multiple copies of the same experiment may be
                #  effectively erased, but without this one cannot run e.g. a cz ramsey or rpe
                #  experiment while measuring multiple qubits simultaneously.
                g.add_edge(node1, node2)

    # get the largest groups of nodes with shared edges, as each can be run simultaneously
    _, cliques = clique_removal(g)

    return [[layers[idx] for idx in clique] for clique in cliques]


def _partition_settings(layers: Sequence[Layer]):
    # assume layer sequences act on separate qubits or else are equivalent.
    layer_map = {}
    settings_pool = []
    g = nx.Graph()
    for layer in layers:
        for setting in layer.settings:
            if setting not in g:
                g.add_node(setting, count=1)
                layer_map[setting] = [layer]
                settings_pool.append(setting)
            else:
                g.nodes[setting]['count'] += 1
                layer_map[setting].append(layer)

    for setting1, setting2 in itertools.combinations(settings_pool, r=2):
        max_weight_in = _max_weight_state([setting1.in_state, setting2.in_state])
        max_weight_out = _max_weight_operator([setting1.out_operator, setting2.out_operator])
        if max_weight_in is not None and max_weight_out is not None:
            g.add_edge(setting1, setting2)

    _, cliqs = clique_removal(g)
    new_cliqs = []
    associated_layers = []
    for cliq in cliqs:
        cliq_layers = []
        new_cliq = []
        for setting in cliq:
            # duplicate `count` times
            new_cliq += [setting] * g.nodes[setting]['count']
            cliq_layers += layer_map[setting]

        new_cliqs += [new_cliq]
        associated_layers.append(cliq_layers)

    return new_cliqs, associated_layers


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


def acquire_stratified_data(qc, experiments: Sequence[StratifiedExperiment], num_shots: int = None,
                            parallelize_layers: bool = True,
                            allowed_parallel_types: Sequence[Sequence[str]] = None,
                            randomize_layers_within_depth: bool = True,
                            randomize_all_layers: bool = True,
                            randomize_all_settings: bool = True,
                            use_active_reset: bool = False):
    """
    Takes in StratifiedExperiments and simultaneously runs individual layers of equal depth
    across separate experiments with allowed_parallel_types.

    First, we group all the Layers that share a depth. Within one such group, we then group all
    of the Components that act on disjoint sets of qubits and so can be run simultaneously; this
    is done irrespective of experiment type

    For each of these groups we form a single simultaneous program that is run on the qc
    and randomly partition measurement settings to be measured simultaneously on this program
    until all of the required measurements for each layer are taken (note this might mean
    that the sequences of layers with fewer measurements are simply run without being
    measured for some runs).

    :param qc:
    :param experiments:
    :param num_shots:
    :param parallelize_layers: if true, layers of equal depth will be run in parallel
        wherever possible. Layers may be run in parallel if they act on disjoint sets of
        qubits. You may require that grouping of experiments only occur for certain types; see
        param allowed_parallel_types and method _group_by_depth
    :param allowed_parallel_types: by default, when parallelize_layers is set to true only
        layers in experiments of the same type will ever possibly be run in parallel. If
        allowed_parallel_types is specified then layers whose experiment types share an inner
        list may be run in parallel. Note that running a T1/T2 experiment in parallel with any other
        type may affect results since measurement of all qubits occurs at the end of each program;
        if the non T1/T2 experiment sequences take more time than the DELAY time specified in T1/T2,
        those results will not match the intended delay time.
    :param randomize_layers_within_depth: shuffle layers within each depth before data for that
        depth is collected
    :param randomize_all_layers: shuffle all of the layers before they are run, regardless of depth
    :return: Currently mutates the input StratifiedExperiments.
    """
    if isinstance(experiments, StratifiedExperiment):
        experiments = [experiments]

    # make a copy of each experiment
    experiments = [deepcopy(expt) for expt in experiments]

    if parallelize_layers:
        # by default only layers of the same expt type will possibly be parallelized.
        expts_by_type_group = _group_allowed_types(experiments, allowed_parallel_types)
    else:
        # each layer of each experiment will be run iteratively
        expts_by_type_group = [[expt] for expt in experiments]

    # for each depth get a list of groups of layers from experiment types that can be parallelized
    parallelizable_groups_by_depth = _group_by_depth(expts_by_type_group)
    # TODO: any reason to supply option for not randomizing order of depths?
    shuffle(parallelizable_groups_by_depth)

    all_simultaneous_groups = []
    for parallelizable_groups in parallelizable_groups_by_depth:  # iterate over each depth

        if randomize_layers_within_depth:  # randomize (parallel groups of) layers within this depth
            shuffle(parallelizable_groups)

        simultaneous_groups_at_depth = []
        if parallelize_layers:
            # iterate over each parallelizable group of layers
            for layers in parallelizable_groups:
                # form simultaneous groups of layers acting on separate qubits that can be run in
                # parallel.
                simult_groups = _get_simultaneous_groups(layers)
                simultaneous_groups_at_depth += simult_groups
        else:
            # layers are already partitioned into groups of one and each will be run series
            simultaneous_groups_at_depth = parallelizable_groups

        # form a flat list of each group of simultaneous layers that will be run.
        all_simultaneous_groups += simultaneous_groups_at_depth

    if randomize_all_layers:
        # shuffle all of the layers; any randomization before this is moot since each (group of)
        # layer(s) will be run in random order irrespective of depth or expt_type.
        shuffle(all_simultaneous_groups)

    simultaneous_experiment_settings = []
    for group in all_simultaneous_groups:

        sequence = merge_sequences([layer.sequence for layer in group])
        qubits = [q for layer in group for q in layer.qubits]

        program = basic_compile(merge_programs(sequence))
        for simultaneous_settings, associated_layers in zip(*_partition_settings(group)):
            single_expt = PyQuilTomographyExperiment([simultaneous_settings], program, qubits)
            simultaneous_experiment_settings.append((single_expt, associated_layers))

    if randomize_all_settings:
        shuffle(simultaneous_experiment_settings)

    for (single_expt, layers) in simultaneous_experiment_settings:

        if num_shots is None:
            max_num_shots = max([layer.num_shots for layer in layers])
        else:
            max_num_shots = num_shots
            for layer in layers:
                layer.num_shots = num_shots

        group_results = list(measure_observables(qc, single_expt, max_num_shots,
                                                 active_reset=use_active_reset))

        for layer, result in zip(layers, group_results):
            if layer.results is None:
                layer.results = [result]
            else:
                layer.results.append(result)

    return experiments