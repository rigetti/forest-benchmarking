from math import pi
from typing import Iterable, List, Sequence, Tuple, Dict

import numpy as np
from lmfit.model import ModelResult
from numpy import pi
from tqdm import tqdm

from pyquil.api import BenchmarkConnection, QuantumComputer
from pyquil.gates import CZ, RX, RZ
from pyquil.quilbase import Gate
from pyquil.quil import merge_programs
from pyquil import Program

from forest.benchmarking.tomography import _state_tomo_settings
from forest.benchmarking.utils import all_traceless_pauli_z_terms, is_pos_pow_two
from forest.benchmarking.analysis.fitting import fit_base_param_decay
from forest.benchmarking.observable_estimation import ExperimentSetting, ExperimentResult, \
    zeros_state, estimate_observables, ObservablesExperiment, group_settings, \
    get_results_by_qubit_groups


def get_stats_by_qubit_group(qubit_groups: Sequence[Sequence[int]],
                               expt_results: Iterable[Iterable[ExperimentResult]]) \
        -> Dict[Tuple[int, ...], Dict[str, List[List[float]]]]:
    """
    Organize the results of a simultaneous RB experiment into lists of expectations and std_errs
    for each sequence; these lists are stored in a dict for each qubit group.

    :param qubit_groups: disjoint groups of qubits for which we have simultaneous RB results.
    :param expt_results: ExperimentResults for each ObservablesExperiment run as part of a
        RB experiment
    :return: a dict whose keys are qubit groups (as tuples) with corresponding value which is an
        inner dict with 'expectation' and 'std_err' estimates for each group of observables
        measured on each sequence.
    """
    qubits = [tuple(group) for group in qubit_groups]
    stats_by_qubit_group = {group: {'expectation': [], 'std_err': []} for group in qubit_groups}
    for results in expt_results:
        res_by_qubit_group = get_results_by_qubit_groups(results, qubits)

        for group in qubits:
            # pull out the expectation and std_err from each result
            stats_by_qubit_group[group]['expectation'].append([res.expectation for res in
                                                               res_by_qubit_group[group]])
            stats_by_qubit_group[group]['std_err'].append([res.std_err for res in
                                                           res_by_qubit_group[group]])

    return stats_by_qubit_group


def oneq_rb_gateset(qubit: int) -> Gate:
    """
    Yield the gateset for 1-qubit randomized benchmarking.

    :param qubit: The qubit to effect the gates on.
    """
    for angle in [-pi, -pi / 2, pi / 2, pi]:
        for gate in [RX, RZ]:
            yield gate(angle, qubit)


def twoq_rb_gateset(q1: int, q2: int) -> Iterable[Gate]:
    """
    Yield the gateset for 2-qubit randomized benchmarking.

    This is two 1-q gatesets and ``CZ``.

    :param q1: The first qubit.
    :param q2: The second qubit.
    """
    yield from oneq_rb_gateset(q1)
    yield from oneq_rb_gateset(q2)
    yield CZ(q1, q2)


def get_rb_gateset(qubits: Sequence[int]) -> List[Gate]:
    """
    A wrapper around the gateset generation functions.

    :param qubits: the qubits on which to the gates should act
    :returns: list of gates, tuple of qubits
    """
    if len(qubits) == 1:
        return list(oneq_rb_gateset(qubits[0]))

    if len(qubits) == 2:
        return list(twoq_rb_gateset(*qubits))

    raise ValueError(f"No RB gateset for more than two qubits.")


def merge_sequences(sequences: List[List[Program]]) -> List[Program]:
    """
    Takes a list of equal-length "sequences" (lists of Programs) and merges them element-wise,
    returning the merged outcome.
    :param sequences: List of equal-length Lists of Programs
    :return: A single List of Programs
    """
    depth = len(sequences[0])
    assert all([len(s) == depth for s in sequences])
    return [merge_programs([seq[idx] for seq in sequences]) for idx in range(depth)]


def generate_rb_sequence(benchmarker: BenchmarkConnection, qubits: Sequence[int], depth: int,
                         interleaved_gate: Program = None, random_seed: int = None) \
        -> List[Program]:
    """
    Generate a complete randomized benchmarking sequence.

    :param benchmarker: object returned from get_benchmarker() used to generate clifford sequences
    :param qubits: qubits on which the sequence will act
    :param depth: The total number of Cliffords in the sequence (including inverse)
    :param random_seed: Random seed passed to the benchmarker to seed sequence generation.
    :param interleaved_gate: See [IRB]_; this gate will be interleaved into the sequence
    :return: A list of programs constituting Clifford gates in a self-inverting sequence.
    """
    if depth < 2:
        raise ValueError("Sequence depth must be at least 2 for rb sequences, or at least 1 for "
                         "unitarity sequences.")
    gateset = get_rb_gateset(qubits)
    programs = benchmarker.generate_rb_sequence(depth=depth, gateset=gateset,
                                                interleaver=interleaved_gate, seed=random_seed)
    # return a sequence composed of depth-many Cliffords.
    return programs


def generate_rb_experiment_sequences(benchmarker: BenchmarkConnection, qubits: Sequence[int],
                                     depths: Sequence[int], interleaved_gate: Program = None,
                                     random_seed: int = None, use_self_inv_seqs=True) \
        -> List[List[Program]]:
    """
    Generate the sequences of Clifford gates necessary to run a randomized benchmarking
    experiment for a single (group of) qubit(s).

    A Clifford is given as a compiled sequence of native gates in a Program. The compilation is
    done by the BenchmarkConnection object. A sequence at a depth d is thus provided as a list of
    d programs each representing a random Clifford.

    Calling this method separately on several different groups of qubits with the same depths
    will yield separate lists of sequences which can be passed to
    group_sequences_into_parallel_experiments in order to generate a 'simultaneous' RB experiment.

    :param benchmarker: object returned from get_benchmarker() used to generate clifford sequences
    :param qubits: the qubits for a single isolated rb experiment
    :param depths: the depth of each sequence in the experiment.
    :param interleaved_gate: optional gate to interleave throughout the sequence, see [IRB]_
    :param random_seed: Random seed passed to benchmarker to seed sequence generation.
    :param use_self_inv_seqs: by default True, the last Clifford of the sequence will be the
        inverse of the composition of the previous Cliffords in the sequence; the entire sequence
        is thus the identity operation in the ideal case. If set to False then this last gate is
        simply omitted but the total number of gates is preserved.
    :return: a list of all of the len(depths) many different Clifford sequences.
    """
    sequences = []  # we will have len(depths) many sequences
    for depth in depths:
        if random_seed is not None:  # need to change the base seed for each sequence generated
            random_seed += 1

        if use_self_inv_seqs:
            # a sequence is a list of Cliffords, with last Clifford inverting the sequence
            sequence = generate_rb_sequence(benchmarker, qubits, depth, interleaved_gate,
                                            random_seed)
        else: # this might be desired for unitarity experiments
            # First we provide larger depth, then strip inverse from end of the sequence
            sequence = generate_rb_sequence(benchmarker, qubits, depth + 1,
                                            random_seed=random_seed)[:-1]

        sequences.append(sequence)

    return sequences


def group_sequences_into_parallel_experiments(parallel_expts_seqs: Sequence[List[List[Program]]],
                                              qubit_groups: Sequence[Sequence[int]],
                                              is_unitarity_expt: bool = False) \
        -> List[ObservablesExperiment]:
    """
    Consolidates randomized benchmarking sequences on separate groups of qubits into a flat list
    of ObservablesExperiments which merge parallel sets of distinct sequences.

    Each returned ObservablesExperiment constitutes a single 'parallel RB sequence' where all of
    the qubits are acted upon and measured. Running all of these ObservablesExperiments in series
    constitutes a 'parallel RB' experiment from which you can determine a decay constant for each
    group of qubits. Note that there is an important physical distinction (e.g. due to
    cross-talk) between running separate RB experiments on different groups of qubits and running
    a 'parallel RB' experiment on the collection of those groups. For this reason one should not
    expect in general that the rb decay for a particular group of qubits is comparable between
    the individual and parallel modes of rb experiment.

    :param parallel_expts_seqs: the outer Sequence is indexed by disjoint groups of qubits;
        Clifford sequences from each of these different groups (which should be of the same depth
        across qubit groups) will be merged together into a single program. The intended use-case
        is that each List[List[program]] of sequences of Cliffords is an output of
        generate_rb_experiment_sequences for disjoint groups of qubits but with identical
        depths input (see generate_rb_experiments for example). If sequences of different depth are
        merged into a Program then some qubits may be sitting idle while the sequences of greater
        depth continue running. Measurement occurs only when all sequences have terminated.
    :param qubit_groups: The partition of the qubits into groups for each of which you would like to
        estimate an rb decay. Typically this grouping of qubits should match the qubits that are
        acted on by each sequence in the corresponding List[List[Program]] of the input
        parallel_expts_seqs.
    :param is_unitarity_expt: True if the desired experiment is a unitarity experiment, in which
        case additional settings are required to estimate the purity of the sequence output.
    :return: a list of ObservablesExperiments constituting a parallel rb experiment.
    """
    expts = []
    for parallel_sequence_group in zip(*parallel_expts_seqs):
        program = merge_programs(merge_sequences(parallel_sequence_group))

        if is_unitarity_expt:
            settings = [sett for group in qubit_groups for sett in _state_tomo_settings(group)]
            expt = group_settings(ObservablesExperiment(settings, program))
        else:
            # measure observables of products of I and Z on qubits in the group, excluding all I
            settings = [ExperimentSetting(zeros_state(group), op)
                    for group in qubit_groups for op in all_traceless_pauli_z_terms(group)]
            expt = ObservablesExperiment([settings], program)
        expts.append(expt)
    return expts


def generate_rb_experiments(benchmarker: BenchmarkConnection, qubit_groups: Sequence[Sequence[int]],
                            depths: Sequence[int], interleaved_gate: Program = None,
                            random_seed: int = None) -> List[ObservablesExperiment]:
    """
    Creates list of ObservablesExperiments which, when run in series, constitute a
    simultaneous randomized benchmarking experiment on the disjoint qubit_groups.

    The number of ObservablesExperiments returned is equal to len(depths). A particular
    ObservablesExperiment consists of

    - a program, which is a random sequence of Clifford gates compiled down to native gates.
        If len(qubit_groups) > 1 then the program is actually the sum of len(qubit_groups) many
        separate random sequences; each sequence acts only on the group of qubits in a
        particular element of the input 'qubit_groups' list.

    - settings; for each group within qubit_groups there will be settings which dictate that
        each qubit in that group is initialized to the `|0>` state and that some observable
        which is a tensor product of Z and I factors is measured for that group. All of these
        settings are initialized within the ObservablesExperiment to be run in parallel.

    Specifying a interleaved_gate will generate a Clifford sequence which alternates depth many
    times between a random Clifford and the specified gate. The gate itself should be a Program
    written as a sequence of native gates implementing a Clifford element. The sequence will
    still contain depth many random Cliffords (excluding the interleaved gate) including the
    final inverting Clifford.

    For standard RB see [RB]_. For interleaved RB see [IRB]_.

    .. [RB] Scalable and Robust Randomized Benchmarking of Quantum Processes.
         Magesan et al.
         Phys. Rev. Lett. 106, 180504 (2011).
         https://dx.doi.org/10.1103/PhysRevLett.106.180504
         https://arxiv.org/abs/1009.3639

    .. [IRB] Efficient measurement of quantum gate error by interleaved randomized benchmarking.
        Magesan et al.
        Phys. Rev. Lett. 109, 080505 (2012).
        https://dx.doi.org/10.1103/PhysRevLett.109.080505
        https://arxiv.org/abs/1203.4550

    :param benchmarker: object returned from get_benchmarker() used to generate clifford sequences
    :param qubit_groups: the disjoint groups qubits for which random sequences will be
        generated and merged into a series of programs each of which runs groups of disjoint
        sequences 'simultaneously'.
    :param depths: the depth of each sequence in the experiment
    :param interleaved_gate: optional gate to interleave throughout the sequence, see [IRB]
    :param random_seed: Random seed passed to benchmarker to seed sequence generation.
    :return: a list of ObservablesExperiments which constitute a simultaneous RB or IRB experiment
    """
    parallel_sequences = []
    for group in qubit_groups:
        if random_seed is not None:
            # need to change the base seed for each set of qubits
            random_seed += len(depths)

        parallel_sequences.append(generate_rb_experiment_sequences(benchmarker, group, depths,
                                                                    interleaved_gate, random_seed))

    return group_sequences_into_parallel_experiments(parallel_sequences, qubit_groups)


def acquire_rb_data(qc: QuantumComputer, experiments: Iterable[ObservablesExperiment],
                    num_shots: int = 500, active_reset: bool = False,
                    show_progress_bar: bool =  False) \
        -> List[List[ExperimentResult]]:
    """
    Runs each ObservablesExperiment and returns each group of resulting ExperimentResults

    :param qc: a quantum computer, e.g. QVM or QPU, that runs the experiments
    :param experiments: a list of Observables experiments
    :param num_shots: the number of shots to run each group of simultaneous ExperimentSettings
    :param active_reset: Boolean flag indicating whether experiments should begin with an
        active reset instruction (this can make the collection of experiments run a lot faster).
    :param show_progress_bar: displays a progress bar via tqdm if true.
    :return: a list of ExperimentResults for each ObservablesExperiment
    """
    results = []
    for expt in tqdm(experiments, disable=not show_progress_bar):
        results.append(list(estimate_observables(qc, expt, num_shots, active_reset=active_reset)))
    return results


def covariances_of_all_iz_obs(expectations: Sequence[float], num_shots: int):
    """
    Calculate the sum of the pairwise covariance of every distinct pair of observables whose
    expectations are given.

    It is assumed that the list of expectations corresponds to 2**num_qubits - 1 = dim - 1 many
    observables, where the observables all consist of every combination of I and Z acting on
    different qubits. Calculating this covariance is necessary if

        1) all observables were calculated from the same set of shot data
        2) you are calculating the variance for a sum of the expectations.

    We calculate the covariance using

    .. math::

        COVAR(O_i, O_j) = E[O_i O_j] - E[O_i] E[O_j]

    and noticing that the product of two distinct observables :math:`O_i O_j` from our list is
    simply a second observable :math:`1O_k` in the list. Furthermore, taking every possible pairwise
    product simply results in two copies of our original list. Hence if we take the sum over all
    distinct pairs we can calculate the covariance purely as a function of observable expectations.

    :param expectations:
    :param num_shots: the number of shots used to estimate each expectation
    :return:
    """
    assert is_pos_pow_two(len(expectations) + 1)

    # first, we have the contribution of all expectations of products
    # E[O_i O_j] = E[O_j O_i] = E[O_k]
    covariance = 2 * sum(expectations)

    # now we subtract the pairwise products of expectations E[O_i]E[O_j]
    covariance -= sum([exp1 * exp2 for i, exp1 in enumerate(expectations)
                       for j, exp2 in enumerate(expectations) if i != j])
    # return the sample covariance
    return covariance / num_shots


def z_obs_stats_to_survival_statistics(expectations: Sequence[float], std_errs: Sequence[float],
                                       num_shots = None, obs_are_independent = False):
    """
    Convert expectations of the dim - 1 observables which are the nontrivial combinations of tensor
    products of I and Z into survival mean and variance, where survival is the all zeros outcome.

    If dim > 2, i.e. there are more than 2 qubits, and the observable expectations were collected
    simultaneously on the same set of shot data then there will be covariance between the
    different observables; thus to calculate the survival variance we must include the
    contribution of the covariance, which requires knowledge of the number of shots.

    :param expectations:
    :param std_errs:
    :param num_shots: the number of shots used to estimate each expectation
    :param obs_are_independent:
    :return:
    """
    # This assumes inclusion of all terms with at least one Z to make dim-1 many total terms
    dim = len(expectations) + 1  # = 2**num_qubits
    assert is_pos_pow_two(dim)

    survival_probability = (sum(expectations) + 1) / dim
    survival_var = sum(np.asarray(std_errs) ** 2) / dim**2

    if dim > 2 and not obs_are_independent:
        if num_shots is None:
            raise ValueError("The number of shots is necessary information for computing the "
                             "sample covariance.")

        # since the observables are not independent, e.g. they were calculated using the same set
        # of shot data, we need to calculate the sum of the covariance of each distinct pair of
        # observables and add this to our variance, appropriately scaled
        survival_var += covariances_of_all_iz_obs(expectations, num_shots) / dim**2

    return survival_probability, survival_var


def fit_rb_results(depths: Sequence[int], z_expectations: Sequence[Sequence[float]],
                   z_std_errs: Sequence[Sequence[float]], num_shots: int = None,
                   param_guesses: tuple = None) -> ModelResult:
    """
    Fits the results of a standard RB or IRB experiment by converting expectations into survival
    probabilities (probability of measuring zero) and passing these on to the standard fit.

    The estimate for the rb decay can be found in the returned fit.params['decay']

    First for each sequence we calculate the mean and variance of the estimated probability of the
    zeros state given the expectation of all operators with Z terms. We note that the sum
    of all Z/I operators divided by the dimension is the projector onto the all zeros state,
    so the sum of all corresponding expectations (including one for all I operator) divided by
    the dimension is the probability of the all zeros state.

    :param depths: the depth of each sequence over which a decay will be fitted
    :param z_expectations: the groups of 2**(num_qubits) - 1 expectations estimated for each
        sequence, where each group of observables has all traceless tensor products of I and Z.
    :param z_std_errs: the groups of std_errs for each expectation estimate
    :param num_shots:
    :param param_guesses: guesses for the (amplitude, decay, baseline) parameters
    :return: a ModelResult fit with estimates of the Model parameters, including the rb 'decay'
    """
    survivals = []
    variances = []

    assert len(depths) == len(z_expectations), 'There should be one expectation per sequence and ' \
                                               'depths should give the depth of each sequence.'

    for depth, expectations, std_errs in zip(depths, z_expectations, z_std_errs):
        # get the fraction of all zero outcomes 00...00
        survival_prob, survival_var = z_obs_stats_to_survival_statistics(expectations, std_errs,
                                                                         num_shots)

        survivals.append(survival_prob)
        variances.append(survival_var)

    if param_guesses is None:  # make some standard reasonable guess (amplitude, decay, baseline)
        param_guesses = (survivals[0] - survivals[-1], 0.95, survivals[-1])

    err = np.sqrt(variances)
    non_zero = [v for v in err if v > 0]
    if len(non_zero) == 0:
        weights = None
    else:
        # TODO: does this handle 0 var appropriately?
        # Other possibility is to use unbiased prior in std_err estimate.
        min_non_zero = min(non_zero)
        non_zero_err = np.asarray([v if v > 0 else min_non_zero for v in err])

        weights = 1 / non_zero_err

    return fit_base_param_decay(np.asarray(depths), np.asarray(survivals), weights, param_guesses)


def generate_unitarity_experiments(benchmarker: BenchmarkConnection,
                                   qubit_groups: Sequence[Sequence[int]], depths: Sequence[int],
                                   random_seed: int = None, use_self_inv_seqs=False) \
        -> List[ObservablesExperiment]:
    """
    Creates list of ObservablesExperiments which, when run in series, constitute a
    simultaneous unitarity experiment on the disjoint qubit_groups.

    Similar to a standard RB experiment, save for two changes:
        1) the sequence of Cliffords need not be self-inverting
        2) currently the purity of the output state is estimated by measuring each of the
            observables in the Pauli basis on the given qubits. As such not all Observables can
            be estimate simultaneously and we use the simultaneous grouping offered by
            operator_estimation

    Unitarity algorithm is due to [ECN]_.

    .. [ECN]  Estimating the Coherence of Noise.
           Wallman et al.
           New Journal of Physics 17, 113020 (2015).
           https://dx.doi.org/10.1088/1367-2630/17/11/113020
           https://arxiv.org/abs/1503.07865

    :param benchmarker: object returned from get_benchmarker() used to generate clifford sequences
    :param qubit_groups: the disjoint groups qubits for which random sequences will be
        generated and merged into a series of programs each of which runs groups of disjoint
        sequences 'simultaneously'.
    :param depths: the depth of each sequences in the experiment.
    :param random_seed: Random seed passed to benchmarker to seed sequence generation.
    :param use_self_inv_seqs: by default False, unlike with a typical RB sequence the last
        Clifford does not invert the sequence. If True, the subset of Z*I observable experiment
        results can equally well be analyzed as a unitarity or RB experiment. This argument does
        not affect the total number of Cliffords in the sequence.
    :return: a list of ObservablesExperiments which constitute a simultaneous unitarity experiment
    """
    parallel_sequences = []
    for group in qubit_groups:
        if random_seed is not None:
            # need to change the base seed for each set of qubits
            random_seed += len(depths)

        parallel_sequences.append(
            generate_rb_experiment_sequences(benchmarker, group, depths, random_seed=random_seed,
                                             use_self_inv_seqs=use_self_inv_seqs))

    return group_sequences_into_parallel_experiments(parallel_sequences, qubit_groups,
                                                     is_unitarity_expt = True)


def estimate_purity(dim: int, op_expect: np.ndarray, renorm: bool=True):
    """
    The renormalized, or 'shifted', purity is given in equation (10) of [ECN]_
    where d is the dimension of the Hilbert space, 2**num_qubits

    :param dim: dimension of the hilbert space
    :param op_expect: array of estimated expectations of each operator being measured
    :param renorm: flag that renormalizes result to be between 0 and 1
    :return: purity given the operator expectations
    """
    # assumes op_expect includes expectation of I with value 1.
    purity = (1 / dim) * sum(op_expect**2)
    if renorm:
        purity = (dim / (dim - 1.0)) * (purity - 1.0 / dim)
    return purity


def estimate_purity_err(dim: int, op_expect: np.ndarray, op_expect_var: np.ndarray, renorm=True):
    """
    Propagate the observed variance in operator expectation to an error estimate on the purity.
    This assumes that each operator expectation is independent.

    :param dim: dimension of the Hilbert space
    :param op_expect: array of estimated expectations of each operator being measured
    :param op_expect_var: array of estimated variance for each operator expectation
    :param renorm: flag that provides error for the renormalized purity
    :return: purity given the operator expectations
    """
    # TODO: incorporate covariance of observables estimated simultaneously; see covariances_of_all_iz_obs

    #TODO: check validitiy of approximation |op_expect| >> 0, and functional form below (squared?)
    var_of_square_op_expect = (2 * np.abs(op_expect)) ** 2 * op_expect_var
    #TODO: check if this adequately handles |op_expect| >\> 0
    need_second_order = np.isclose([0.]*len(var_of_square_op_expect), var_of_square_op_expect, atol=1e-6)
    var_of_square_op_expect[need_second_order] = op_expect_var[need_second_order]**2

    purity_var = (1 / dim) ** 2 * (np.sum(var_of_square_op_expect))

    if renorm:
        purity_var = (dim / (dim - 1.0)) ** 2 * purity_var

    return np.sqrt(purity_var)


def fit_unitarity_results(depths: Sequence[int], expectations: Sequence[Sequence[float]],
                          std_errs: Sequence[Sequence[float]], param_guesses: tuple = None) \
        -> ModelResult:
    """
    Fits the results of a unitarity experiment by first calculating shifted purities and
    subsequently passing these on to the standard decay fit.

    The estimate for the unitarity (the decay) can be found in the returned fit.params['decay']

    :param depths: the depth of each sequence over which a decay will be fitted
    :param expectations: the groups of 4**(num_qubits) - 1 expectations estimated for each sequence
    :param std_errs: the groups of std_errs for each expectation estimate
    :param param_guesses: guesses for the (amplitude, decay, baseline) parameters
    :return: a ModelResult fit with estimates of the Model parameters, including the 'decay',
        which is the unitarity parameter. Note that [ECN]_ parameterizes the decay differently;
        effectively, the 'amplitude' reported here absorbs a factor 1/unitarity.
        Comparing to 'B' in equation 8), fit.params['amplitude'] = B / fit.params['decay']
    """
    shifted_purities = []
    shifted_purity_errs = []

    assert len(depths) == len(expectations), 'There should be one group of 4**(num_qubits) - 1 ' \
                                             'expectations per sequence and depths should give ' \
                                             'the depth of each sequence.'

    for depth, exps, errs in zip(depths, expectations, std_errs):
        # This assumes inclusion of all terms with at least one observable to make dim**2-1 many
        # total terms
        dim = int(np.sqrt(len(exps) + 1))  # = 2**num_qubits

        # the estimate_purity methods assume inclusion of all Id term with expectation 1
        exps = np.asarray(list(exps) + [1.])
        op_vars = np.asarray(list(errs) + [0.])**2

        # shifted_purity is the estimated purity re-scaled to lay between 0 and 1 (see [ECN] eq. 10)
        shifted_purity = estimate_purity(dim, exps)
        shifted_purity_error = estimate_purity_err(dim, exps, op_vars)

        shifted_purities.append(shifted_purity)
        shifted_purity_errs.append(shifted_purity_error)

    if param_guesses is None:  # make some standard reasonable guess (amplitude, decay, baseline)
        param_guesses = (shifted_purities[0], 0.95, 0)

    non_zero = [v for v in shifted_purity_errs if v > 0]
    if len(non_zero) == 0:
        weights = None
    else:
        # TODO: does this handle 0 var appropriately?
        # Other possibility is to use unbiased prior in std_err estimate.
        min_non_zero = min(non_zero)
        non_zero_err = np.asarray([v if v > 0 else min_non_zero for v in shifted_purity_errs])

        weights = 1 / non_zero_err

    return fit_base_param_decay(np.asarray(depths), np.asarray(shifted_purities), weights,
                                param_guesses)


def unitarity_to_rb_decay(unitarity, dimension) -> float:
    """
    Converts a unitarity decay to a standard RB decay under the assumption that no unitary errors
    present.

    Proposition 8. in [ECN]_ gives the unitarity u as an upper bound of a function of the average
    gate error r and the dimension d:

    .. math::

        u >= (1 - d r/(d-1))^2

    This upper bound is saturated when no unitary errors are present, i.e. the noise is purely
    stochastic, like in the case of depolarizing noise. We can use this saturated equality to
    upper bound the standard RB decay. This allows comparison of measured unitarity and RB
    decays. We might hope that the gates are well calibrated and finely controlled so that the RB
    decay is very close to the upper bound given by the unitarity passed into this function.

    :param unitarity: The measured decay parameter in a unitarity measurement
    :param dimension: The dimension of the Hilbert space, 2**num_qubits
    :return: The upper bound on RB decay, saturated if no unitary errors are present, Proposition
        8 of [ECN]_
    """
    r = (np.sqrt(unitarity) - 1)*(1-dimension)/dimension
    return average_gate_error_to_rb_decay(r, dimension)


def do_rb(qc: QuantumComputer, benchmarker: BenchmarkConnection,
          qubit_groups: Sequence[Sequence[int]], depths: Sequence[int],
          interleaved_gate: Program = None, is_unitarity_expt: bool = False,
          num_shots: int = 1_000, active_reset: bool = False, show_progress_bar: bool = False) \
        -> Tuple[Dict[Tuple[int, ...], float],
                 List[ObservablesExperiment],
                 List[List[ExperimentResult]]]:
    """
    A wrapper around experiment generation, data acquisition, and estimation that runs a RB
    experiment on the qubit_groups and returns the rb_decay along with the experiments and results.

    :param qc: A quantum computer object on which the experiment will run.
    :param benchmarker: object returned from pyquil.api.get_benchmarker() used to generate
        sequences of Clifford elements decomposed into native gates.
    :param qubit_groups: The partition of qubits into groups. For each group we will estimate an
        rb decay. Each decay should be interpreted as a 'simultaneous rb decay' as the sequences
        on each group of qubits will be run concurrently.
    :param depths: the depth of each sequence in the experiment
    :param interleaved_gate: optional gate to interleave throughout the sequence, see [IRB]
    :param is_unitarity_expt: True if the desired experiment is a unitarity experiment, in which
        case additional settings are required to estimate the purity of the sequence output.
    :param num_shots: The number of shots collected for each experiment setting on each sequence.
    :param active_reset: Boolean flag indicating whether experiments should begin with an
        active reset instruction (this can make the collection of experiments run a lot faster).
    :param show_progress_bar: displays a progress bar via tqdm if true.
    :return: The estimated rb decays for each group of qubits, along with the experiment and
        corresponding results.
    """
    if is_unitarity_expt:
        expts = generate_unitarity_experiments(benchmarker, qubit_groups, depths)
    else:
        expts = generate_rb_experiments(benchmarker, qubit_groups, depths,
                                        interleaved_gate=interleaved_gate)

    results = list(acquire_rb_data(qc, expts, num_shots, active_reset=active_reset,
                                   show_progress_bar=show_progress_bar))

    stats_by_group = get_stats_by_qubit_group(qubit_groups, results)

    decays = {}
    for group, stats in stats_by_group.items():
        if is_unitarity_expt:
            fit = fit_unitarity_results(depths, stats['expectation'], stats['std_err'])
        else:
            fit = fit_rb_results(depths, stats['expectation'], stats['std_err'], num_shots)

        decays[group] =  fit.params['decay'].value

    return decays, expts, results


########
# Interleaved RB Analysis
########


def coherence_angle(rb_decay, unitarity):
    """
    Equation 29 of [U+IRB]_

    :param rb_decay: Observed decay parameter in standard rb experiment
    :param unitarity: Observed decay parameter in unitarity experiment
    :return: coherence angle
    """
    return np.arccos(rb_decay / np.sqrt(unitarity))


def gamma(irb_decay, unitarity):
    """
    Corollary 5 of [U+IRB]_, second line

    :param irb_decay: Observed decay parameter in irb experiment with desired gate interleaved between Cliffords
    :param unitarity: Observed decay parameter in unitarity experiment
    :return: gamma
    """
    return irb_decay/np.sqrt(unitarity)


def interleaved_gate_fidelity_bounds(irb_decay, rb_decay, dim, unitarity = None):
    """
    Use observed rb_decay to place a bound on fidelity of a particular gate with given interleaved rb decay.
    Optionally, use unitarity measurement result to provide improved bounds on the interleaved gate's fidelity.

    Bounds are due to [IRB]_. Improved bounds using unitarity are due to [U+IRB]_

    .. [U+IRB]  Efficiently characterizing the total error in quantum circuits.
             Dugas et al.
             arXiv:1610.05296 (2016).
             https://arxiv.org/abs/1610.05296

    :param irb_decay: Observed decay parameter in irb experiment with desired gate interleaved between Cliffords
    :param rb_decay: Observed decay parameter in standard rb experiment
    :param dim: Dimension of the Hilbert space, 2**num_qubits
    :param unitarity: Observed decay parameter in unitarity experiment; improves bounds if provided.
    :return: The pair of lower and upper bounds on the fidelity of the interleaved gate.
    """
    if unitarity is not None:
        # Corollary 5 of [U+IRB]. Here, the channel X corresponds to the interleaved gate
        # whereas Y corresponds to the averaged-Clifford channel of standard rb.

        pm = [-1, 1]
        theta = coherence_angle(rb_decay, unitarity)
        g =  gamma(irb_decay, unitarity)
        # calculate bounds on the equivalent gate-only decay parameter
        decay_bounds = [sign * (sign * g * np.cos(theta) + np.sin(theta) * np.sqrt(1-g**2) ) for sign in pm]
        # convert decay bounds to bounds on fidelity of the gate
        fidelity_bounds = [1 - rb_decay_to_gate_error(decay, dim) for decay in decay_bounds]

    else:
        # Equation 5 of [IRB]

        E1 = (abs(rb_decay - irb_decay/rb_decay) + (1-rb_decay)) * (dim-1)/dim
        E2 = 2*(dim**2 - 1)*(1-rb_decay)/(rb_decay*dim**2) + 4*np.sqrt(1-rb_decay)*np.sqrt(dim**2-1)/rb_decay

        E = min(E1,E2)
        error = irb_decay_to_gate_error(irb_decay, rb_decay, dim)

        fidelity_bounds = [1-error-E, 1-error+E]

    return fidelity_bounds


def gate_error_to_irb_decay(irb_error, rb_decay, dim):
    """
    For convenience, inversion of Eq. 4 of [IRB]_. See irb_decay_to_error

    :param irb_error: error of the interleaved gate.
    :param rb_decay: Observed decay parameter in standard rb experiment.
    :param dim: Dimension of the Hilbert space, 2**num_qubits
    :return: Decay parameter in irb experiment with relevant gate interleaved between Cliffords
    """
    return (1 - irb_error * (dim/(dim-1)) ) * rb_decay


def irb_decay_to_gate_error(irb_decay, rb_decay, dim):
    """
    Eq. 4 of [IRB]_, which provides an estimate of the error of the interleaved gate,
    given both the observed interleaved and standard decay parameters.

    :param irb_decay: Observed decay parameter in irb experiment with desired gate interleaved between Cliffords
    :param rb_decay: Observed decay parameter in standard rb experiment.
    :param dim: Dimension of the Hilbert space, 2**num_qubits
    :return: Estimated gate error of the interleaved gate.
    """
    return ((dim - 1) / dim) * (1 - irb_decay / rb_decay)


def average_gate_error_to_rb_decay(gate_error, dimension):
    """
    Inversion of eq. 5 of [RB]_ arxiv paper.

    :param gate_error: The average gate error.
    :param dimension: Dimension of the Hilbert space, 2^num_qubits
    :return: The RB decay corresponding to the gate_error
    """
    return (gate_error - 1 + 1/dimension)/(1/dimension -1)


def rb_decay_to_gate_error(rb_decay, dimension):
    """
    Eq. 5 of [RB]_ arxiv paper. Note that 'gate' here typically means an element of the Clifford
    group, which comprise standard rb sequences.

    :param rb_decay: Observed decay parameter in standard rb experiment.
    :param dimension: Dimension of the Hilbert space, 2**num_qubits
    :return: The gate error corresponding to the input decay.
    """
    return 1 - rb_decay - (1 - rb_decay)/dimension
