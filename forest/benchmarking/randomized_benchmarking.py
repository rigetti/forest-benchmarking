from math import pi
from typing import Iterable, List, Sequence, Tuple, Dict

import numpy as np
from lmfit.model import ModelResult
from numpy import pi

from pyquil.api import BenchmarkConnection, QuantumComputer
from pyquil.gates import CZ, RX, RZ
from pyquil.quilbase import Gate
from pyquil.quil import merge_programs
from pyquil import Program

from forest.benchmarking.tomography import _state_tomo_settings
from forest.benchmarking.utils import all_pauli_z_terms
from forest.benchmarking.analysis.fitting import fit_base_param_decay
from forest.benchmarking.operator_estimation import ExperimentSetting, ExperimentResult, \
    zeros_state, estimate_observables, ObservablesExperiment, group_experiment_settings


def get_results_by_qubit_group(qubits: Sequence[Sequence[int]],
                               parallel_results: List[ExperimentResult]) \
        -> Dict[Tuple[int], List[ExperimentResult]]:
    """

    :param qubits:
    :param parallel_results:
    :return:
    """
    qubits = [tuple(group) for group in qubits]
    results_by_qubit_group = {group: [] for group in qubits}
    for res in parallel_results:
        res_qs = res.setting.observable.get_qubits()

        for group in qubits:
            if set(res_qs).issubset(set(group)):
                results_by_qubit_group[group].append(res)

    return results_by_qubit_group


def get_stats_by_qubit_group(qubits: Sequence[Sequence[int]],
                               expt_results: List[List[ExperimentResult]]) \
        -> Dict[Tuple[int], Dict[str, List[float]]]:
    qubits = [tuple(group) for group in qubits]
    stats_by_qubit_group = {group: {'expectation': [], 'std_err': []} for group in qubits}
    for results in expt_results:
        res_by_qubit_group = get_results_by_qubit_group(qubits, results)

        for group in qubits:
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


def generate_rb_sequence(bm: BenchmarkConnection, qubits: Sequence[int], depth: int,
                         interleaved_gate: Program = None, random_seed: int = None) \
        -> List[Program]:
    """
    Generate a complete randomized benchmarking sequence.

    :param bm: object returned from get_benchmarker() used to generate clifford sequences
    :param qubits: qubits on which the sequence will act
    :param depth: The total number of Cliffords in the sequence (including inverse)
    :param random_seed: Random seed passed to bm to seed sequence generation.
    :param interleaved_gate: See [IRB]; this gate will be interleaved into the sequence
    :return: A list of programs constituting Clifford gates in a self-inverting sequence.
    """
    if depth < 2:
        raise ValueError("Sequence depth must be at least 2 for rb sequences, or at least 1 for "
                         "unitarity sequences.")
    gateset = get_rb_gateset(qubits)
    programs = bm.generate_rb_sequence(depth=depth, gateset=gateset, interleaver=interleaved_gate,
                                       seed=random_seed)
    return programs


def generate_rb_experiments(bm: BenchmarkConnection, qubits: Sequence[Sequence[int]],
                            depths: Sequence[int], num_sequences: int,
                            interleaved_gate: Program = None, random_seed: int = None) \
        -> Tuple[List[ObservablesExperiment], List[List[List[Program]]]]:
    """
    Creates all of the programs and organizes all information necessary for a RB or IRB
    experiment on the given qubits.

    The StratifiedExperiment returned can be passed to acquire_rb_data or acquire_stratified_data
    for data collection on a Qauntum Computer. To run simultaneous experiments, create multiple
    RB experiments and pass all of them to one of these methods with the appropriate flags set.

    For standard RB see
    [RB] Scalable and Robust Randomized Benchmarking of Quantum Processes
         Magesan et al.,
         Phys. Rev. Lett. 106, 180504 (2011)
         https://dx.doi.org/10.1103/PhysRevLett.106.180504
         https://arxiv.org/abs/1009.3639

    :param bm: object returned from get_benchmarker() used to generate clifford sequences
    :param qubits: the qubits for a single isolated rb experiment (for simultaneous rb,
        create multiple experiments and run them simultaneously)
    :param depths: the depths of the sequences in the experiment
    :param num_sequences: the number of sequences at each depth
    :param interleaved_gate: optional gate to interleave throughout the sequence, see [IRB]
    :param random_seed: Random seed passed to bm to seed sequence generation.
    :return: a StratifiedExperiment with all programs and information necessary for data
        collection in a RB or IRB experiment
    """
    expts = []  # we will have len(depths)*num_sequences many experiments
    sequences = []
    for depth in depths:
        for idx in range(num_sequences):
            parallel_sequences = []
            parallel_settings = []
            for group in qubits:
                if random_seed is not None:  # need to change the base seed for each sequence generated
                    random_seed += 1

                # a sequence is just a list of Cliffords, with last Clifford inverting the sequence
                sequence = generate_rb_sequence(bm, group, depth, interleaved_gate, random_seed)
                parallel_sequences.append(sequence)

                settings = [ExperimentSetting(zeros_state(group), op)
                            for op in all_pauli_z_terms(group)]
                parallel_settings += settings

            program = merge_programs(merge_sequences(parallel_sequences))
            expts.append(ObservablesExperiment([parallel_settings], program))
            sequences.append(parallel_sequences)

    return expts, sequences


def acquire_rb_data(qc: QuantumComputer, experiments: Sequence[ObservablesExperiment],
                    num_shots: int = 500) -> List[List[ExperimentResult]]:
    """

    :param qc: a quantum computer, e.g. QVM or QPU, that runs the experiments
    :param experiments: a list of Observables experiments
    :param num_shots: the number of shots to run each group of simultaneously ran ExperimentSettings
    :return:
    """
    results = []
    for expt in experiments:
        results.append(list(estimate_observables(qc, expt, num_shots)))
    return results


def fit_rb_results(depths: Sequence[int], z_expectations: Sequence[Sequence[float]],
                   z_std_errs: Sequence[Sequence[float]], param_guesses: tuple = None) \
        -> ModelResult:
    """
    Wrapper for fitting the results of RB or IRB; simply extracts key parameters
    and passes on to the standard fit.

    The estimate for the rb decay can be found in the returned fit.params['decay']

    First for each sequence we calculate the mean and variance of the estimated probability of the
    zeros state given the expectation of all operators with Z terms. We note that the sum
    of all Z/I operators divided by the dimension is the projector onto the all zeros state,
    so the sum of all corresponding expectations (including one for all I operator) divided by
    the dimension is the probability of the all zeros state.

    :param param_guesses: guesses for the (amplitude, decay, baseline) parameters
    :return: a ModelResult fit with estimates of the Model parameters, including the rb 'decay'
    """
    survivals = []
    variances = []

    assert len(depths) == len(z_expectations), 'There should be one expectation per sequence. ' \
                                               'The depths used in generate_experiment will need ' \
                                               'to be repeated for the appropriate number of ' \
                                               'sequences.'

    for depth, expectations, z_std_errs in zip(depths, z_expectations, z_std_errs):
        # This assumes inclusion of all terms with at least one Z to make dim-1 many total terms
        dim = len(expectations) + 1  # = 2**num_qubits
        # get the fraction of all zero outcomes 00...00
        survival_probability = (sum(expectations) + 1) / dim
        survival_prob_var = sum(np.asarray(z_std_errs)**2) / dim**2


        survivals.append(survival_probability)
        variances.append(survival_prob_var)

    if param_guesses is None:  # make some standard reasonable guess (amplitude, decay, baseline)
        param_guesses = (survivals[0] - survivals[-1], 0.95, survivals[-1])

    err = np.sqrt(variances)
    min_non_zero = min([v for v in err if v > 0])
    # TODO: does this handle 0 var appropriately? Incorporate unbiased prior into std_err estimate?
    non_zero_err = np.asarray([v if v > 0 else min_non_zero for v in err])

    weights = 1 / non_zero_err

    return fit_base_param_decay(np.asarray(depths), np.asarray(survivals), weights, param_guesses)


def generate_unitarity_experiments(bm: BenchmarkConnection, qubits: Sequence[Sequence[int]],
                                  depths: Sequence[int], num_sequences: int,
                                  use_self_inv_seqs = False, random_seed: int = None) \
        -> Tuple[List[ObservablesExperiment], List[List[List[Program]]]]:
    """
    Creates all of the programs and organizes all information necessary for a unitarity
    experiment on the given qubits.

    Similarly to generate_rb_experiment, the StratifiedExperiment returned can be passed to
    acquire_unitarity_data or acquire_stratified_data for data collection on a Qauntum Computer.

    Unitarity algorithm is due to
    [ECN]  Estimating the Coherence of Noise
           Wallman et al.,
           New Journal of Physics 17, 113020 (2015)
           https://dx.doi.org/10.1088/1367-2630/17/11/113020
           https://arxiv.org/abs/1503.07865

    :param bm: object returned from get_benchmarker() used to generate clifford sequences
    :param qubits: the qubits for a single isolated rb experiment (for simultaneous rb,
        create multiple experiments and run them simultaneously)
    :param depths: the depths of the sequences in the experiment
    :param num_sequences: the number of sequences at each depth
    :param use_self_inv_seqs: if True, the last inverting gate of the sequence will be left off.
        This may allow the sequence to also be interpreted as a regular RB sequence.
    :param random_seed: Random seed passed to bm to seed sequence generation.
    :return: a StratifiedExperiment with all programs and information necessary for data
        collection in a unitarity experiment.
    """
    expts = []  # we will have len(depths)*num_sequences many experiments
    sequences = []
    for depth in depths:
        for idx in range(num_sequences):
            parallel_sequences = []
            parallel_seq_settings = []
            for group in qubits:
                if random_seed is not None:  # need to change the base seed for each sequence generated
                    random_seed += 1

                if use_self_inv_seqs:
                    sequence = generate_rb_sequence(bm, group, depth, random_seed=random_seed)
                else:  # provide larger depth and strip inverse from end of each sequence
                    sequence = generate_rb_sequence(bm, group, depth + 1,
                                                    random_seed=random_seed)[:-1]

                parallel_sequences.append(sequence)
                parallel_seq_settings += list(_state_tomo_settings(group))

            program = merge_programs(merge_sequences(parallel_sequences))
            expt = ObservablesExperiment(parallel_seq_settings, program)
            expt = group_experiment_settings(expt)
            expts.append(expt)
            sequences.append(parallel_sequences)

    return expts, sequences


def estimate_purity(dim: int, op_expect: np.ndarray, renorm: bool=True):
    """
    The renormalized, or 'shifted', purity is given in equation (10) of [ECN]
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
    Wrapper for fitting the results of a unitarity experiment; calculates shifted purities
    and passes these on to the standard fit.

    The estimate for the unitarity (the decay) can be found in the returned fit.params['decay']

    :param param_guesses: guesses for the (amplitude, decay, baseline) parameters
    :return: a ModelResult fit with estimates of the Model parameters, including the 'decay',
        which is the unitarity parameter. Note that [ECN] parameterizes the decay differently;
        effectively, the 'amplitude' reported here absorbs a factor 1/unitarity.
        Comparing to 'B' in equation 8), fit.params['amplitude'] = B / fit.params['decay']
    """
    shifted_purities = []
    shifted_purity_errs = []

    assert len(depths) == len(expectations), 'There should be one expectation per sequence. The ' \
                                             'depths used in generate_experiment will need to be ' \
                                             'repeated for the appropriate number of sequences.'
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

    min_non_zero = min([v for v in shifted_purity_errs if v > 0])
    # TODO: does this handle 0 var appropriately? Incorporate unbiased prior into std_err estimate?
    non_zero_err = np.asarray([v if v > 0 else min_non_zero for v in shifted_purity_errs])

    weights = 1 / non_zero_err

    return fit_base_param_decay(np.asarray(depths), np.asarray(shifted_purities), weights,
                                param_guesses)


def unitarity_to_rb_decay(unitarity, dimension) -> float:
    """
    This allows comparison of measured unitarity and RB decays. This function provides an upper bound on the
    RB decay given the input unitarity, where the upperbound is saturated when no unitary errors are present,
    e.g. in the case of depolarizing noise. For more, see Proposition 8. in [ECN]
        unitarity >= (1-dr/(d-1))^2
    where r is the average gate infidelity and d is the dimension

    :param unitarity: The measured decay parameter in a unitarity measurement
    :param dimension: The dimension of the Hilbert space, 2^num_qubits
    :return: The upperbound on RB decay, saturated if no unitary errors are present Proposition 8 [ECN]
    """
    r = (np.sqrt(unitarity) - 1)*(1-dimension)/dimension
    return average_gate_infidelity_to_rb_decay(r, dimension)


########
# Interleaved RB Analysis
########


def coherence_angle(rb_decay, unitarity):
    """
    Equation 29 of [U+IRB]

    :param rb_decay: Observed decay parameter in standard rb experiment
    :param unitarity: Observed decay parameter in unitarity experiment
    :return: coherence angle
    """
    return np.arccos(rb_decay / np.sqrt(unitarity))


def gamma(irb_decay, unitarity):
    """
    Corollary 5 of [U+IRB], second line

    :param irb_decay: Observed decay parameter in irb experiment with desired gate interleaved between Cliffords
    :param unitarity: Observed decay parameter in unitarity experiment
    :return: gamma
    """
    return irb_decay/np.sqrt(unitarity)


def interleaved_gate_fidelity_bounds(irb_decay, rb_decay, dim, unitarity = None):
    """
    Use observed rb_decay to place a bound on fidelity of a particular gate with given interleaved rb decay.
    Optionally, use unitarity measurement result to provide improved bounds on the interleaved gate's fidelity.

    Bounds due to
    [IRB] Efficient measurement of quantum gate error by interleaved randomized benchmarking
          Magesan et al.,
          Phys. Rev. Lett. 109, 080505 (2012)
          https://dx.doi.org/10.1103/PhysRevLett.109.080505
          https://arxiv.org/abs/1203.4550

    Improved bounds using unitarity due to
    [U+IRB]  Efficiently characterizing the total error in quantum circuits
             Dugas et al.,
             arXiv:1610.05296 (2016)
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
        fidelity_bounds = [rb_decay_to_gate_fidelity(decay, dim) for decay in decay_bounds]

    else:
        # Equation 5 of [IRB]

        E1 = (abs(rb_decay - irb_decay/rb_decay) + (1-rb_decay)) * (dim-1)/dim
        E2 = 2*(dim**2 - 1)*(1-rb_decay)/(rb_decay*dim**2) + 4*np.sqrt(1-rb_decay)*np.sqrt(dim**2-1)/rb_decay

        E = min(E1,E2)
        infidelity = irb_decay_to_gate_infidelity(irb_decay, rb_decay, dim)

        fidelity_bounds = [1-infidelity-E, 1-infidelity+E]

    return fidelity_bounds


def gate_infidelity_to_irb_decay(irb_infidelity, rb_decay, dim):
    """
    For convenience, inversion of Eq. 4 of [IRB]. See irb_decay_to_infidelity

    :param irb_infidelity: Infidelity of the interleaved gate.
    :param rb_decay: Observed decay parameter in standard rb experiment.
    :param dim: Dimension of the Hilbert space, 2**num_qubits
    :return: Decay parameter in irb experiment with relevant gate interleaved between Cliffords
    """
    return (1 - irb_infidelity * (dim/(dim-1)) ) * rb_decay


def irb_decay_to_gate_infidelity(irb_decay, rb_decay, dim):
    """
    Eq. 4 of [IRB], which provides an estimate of the infidelity of the interleaved gate,
    given both the observed interleaved and standard decay parameters.

    :param irb_decay: Observed decay parameter in irb experiment with desired gate interleaved between Cliffords
    :param rb_decay: Observed decay parameter in standard rb experiment.
    :param dim: Dimension of the Hilbert space, 2**num_qubits
    :return: Estimated gate infidelity (1 - fidelity) of the interleaved gate.
    """
    return ((dim - 1) / dim) * (1 - irb_decay / rb_decay)


def average_gate_infidelity_to_rb_decay(gate_infidelity, dimension):
    """
    Inversion of eq. 5 of [RB] arxiv paper.

    :param gate_infidelity: The average gate infidelity.
    :param dimension: Dimension of the Hilbert space, 2^num_qubits
    :return: The RB decay corresponding to the gate_infidelity
    """
    return (gate_infidelity - 1 + 1/dimension)/(1/dimension -1)


def rb_decay_to_gate_fidelity(rb_decay, dimension):
    """
    Derived from eq. 5 of [RB] arxiv paper. Note that 'gate' here typically means an element of the Clifford group,
    which comprise standard rb sequences.

    :param rb_decay: Observed decay parameter in standard rb experiment.
    :param dimension: Dimension of the Hilbert space, 2**num_qubits
    :return: The gate fidelity corresponding to the input decay.
    """
    return 1/dimension - rb_decay*(1/dimension -1)
