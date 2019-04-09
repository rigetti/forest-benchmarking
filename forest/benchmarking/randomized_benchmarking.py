from math import pi
from typing import Iterable, List, Tuple, Sequence

import numpy as np
from lmfit import Model
from numpy import pi
from scipy.stats import beta

from pyquil.api import BenchmarkConnection, QuantumComputer, get_benchmarker
from pyquil.gates import CZ, RX, RZ
from pyquil.quilbase import Gate
from pyquil import Program
from pyquil.operator_estimation import ExperimentSetting, zeros_state

from forest.benchmarking.tomography import _state_tomo_settings
from forest.benchmarking.utils import all_pauli_z_terms
from forest.benchmarking.compilation import basic_compile
from forest.benchmarking.stratified_experiment import StratifiedExperiment, Layer, \
    acquire_stratified_data


bm = get_benchmarker()


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

    :param rb_type: "1q" or "2q".
    :returns: list of gates, tuple of qubits
    """
    if len(qubits) == 1:
        return list(oneq_rb_gateset(qubits[0]))

    if len(qubits) == 2:
        return list(twoq_rb_gateset(*qubits))

    raise ValueError(f"No RB gateset for more than two qubits.")


def generate_rb_sequence(qubits: Sequence[int], depth: int,  interleaved_gate: Program = None,
                         random_seed: int = None) -> List[Program]:
    """
    Generate a complete randomized benchmarking sequence.

    :param qubits:
    :param depth: The total number of Cliffords in the sequence (including inverse)
    :param random_seed: Random seed passed to compiler to seed sequence generation.
    :param interleaved_gate:
    :return:
    """
    if depth < 2:
        raise ValueError("Sequence depth must be at least 2 for rb sequences, or at least 1 for "
                         "unitarity sequences.")
    gateset = get_rb_gateset(qubits)
    programs = bm.generate_rb_sequence(depth=depth, gateset=gateset, interleaver=interleaved_gate,
                                       seed=random_seed)
    return programs


def generate_rb_experiment(qubits: Sequence[int], depths: Sequence[int], num_sequences: int,
                           interleaved_gate: Program = None, random_seed: int = None) \
        -> StratifiedExperiment:
    """

    :param qubits: the qubits for a single isolated rb experiment
    :param depths:
    :param num_sequences:
    :param interleaved_gate:
    :param random_seed:
    :return:
    """
    layers = []  # we will have len(depths)*num_sequences many layers, one layer per sequence.
    for depth in depths:
        for idx in range(num_sequences):
            if random_seed is not None:  # need to change the base seed for each sequence generated
                random_seed += 1

            # a sequence is just a list of Cliffords, with last Clifford inverting the sequence
            sequence = generate_rb_sequence(qubits, depth, interleaved_gate, random_seed)
            settings = [ExperimentSetting(zeros_state(qubits), op)
                        for op in all_pauli_z_terms(qubits)]
            layers.append(Layer(depth, tuple(sequence), tuple(settings), tuple(qubits),
                                f'Seq{idx}'))

    expt_type = "RB"
    if interleaved_gate is not None:
        expt_type = "I" + expt_type  #interleaved rb.

    return StratifiedExperiment(tuple(layers), tuple(qubits), expt_type)


def generate_unitarity_experiment(qubits: Sequence[int], depths: Sequence[int], num_sequences: int,
                                  use_self_inv_seqs = False, random_seed: int = None) \
        -> StratifiedExperiment:
    """
    Essentially the same as generate_rb_experiment, just stripping off last gate, no option for
    interleaving.

    :param qubits: qubits for a single isolated unitary rb experiment
    :param depths:
    :param num_sequences:
    :param use_self_inv_seqs:
    :param random_seed:
    :return:
    """
    layers = []  # we will have len(depths)*num_sequences many layers, one layer per sequence.
    for depth in depths:
        for idx in range(num_sequences):
            if random_seed is not None:
                random_seed += 1

            if use_self_inv_seqs:
                sequence = generate_rb_sequence(qubits, depth, random_seed)
            else:  # provide larger depth and strip inverse from end of each sequence
                sequence = generate_rb_sequence(qubits, depth + 1, random_seed)[:-1]
            settings = _state_tomo_settings(qubits)
            layers.append(Layer(depth, tuple(sequence), tuple(settings), tuple(qubits),
                                f'Seq{idx}'))

    expt_type = "URB"

    return StratifiedExperiment(tuple(layers), tuple(qubits), expt_type)


def populate_rb_survival_statistics(expt: StratifiedExperiment):
    """
    Calculate the mean and variance of the estimated probability of the zeros state given the
    expectation of all operators with Z terms.

    We first convert the operator expectation values into a density matrix, then take the (0, 0)
    component of this matrix as the probability of the all zeros state. For binary classified data
    with N counts of 1 and M counts of 0, these can be estimated using the mean and variance of
    the beta distribution beta(N+1, M+1) where the +1 is used to incorporate an unbiased Bayes
    prior.

    :param expt: A StratifiedExperiment whose Components have been populated with results.
    :return:
    """
    for layer in expt.layers:
        # exclude operators that include x or y; no change for a standard rb experiment
        z_terms = [result for result in layer.results
                   if 'X' not in result.setting.out_operator.compact_str()
                   and 'Y' not in result.setting.out_operator.compact_str()]

        # This assumes inclusion of I term with expectation 1 to make dim many total terms
        assert 2**len(layer.qubits) == len(z_terms)
        # get the fraction of all zero outcomes 00...00
        fraction_survived = np.mean([result.expectation for result in z_terms])
        num_survived = fraction_survived * layer.num_shots
        num_died =  layer.num_shots - num_survived  # the number of non-zero results

        # mean and variance given by beta distribution with a uniform prior
        survival_mean = beta.mean(num_survived + 1, num_died + 1)
        survival_var = beta.var(num_survived + 1, num_died + 1)

        survival_stats = {"Survival": (survival_mean, np.sqrt(survival_var))}
        layer.estimates = survival_stats


def populate_unitarity_purity_statistics(expt: StratifiedExperiment):
    """
    Calculate the mean and variance of the estimated probability of the zeros state given the
    expectation of all operators with Z terms.

    We first convert the operator expectation values into a density matrix, then take the (0, 0)
    component of this matrix as the probability of the all zeros state. For binary classified data
    with N counts of 1 and M counts of 0, these can be estimated using the mean and variance of
    the beta distribution beta(N+1, M+1) where the +1 is used to incorporate an unbiased Bayes
    prior.

    :param expt: A StratifiedExperiment whose Components have been populated with results.
    :return:
    """
    for layer in expt.layers:
        # This assumes inclusion of I term with expectation 1 to make dim**2 many total terms
        dim = 2**len(layer.qubits)
        assert dim**2 == len(layer.results), "Ensure identity term is included."

        expectations = np.array([result.expectation for result in layer.results])
        variances = np.array([result.stddev**2 for result in layer.results])

        shifted_purity = estimate_purity(dim, expectations)
        shifted_purity_error = estimate_purity_err(dim, expectations, variances)
        shifted_purity_stats = {"Shifted Purity": (shifted_purity, shifted_purity_error)}
        layer.estimates = shifted_purity_stats


def acquire_rb_data(qc, experiments: Sequence[StratifiedExperiment], num_shots: int = 500):
    if not isinstance(experiments, Sequence):
        experiments = [experiments]
    compile_method = qc.compiler.quil_to_native_quil
    qc.compiler.quil_to_native_quil = basic_compile
    acquire_stratified_data(qc, experiments, num_shots)
    qc.compiler.quil_to_native_quil = compile_method  # restore the original compilation
    for expt in experiments:
        populate_rb_survival_statistics(expt) # populate with relevant estimates


def acquire_unitarity_data(qc, experiments: Sequence[StratifiedExperiment], num_shots: int = 500):
    if not isinstance(experiments, Sequence):
        experiments = [experiments]
    compile_method = qc.compiler.quil_to_native_quil
    qc.compiler.quil_to_native_quil = basic_compile
    acquire_stratified_data(qc, experiments, num_shots)
    qc.compiler.quil_to_native_quil = compile_method  # restore the original compilation
    for expt in experiments:
        populate_unitarity_purity_statistics(expt) # populate with relevant estimates


def standard_rb(x, baseline, amplitude, decay):
    """
    Fitting function for randomized benchmarking.

    :param numpy.ndarray x: Independent variable
    :param float baseline: Offset value
    :param float amplitude: Amplitude of exponential decay
    :param float decay: Decay parameter
    :return: Fit function
    """
    return baseline + amplitude * decay ** x


def standard_rb_guess(model: Model, y):
    """
    Guess the parameters for a fit.

    :param model: an lmfit model to make guess parameters for. This should probably be an
        instance of ``Model(standard_rb)``.
    :param y: Dependent variable
    :return: Lmfit parameters object appropriate for passing to ``Model.fit()``.
    """
    b_guess = y[-1]
    a_guess = y[0] - y[-1]
    d_guess = 0.95
    return model.make_params(baseline=b_guess, amplitude=a_guess, decay=d_guess)


def _check_data(x, y, weights):
    if not len(x) == len(y):
        raise ValueError("Lengths of x and y arrays must be equal.")
    if weights is not None and not len(x) == len(weights):
        raise ValueError("Lengths of x and weights arrays must be equal is weights is not None.")


def fit_standard_rb(depths, survivals, weights=None):
    """
    Construct and fit an RB curve with appropriate guesses

    :param depths: The clifford circuit depths (independent variable)
    :param survivals: The survival probabilities (dependent variable)
    :param weights: Optional weightings of each point to use when fitting.
    :return: an lmfit Model
    """
    _check_data(depths, survivals, weights)
    rb_model = Model(standard_rb)
    params = standard_rb_guess(model=rb_model, y=survivals)
    return rb_model.fit(survivals, x=depths, params=params, weights=weights)


def fit_rb_results(experiment):
    """
    Wrapper for plotting the results of a StratifiedExperiment; simply extracts key parameters
    and passes on to the standard fit.

    :param experiment:
    :return:
    """
    depths = []
    survivals = []
    weights = []
    for layer in experiment.layers:
        depths.append(layer.depth)
        survivals.append(layer.estimates["Survival"][0])
        weights.append(1/layer.estimates["Survival"][1])
    return fit_standard_rb(depths, survivals, np.asarray(weights))


########
# Unitarity
########


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
    purity = (1 / dim) * np.sum(op_expect * op_expect)
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


def unitarity_fn(x, baseline, amplitude, unitarity):
    """
    Fitting function for unitarity randomized benchmarking, equation (8) of [ECN]

    :param numpy.ndarray x: Independent variable
    :param float baseline: Offset value
    :param float amplitude: Amplitude of exponential decay
    :param float decay: Decay parameter
    :return: Fit function
    """
    return baseline + amplitude * unitarity ** (x-1)

#TODO: confirm validity or update guesses
def unitarity_guess(model: Model, y):
    """
    Guess the parameters for a fit.

    :param model: an lmfit model to make guess parameters for. This should probably be an
        instance of ``Model(unitarity)``.
    :param y: Dependent variable
    :return: Lmfit parameters object appropriate for passing to ``Model.fit()``.
    """
    b_guess = 0.
    a_guess = y[0]
    d_guess = 0.95
    return model.make_params(baseline=b_guess, amplitude=a_guess, unitarity=d_guess)


def fit_unitarity(depths, shifted_purities, weights=None):
    """Construct and fit an RB curve with appropriate guesses

    :param depths: The clifford circuit depths (independent variable)
    :param shifted_purities: The shifted purities (dependent variable)
    :param weights: Optional weightings of each point to use when fitting.
    :return: an lmfit Model
    """
    _check_data(depths, shifted_purities, weights)
    unitarity_model = Model(unitarity_fn)
    params = unitarity_guess(model=unitarity_model, y=shifted_purities)
    return unitarity_model.fit(shifted_purities, x=depths, params=params, weights=weights)


def fit_unitarity_results(experiment):
    depths = []
    shifted_purities = []
    weights = []
    for layer in experiment.layers:
        depths.append(layer.depth)
        shifted_purities.append(layer.estimates["Shifted Purity"][0])
        weights.append(1/layer.estimates["Shifted Purity"][1])
    return fit_unitarity(depths, shifted_purities, np.asarray(weights))


def unitarity_to_rb_decay(unitarity, dimension):
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
    return average_gate_infidelity_to_RB_decay(r, dimension)


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
            Magesan et al., Physical Review Letters 109, 080505 (2012)
            arXiv:1203.4550

    Improved bounds using unitarity due to:
        [U+IRB]  Efficiently characterizing the total error in quantum circuits
            Dugas, Wallman, and Emerson (2016)
            arXiv:1610.05296v2

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
