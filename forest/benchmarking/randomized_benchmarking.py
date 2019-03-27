from math import pi
from typing import Iterable, List, Tuple, Sequence

import numpy as np
from lmfit import Model
from numpy import bincount
from numpy import pi
from pyquil.operator_estimation import measure_observables, \
    TomographyExperiment as PyQuilTomographyExperiment
from scipy.stats import beta

from pyquil.api import BenchmarkConnection, QuantumComputer, get_benchmarker
from pyquil.gates import CZ, RX, RZ
from pyquil.quilbase import Gate
from pyquil.quil import merge_programs
from pyquil import Program
from pyquil.quilatom import QubitPlaceholder

from forest.benchmarking.tomography import _state_tomo_settings
from forest.benchmarking.tomography import generate_state_tomography_experiment, acquire_tomography_data
from forest.benchmarking.compilation import basic_compile

from dataclasses import dataclass


bm = get_benchmarker()


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
    # add list of ExperimentSettings here? Good for unitarity and RPE, though not necessary for RB.
    # Would allow natural use of measure_observables, labeling measurement types, symmetrized ro.
    # Perhaps a component should actually only house a single ExperimentSetting and its
    # corresponding ExperimentResult (+ metadata)? The distinction is whether to think of a
    # component as a single sequence being measured with different operators or as a single
    # runnable program with a single expectation.
    measure_qubits: Tuple[int]
    num_shots: int = None
    results: np.ndarray = None  # raw shots for rb, but list of ExperimentResults for unitarity
    mean: int = None  # currently also houses purity for unitarity
    stddev: int = None  # currently alos houses purity_error for unitarity


    def __str__(self):
        return '[' + ', '.join([str(instr) for instr in self.sequence[0]]) + '] ... [' + \
               ', '.join([str(instr) for instr in self.sequence[-1]]) + ']'


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

    :param depth: The total number of Cliffords in the sequence (including inverse)
    :param random_seed: Random seed passed to compiler to seed sequence generation.
    :return:
    """
    if depth < 2:
        raise ValueError("Sequence depth must be at least 2 for rb sequences, or at least 1 for "
                         "unitarity sequences.")
    gateset = get_rb_gateset(qubits)
    programs = bm.generate_rb_sequence(depth=depth, gateset=gateset, interleaver=interleaved_gate,
                                       seed=random_seed)
    return programs


def generate_rb_experiment(qubits: Sequence[int], depths: List[int], num_sequences: int,
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
    layers = []  # we will have len(depths) many layers, one layer per depth.
    for depth in depths:
        components = []  # each layer will have num_sequences many components.
        for _ in range(num_sequences):
            if random_seed is not None:  # need to change the base seed for each sequence generated
                random_seed += 1

            # a sequence is just a list of Cliffords, with last Clifford inverting the sequence
            sequence = generate_rb_sequence(qubits, depth, interleaved_gate, random_seed)
            components.append(Component(tuple(sequence), qubits))
        layers.append(Layer(depth, components))

    exp_type = "RB"
    if interleaved_gate is not None:
        exp_type = "I" + exp_type  #interleaved rb.

    return StratifiedExperiment(tuple(layers), tuple(qubits), exp_type)


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


def survival_statistics(bitstrings):
    """
    Calculate the mean and variance of the estimated probability of the ground state given shot
    data on one or more bits.

    For binary classified data with N counts of 1 and M counts of 0, these
    can be estimated using the mean and variance of the beta distribution beta(N+1, M+1) where the
    +1 is used to incorporate an unbiased Bayes prior.

    :param ndarray bitstrings: A 2D numpy array of repetitions x bit-arrays.
    :return: (survival mean, sqrt(survival variance))
    """
    survived = np.sum(bitstrings, axis=1) == 0

    # count obmurrences of 000...0 and anything besides 000...0
    n_died, n_survived = bincount(survived, minlength=2)

    # mean and variance given by beta distribution with a uniform prior
    survival_mean = beta.mean(n_survived + 1, n_died + 1)
    survival_var = beta.var(n_survived + 1, n_died + 1)
    return survival_mean, np.sqrt(survival_var)


# do we want num_shots specified at run time or experiment creation?
def acquire_rb_data(qc, experiments: Sequence[StratifiedExperiment], num_shots: int = 500):
    """
    Takes in StratifiedExperiments and simultaneously runs individual Components of separate
    experiments that are in Layers of equal depth.

    Currently assumes all experiments act on different qubits, that each experiment is
    comprised of layers with the same depths, and that each layer is comprised of the same number of
    components.

    :param qc:
    :param experiments:
    :param num_shots:
    :return: Currently mutates the input StratifiedExperiments.
    """
    # make a copy of each experiment, or return a separate results dataclass?
    # copies = [copy.deepcopy(expt) for expt in experiments]

    if not isinstance(experiments, Sequence):
        experiments = [experiments]

    # iterate over all of the layers of equal depth together.
    for layers in zip(*[expt.layers for expt in experiments]):
        # assert len(layers) == len(experiments)
        # assert all([layer.depth = layers[0].depth for layer in layers ])

        # iterate over each slice of components across the combined layers.
        for components in zip(*[layer.components for layer in layers]):
            # merge each of the sequences of all the components so that the nth gate of each
            # sequence happens in the merged sequence at the nth step. (can't just sum sequences)
            sequence = merge_sequences([component.sequence for component in components])
            measure_qubits = [q for component in components for q in component.measure_qubits]

            # basic compile probably not necessary here, but in general may be for other modules.
            program = basic_compile(merge_programs(sequence))
            # add measurements of the collection of qubits across all the components
            ro = program.declare("ro", "BIT", len(measure_qubits))
            for idx, qubit in enumerate(measure_qubits):
                program.measure(qubit, ro[idx])
            program.wrap_in_numshots_loop(num_shots)
            # run the merged sequence program
            exe = qc.compiler.native_quil_to_executable(program)
            results = qc.run(exe)

            # separate the results and store them in the respective components,
            # along with mean and stddev
            offset = 0
            for component in components:
                component.num_shots = num_shots
                component.results = results[:, offset: offset + len(component.measure_qubits)]
                component.mean, component.stddev = survival_statistics(component.results)

    # return copies? return separate Results datastructure similar to ExperimentResults?


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
    depths = [layer.depth for layer in experiment.layers for _ in layer.components]
    survivals = [comp.mean for layer in experiment.layers for comp in layer.components]
    weights = [1/comp.stddev for layer in experiment.layers for comp in layer.components]
    return fit_standard_rb(depths, survivals, np.asarray(weights))


########
# Unitarity
########


def generate_unitarity_experiment(qubits: Sequence[int], depths: List[int], num_sequences: int,
                           random_seed: int = None) -> StratifiedExperiment:
    """
    Essentially the same as generate_rb_experiment, just stripping off last gate, no option for
    interleaving.

    :param qubits: qubits for a single isolated unitary rb experiment
    :param depths:
    :param num_sequences:
    :param random_seed:
    :return:
    """
    layers = []
    for depth in depths:
        components = []
        for _ in range(num_sequences):
            if random_seed is not None:
                random_seed += 1

            sequence = generate_rb_sequence(qubits, depth + 1, random_seed)[:-1]  # strip inverse
            components.append(Component(tuple(sequence), qubits))
        layers.append(Layer(depth, components))

    exp_type = "URB"

    return StratifiedExperiment(tuple(layers), tuple(qubits), exp_type)


def estimate_purity(D: int, op_expect: np.ndarray, renorm: bool=True):
    """
    The renormalized, or 'shifted', purity is given in equation (10) of [ECN]
    where d is the dimension of the Hilbert space, 2**num_qubits

    :param D: dimension of the hilbert space
    :param op_expect: array of estimated expectations of each operator being measured
    :param renorm: flag that renormalizes result to be between 0 and 1
    :return: purity given the operator expectations
    """
    purity = (1 / D) * (1 + np.sum(op_expect * op_expect))
    if renorm:
        purity = (D / (D - 1.0)) * (purity - 1.0 / D)
    return purity


def estimate_purity_err(D: int, op_expect: np.ndarray, op_expect_var: np.ndarray, renorm=True):
    """
    Propagate the observed variance in operator expectation to an error estimate on the purity.
    This assumes that each operator expectation is independent.

    :param D: dimension of the Hilbert space
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

    purity_var = (1 / D) ** 2 * (np.sum(var_of_square_op_expect))

    if renorm:
        purity_var = (D / (D - 1.0)) ** 2 * purity_var

    return np.sqrt(purity_var)


def acquire_unitarity_data(qc, experiments, num_shots: int = 500):
    """


    :param qc:
    :param experiments:
    :param num_shots:
    :return:
    """
    # make a copy of each experiment, or return a separate results dataclass?
    # copies = [copy.deepcopy(expt) for expt in experiments]

    if not isinstance(experiments, Sequence):
        experiments = [experiments]

    for layers in zip(*[expt.layers for expt in experiments]):
        for components in zip(*[layer.components for layer in layers]):
            sequence = merge_sequences([component.sequence for component in components])
            program = basic_compile(merge_programs(sequence))
            # diverges from standard rb at this point

            # we need to essentially perform a simultaneous tomography experiment for each of the
            # component sequences, where each pauli observable of the same type is run
            # simultaneously on different qubits. (This is notably NOT the same as one big
            # tomography experiment on all of the qubits, which has exponential scaling)
            # Unfortunately, we can't currently run different PyquilTomographyExperiments
            # simultaneously with measure_observables--we can only run different ExperimentSettings
            # within the SAME PyquilTomographyExperiment. Instead of making a bunch of
            # TomographyExperiments we directly work with the exp_settings that comprise each.

            # handles mixing 1q and 2q experiments simultaneously.
            max_num_settings = 4**max([len(component.measure_qubits) for component in components])

            # the inner list here will consist of experiments with the same observable on
            # different qubits. (not exactly if 1q and 2q or mixed, but same idea)
            parallel_settings = [[] for _ in range(max_num_settings)]
            # this will help us keep track of which components a result should be associated with.
            indices = [[] for _ in range(max_num_settings)]
            qubits = []
            # get all of the exp_settings that would comprise a tomography experiment on each
            # component. Add each component to its corresponding inner list of parallel_settings.
            for idx, component in enumerate(components):
                exp_settings = _state_tomo_settings(component.measure_qubits)
                for setting, settings, idxs in zip(exp_settings, parallel_settings, indices):
                    settings.append(setting)  # group parallel settings
                    idxs.append(idx)
                qubits.append(component.measure_qubits)

            # this is a TomographyExperiment which runs each inner list of parallel_settings in
            # parallel for the program.
            tomo_expt = PyQuilTomographyExperiment(settings=parallel_settings, program=program,
                                                  qubits=qubits)
            # this is a flat list of results, no grouping is given. This is why we made 'indices'
            expt_results = measure_observables(qc, tomo_expt, num_shots)

            # regroup each result into its corresponding component.
            component_results = [[] for _ in components]
            for comp_idx, result in zip([idx for group in indices for idx in group], expt_results):
                component_results[comp_idx].append(result)

            for component, results in zip(components, component_results):
                # results aren't quite the same as standard rb. We /could/ cast standard rb in the
                # form of ExperimentSettings and use measure observables for the sake of unity.
                component.results = results[1:]
                exps = np.asarray([res.expectation for res in results[1:]])
                variances = np.asarray([res.stddev**2 for res in results[1:]])
                # the intermediate result is not a mean, but this is semantically where it should go
                component.mean  = estimate_purity(2**len(component.measure_qubits), exps)
                component.stddev  = estimate_purity_err(2**len(component.measure_qubits), exps,
                                                        variances)

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
    # almost identitical to fit_rb_results, could probably be combined by examining the expt.type
    depths = [layer.depth for layer in experiment.layers for _ in layer.components]
    shifted_purities = [comp.mean for layer in experiment.layers for comp in layer.components]
    weights = [1/comp.stddev for layer in experiment.layers for comp in layer.components]
    return fit_unitarity(depths, shifted_purities, np.asarray(weights))


def unitarity_to_RB_decay(unitarity, dimension):
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
        fidelity_bounds = [RB_decay_to_gate_fidelity(decay, dim) for decay in decay_bounds]

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


def average_gate_infidelity_to_RB_decay(gate_infidelity, dimension):
    """
    Inversion of eq. 5 of [RB] arxiv paper.

    :param gate_infidelity: The average gate infidelity.
    :param dimension: Dimension of the Hilbert space, 2^num_qubits
    :return: The RB decay corresponding to the gate_infidelity
    """
    return (gate_infidelity - 1 + 1/dimension)/(1/dimension -1)


def RB_decay_to_gate_fidelity(rb_decay, dimension):
    """
    Derived from eq. 5 of [RB] arxiv paper. Note that 'gate' here typically means an element of the Clifford group,
    which comprise standard rb sequences.

    :param rb_decay: Observed decay parameter in standard rb experiment.
    :param dimension: Dimension of the Hilbert space, 2**num_qubits
    :return: The gate fidelity corresponding to the input decay.
    """
    return 1/dimension - rb_decay*(1/dimension -1)
