from typing import Sequence, List, Dict, Tuple

import numpy as np
from numpy import pi
from lmfit.model import ModelResult
from tqdm import tqdm

from pyquil.api import QuantumComputer
from pyquil.gates import RX, RY, RZ, CZ, MEASURE
from pyquil.quil import Program
from pyquil.quilbase import Pragma
from pyquil.paulis import PauliTerm

from forest.benchmarking.utils import transform_pauli_moments_to_bit
from forest.benchmarking.analysis.fitting import fit_decay_time_param_decay, \
    fit_decaying_cosine, fit_shifted_cosine
from forest.benchmarking.observable_estimation import ObservablesExperiment, ExperimentResult, \
    ExperimentSetting, estimate_observables, minusZ, plusZ, minusY

MICROSECOND = 1e-6  # A microsecond (us) is an SI unit of time

# A Hertz (Hz) is a derived unit of frequency in SI Units; 1 Hz is defined as one cycle per second.
MHZ = 1e6  # MHz, megahertz


def acquire_qubit_spectroscopy_data(qc: QuantumComputer,
                                    experiments: Sequence[ObservablesExperiment],
                                    num_shots: int = 500, show_progress_bar: bool = False) \
        -> List[List[ExperimentResult]]:
    """
    A standard data acquisition method for all experiments in this module.

    Each input ObservablesExperiment is simply run in series, and a list of results are returned
    for each experiment in the corresponding order.

    :param qc: a quantum computer on which to run the experiments
    :param experiments: the ObservablesExperiments to run on the given qc
    :param num_shots: the number of shots to collect for each experiment.
    :param show_progress_bar: displays a progress bar via tqdm if true.
    :return: a list of ExperimentResults for each ObservablesExperiment, returned in order of the
        input sequence of experiments.
    """
    results = []
    for expt in tqdm(experiments, disable=not show_progress_bar):
        results.append(list(estimate_observables(qc, expt, num_shots)))
    return results


def get_stats_by_qubit(expt_results: List[List[ExperimentResult]]) \
        -> Dict[int, Dict[str, List[float]]]:
    """
    Organize the mean and std_err of a single-observable experiment by qubit.

    Each individual experiment result is assumed to consist of a single qubit observable. The
    inner list of observables should be a group of such experiments acting on different qubits.
    The outer list dictates the order of the returned statistics for each qubit. For example,
    in a T1 experiment the outer list of results is indexed by increasing time; the returned
    statistics will also be arranged in order of increasing time for each qubit separately.

    :param expt_results:
    :return: a dictionary indexed by qubit label, where each value is itself a dictionary with
        'expectation' and 'std_err' values for the given qubit.
    """
    stats_by_qubit = {}
    for results in expt_results:
        for res in results:
            qubits = res.setting.observable.get_qubits()
            if len(qubits) > 1:
                raise ValueError("This method is intended for single qubit observables.")
            qubit = qubits[0]

            if qubit not in stats_by_qubit:
                stats_by_qubit[qubit] = {'expectation': [], 'std_err': []}

            stats_by_qubit[qubit]['expectation'].append(res.expectation)
            stats_by_qubit[qubit]['std_err'].append(res.std_err)

    return stats_by_qubit


# ==================================================================================================
#   T1
# ==================================================================================================


def generate_t1_experiments(qubits: Sequence[int], times: Sequence[float]) \
        -> List[ObservablesExperiment]:
    """
    Return a ObservablesExperiment containing programs which constitute a t1 experiment to
    measure the decay time from the excited state to ground state for each qubit in qubits.

    For each delay time in times a single program will be generated in which all qubits are
    initialized to the excited state (`|1>`) and simultaneously measured after the given delay.

    :param qubits: list of qubits to measure.
    :param times: The times at which to measure, given in seconds. Each time is rounded to the
        nearest .1 microseconds.
    :return: ObservablesExperiments which will measure the decay of each qubit after
        initialization to the 1 state and delay of t seconds for each t in times.
    """
    expts = []
    for t in times:
        t = round(t, 7)  # enforce 100ns boundaries
        program = Program()
        settings = []
        for q in qubits:
            program += Pragma('DELAY', [q], str(t))
            settings.append(ExperimentSetting(minusZ(q), PauliTerm('Z', q)))

        expts.append(ObservablesExperiment([settings], program))

    return expts


def fit_t1_results(times: Sequence[float], z_expectations: Sequence[float],
                   z_std_errs: Sequence[float] = None, param_guesses: tuple = (1.0, 15, 0.0)) \
        -> ModelResult:
    """
    Wrapper for fitting the results of a T1 experiment for a single qubit; simply extracts key
    parameters and passes on to the standard fit.

    The estimate for T1 can be found in the returned fit.params['decay_constant']

    :param times: the times at which the z_expectations were measured. The units of the time
        determine the units of the T1 estimate, decay_constant. Here we set the default guess to
        O(10) which corresponds to the times being given in units of microseconds.
    :param z_expectations: expectation of Z at each time for a qubit initialized to 1
    :param z_std_errs: std_err of the Z expectation, optionally used to weight the fit.
    :param param_guesses: guesses for the (amplitude, decay_constant, offset) parameters. Here
        the default decay_constant of 15 assumes that times are given in units of microseconds.
    :return: a ModelResult fit with estimates of the Model parameters, including the T1
        'decay_constant'
    """
    z_expectations = np.asarray(z_expectations)
    if z_std_errs is not None:
        probability_one, var = transform_pauli_moments_to_bit(np.asarray(-1 * z_expectations),
                                                              np.asarray(z_std_errs)**2)
        err = np.sqrt(var)
        non_zero = [v for v in err if v > 0]
        if len(non_zero) == 0:
            weights = None
        else:
            # TODO: does this handle 0 var appropriately?
            # Other possibility is to use unbiased prior into std_err estimate.
            min_non_zero = min(non_zero)
            non_zero_err = np.asarray([v if v > 0 else min_non_zero for v in err])

            weights = 1 / non_zero_err
    else:
        probability_one, _ = transform_pauli_moments_to_bit(np.asarray(-1 * z_expectations), 0)
        weights = None

    return fit_decay_time_param_decay(np.asarray(times), probability_one, weights,
                                      param_guesses)


def do_t1_or_t2(qc: QuantumComputer, qubits: Sequence[int], times: Sequence[float],
                kind: str, num_shots: int = 500, show_progress_bar: bool = False) \
        -> Tuple[Dict[int, float], List[ObservablesExperiment], List[List[ExperimentResult]]]:
    """
    A wrapper around experiment generation, data acquisition, and estimation that runs a t1,
    t2 echo, or t2* experiment on each qubit in qubits and returns the rb_decay along with the
    experiments and results.

    :param qc: a quantum computer on which to run the experiments
    :param qubits: list of qubits to measure.
    :param times: The times at which to measure, given in seconds. Each time is rounded to the
        nearest .1 microseconds.
    :param kind: which kind of experiment to do, one of 't1', 't2_star', or 't2_echo'
    :param num_shots: the number of shots to collect for each experiment.
    :param show_progress_bar: displays a progress bar via tqdm if true.
    :return:
    """
    if kind.lower() == 't1':
        gen_method = generate_t1_experiments
        fit_method = fit_t1_results
    elif kind.lower() == 't2_star':
        gen_method = generate_t2_star_experiments
        fit_method = fit_t2_results
    elif kind.lower() == 't2_echo':
        gen_method = generate_t2_echo_experiments
        fit_method = fit_t2_results
    else:
        raise ValueError('Kind must be one of \'t1\', \'t2_star\', or \'t2_echo\'.')

    expts = gen_method(qubits, times)
    results = acquire_qubit_spectroscopy_data(qc, expts, num_shots, show_progress_bar)
    stats = get_stats_by_qubit(results)
    decay_time_by_qubit = {}
    for qubit in qubits:
        fit = fit_method(np.asarray(times) / MICROSECOND, stats[qubit]['expectation'],
                         stats[qubit]['std_err'])
        decay_time = fit.params['decay_time'].value  # in us
        decay_time_by_qubit[qubit] = float(decay_time)

    return decay_time_by_qubit, expts, results


# ==================================================================================================
#   T2 star and T2 echo functions
# ==================================================================================================
def generate_t2_star_experiments(qubits: Sequence[int], times: Sequence[float],
                                 detuning: float = 1e6) -> List[ObservablesExperiment]:
    """
    Return ObservablesExperiments containing programs which constitute a T2 star experiment to
    measure the T2 star coherence decay time for each qubit in qubits.

    For each delay time in times a single program will be generated in which all qubits are
    initialized to the minusY state and simultaneously measured along the Y axis after the given
    delay and Z rotation. If the qubit frequency is perfectly calibrated then the Y expectation
    will oscillate at the given detuning frequency as the qubit is rotated about the Z axis (with
    respect to the lab frame, which by hypothesis matches the natural qubit frame).

    :param qubits: list of qubits to measure.
    :param times: the times at which to measure, given in seconds. Each time is rounded to the
        nearest .1 microseconds.
    :param detuning: The additional detuning frequency about the z axis in Hz.
    :return: ObservablesExperiments which can be run to acquire an estimate of T2* for each qubit
    """
    expts = []
    for t in times:
        t = round(t, 7)  # enforce 100ns boundaries
        program = Program()
        settings = []
        for q in qubits:
            program += Pragma('DELAY', [q], str(t))
            program += RZ(2 * pi * t * detuning, q)
            settings.append(ExperimentSetting(minusY(q), PauliTerm('Y', q)))

        expts.append(ObservablesExperiment([settings], program))

    return expts


def generate_t2_echo_experiments(qubits: Sequence[int], times: Sequence[float],
                                 detuning: float = 1e6) -> List[ObservablesExperiment]:
    """
    Return ObservablesExperiments containing programs which constitute a T2 echo experiment to
    measure the T2 echo coherence decay time.

    For each delay time in times a single program will be generated in which all qubits are
    initialized to the minusY state and later simultaneously measured along the Y axis. Unlike in
    the t2_star experiment above there is a 'echo' applied in the middle of the delay in which
    the qubit is rotated by pi radians around the Y axis.

    Similarly to t2_star, if the qubit frequency is perfectly calibrated then the Y expectation
    will oscillate at the given detuning frequency as the qubit is rotated about the Z axis (with
    respect to the lab frame, which by hypothesis matches the natural qubit frame). Unlike in a
    t2_star experiment, even if the qubit frequency is off such that there is some spurious
    rotation about the Z axis during the DELAY, the effect of an ideal echo is to cancel the
    effect of this rotation so that the qubit returns to the initial state minusY before the
    detuning rotation is applied.

    :param qubits: list of qubits to measure.
    :param times: the times at which to measure, given in seconds. Each time is rounded to the
        nearest .1 microseconds.
    :param detuning: The additional detuning frequency about the z axis.
    :return: ObservablesExperiments which can be run to acquire an estimate of T2 for each qubit.
    """
    expts = []
    for t in times:
        half_time = round(t/2, 7)  # enforce 100ns boundaries
        t = round(t, 7)
        program = Program()
        settings = []
        for q in qubits:
            half_delay = Pragma('DELAY', [q], str(half_time))
            # echo
            program += [half_delay, RY(pi, q), half_delay]
            # apply detuning
            program += RZ(2 * pi * t * detuning, q)
            settings.append(ExperimentSetting(minusY(q), PauliTerm('Y', q)))

        expts.append(ObservablesExperiment(settings, program))

    return expts


def fit_t2_results(times: Sequence[float], y_expectations: Sequence[float],
                   y_std_errs: Sequence[float] = None, detuning: float = 1e6,
                   param_guesses: tuple = None) -> ModelResult:
    """
    Wrapper for fitting the results of a ObservablesExperiment; simply extracts key parameters
    and passes on to the standard fit.

    The estimate for T2 can be found in the returned fit.params['decay_constant']

    :param times: the times at which the y_expectations were measured. The units of the time
        determine the units of the T2 estimate, decay_constant. Here we set the default guess to
        O(10) which corresponds to the times being given in units of microseconds.
    :param y_expectations: expectation of Y measured at each time for a qubit
    :param y_std_errs: std_err of the Y expectation, optionally used to weight the fit.
    :param detuning: the detuning specified in creation of the experiment
    :param param_guesses: guesses for the (amplitude, decay_constant, offset, baseline,
        frequency) parameters. The default values assume time is provided in microseconds and
        detuning is provided in HZ, whereas the frequency is reported in MHZ.
    :return: a ModelResult fit with estimates of the Model parameters, including the T2
        'decay_constant'
    """
    if param_guesses is None:  # make some standard reasonable guess
        param_guesses = (.5, 10, 0.0, 0.5, detuning / MHZ)

    y_expectations = np.asarray(y_expectations)
    if y_std_errs is not None:
        probability_one, var = transform_pauli_moments_to_bit(np.asarray(-1 * y_expectations),
                                                              np.asarray(y_std_errs)**2)
        err = np.sqrt(var)
        non_zero = [v for v in err if v > 0]
        if len(non_zero) == 0:
            weights = None
        else:
            # TODO: does this handle 0 var appropriately? 
            # Other possibility is to use unbiased prior in std_err estimate.
            min_non_zero = min(non_zero)
            non_zero_err = np.asarray([v if v > 0 else min_non_zero for v in err])

            weights = 1 / non_zero_err
    else:
        probability_one, _ = transform_pauli_moments_to_bit(np.asarray(-1 * y_expectations), 0)
        weights = None

    return fit_decaying_cosine(np.asarray(times), probability_one, weights, param_guesses)


# ==================================================================================================
#   TODO CPMG
# ==================================================================================================


# ==================================================================================================
#   Rabi
# ==================================================================================================
def generate_rabi_experiments(qubits: Sequence[int], angles: Sequence[float]) \
        -> List[ObservablesExperiment]:
    """
    Return ObservablesExperiments containing programs which constitute a Rabi experiment.

    Rabi oscillations are observed by applying successively larger rotations to the same initial
    state.

    :param qubits: list of qubits to measure.
    :param angles: A list of angles at which to make a measurement
    :return: ObservablesExperiments which can be run to verify the  RX(angle, q) calibration
        for each qubit
    """
    expts = []
    for angle in angles:
        program = Program()
        settings = []
        for q in qubits:
            program += RX(angle, q)
            settings.append(ExperimentSetting(plusZ(q), PauliTerm('Z', q)))

        expts.append(ObservablesExperiment([settings], program))

    return expts


def fit_rabi_results(angles: Sequence[float], z_expectations: Sequence[float],
                     z_std_errs: Sequence[float] = None, param_guesses: tuple = (-.5, 0, .5, 1.)) \
        -> ModelResult:
    """
    Wrapper for fitting the results of a rabi experiment on a qubit; simply extracts key parameters
    and passes on to the standard fit.

    Note the following interpretation of the model fit parameters

    x
        the independent variable is the angle that we specify when writing a gate instruction. If
        our gates are incorrectly calibrated then a given control angle will result in a different
        angle than intended by the multiplicative 'frequency' of the model

    amplitude
        this will have magnitude (p1_given_1 - p1_given_0) / 2 where
        p1_given_1 is the probability of measuring 1 when the qubit is in the `|1>` state.
        p1_given_0 is the probability of measuring 1 when the qubit is in the `|0>` state.

    offset
        this is the offset phase, in radians, with respect to the true rotation frequency.
        e.g. if our gate is mis-calibrated resulting in an offset 'off' then we require a control
        angle of RX(-off / frequency) to correct the offset

    baseline
        this is the amplitude + p1_given_0

    frequency
        the ratio of the actual angle rotated over the intended control angle
        e.g. If our gates are incorrectly calibrated to apply an over-rotation then
        frequency will be greater than 1; the intended control angle will be smaller than the
        true angle rotated.

    :param angles: the angles at which the z_expectations were measured.
    :param z_expectations: expectation of Z at each angle for a qubit initialized to 0
    :param z_std_errs: std_err of the Z expectation, optionally used to weight the fit.
    :param param_guesses: guesses for the (amplitude, offset, baseline, frequency) parameters
    :return: a ModelResult fit with estimates of the Model parameters, including the frequency
        which gives the ratio of actual angle over intended control angle
    """
    z_expectations = np.asarray(z_expectations)
    if z_std_errs is not None:
        probability_one, var = transform_pauli_moments_to_bit(np.asarray(-1 * z_expectations),
                                                              np.asarray(z_std_errs)**2)
        err = np.sqrt(var)
        non_zero = [v for v in err if v > 0]
        if len(non_zero) == 0:
            weights = None
        else:
            # TODO: does this handle 0 var appropriately? 
            # Other possibility is to use unbiased prior in std_err estimate.
            min_non_zero = min(non_zero)
            non_zero_err = np.asarray([v if v > 0 else min_non_zero for v in err])

            weights = 1 / non_zero_err
    else:
        probability_one, _ = transform_pauli_moments_to_bit(np.asarray(-1 * z_expectations), 0)
        weights = None

    return fit_shifted_cosine(np.asarray(angles), probability_one, weights, param_guesses)


# ==================================================================================================
#   CZ phase Ramsey
# ==================================================================================================
def generate_cz_phase_ramsey_experiments(cz_qubits: Sequence[int], measure_qubit: int,
                                         angles: Sequence[float]) -> List[ObservablesExperiment]:
    """
    Return ObservablesExperiments containing programs that constitute a CZ phase ramsey experiment.

    :param cz_qubits: the qubits participating in the cz gate
    :param measure_qubit: Which qubit to measure.
    :param angles: A list of angles at which to make a measurement
    :return: ObservablesExperiments which can be run to estimate the effective RZ rotation
        applied to a single qubit during the application of a CZ gate.
    """
    expts = []
    for angle in angles:
        settings = []
        program = Program()
        # apply CZ, possibly inducing an effective RZ on measure qubit by some angle
        program += CZ(*cz_qubits)
        # apply phase to measure_qubit akin to T2 experiment
        program += RZ(angle, measure_qubit)
        settings = [ExperimentSetting(minusY(measure_qubit), PauliTerm('Y', measure_qubit))]

        expts.append(ObservablesExperiment([settings], program))

    return expts


def fit_cz_phase_ramsey_results(angles: Sequence[float], y_expectations: Sequence[float],
                                y_std_errs: Sequence[float] = None,
                                param_guesses: tuple = (.5, 0, .5, 1.)) -> ModelResult:
    """
    Wrapper for fitting the results of a ObservablesExperiment; simply extracts key parameters
    and passes on to the standard fit.

    Note the following interpretation of the model fit:

    x
        the independent variable is the control angle that we specify when writing a gate
        instruction. If our gates are incorrectly calibrated then a given control angle will
        result in a different angle than intended by the multiplicative 'frequency' of the model

    amplitude
        this will have magnitude (p1_given_1 - p1_given_0) / 2 where
        p1_given_1 is the probability of measuring 1 when the qubit is in the `|1>` state.
        p1_given_0 is the probability of measuring 1 when the qubit is in the `|0>` state.

    offset
        this is the offset phase, in radians, with respect to the true rotation frequency.
        e.g. say that our RZ gate is perfectly calibrated and the CZ gate imparts an effective
        RZ(pi/5) rotation to the measure qubit; in this case offset is pi/5, and frequency is one,
        so the offset phase could be corrected by applying the gate RZ(-pi/5, qubit) after CZ. If
        our RZ gate was instead found to be mis-calibrated, a correction using our mis-calibrated
        RZ gate would require a control angle of RZ(-pi/5 / frequency, qubit)

    baseline
        this is the amplitude + p1_given_0

    frequency
        the ratio of the actual angle rotated over the intended control angle
        e.g. If our gates are incorrectly calibrated to apply an over-rotation then
        frequency will be greater than 1; the intended control angle will be smaller than the
        true angle rotated.

    :param angles: the angles at which the z_expectations were measured.
    :param y_expectations: expectation of Y measured at each time for a qubit
    :param y_std_errs: std_err of the Y expectation, optionally used to weight the fit.
    :param param_guesses: guesses for the (amplitude, offset, baseline, frequency) parameters
    :return: a ModelResult fit with estimates of the Model parameters, including the offset which
        is an estimate of the phase imparted on the measure qubit by the CZ gate.
    """
    y_expectations = np.asarray(y_expectations)
    if y_std_errs is not None:
        probability_one, var = transform_pauli_moments_to_bit(np.asarray(-1 * y_expectations),
                                                              np.asarray(y_std_errs)**2)
        err = np.sqrt(var)
        non_zero = [v for v in err if v > 0]
        if len(non_zero) == 0:
            weights = None
        else:
            # TODO: does this handle 0 var appropriately? 
            # Other possibility is to use unbiased prior in std_err estimate.
            min_non_zero = min(non_zero)
            non_zero_err = np.asarray([v if v > 0 else min_non_zero for v in err])

            weights = 1 / non_zero_err
    else:
        probability_one, _ = transform_pauli_moments_to_bit(np.asarray(-1 * y_expectations), 0)
        weights = None

    return fit_shifted_cosine(np.asarray(angles), probability_one, weights, param_guesses)
