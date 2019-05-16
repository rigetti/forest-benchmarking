from typing import Sequence, List, Dict

import numpy as np
from numpy import pi
from lmfit.model import ModelResult
from matplotlib import pyplot as plt

from pyquil.api import QuantumComputer
from pyquil.gates import RX, RZ, CZ, MEASURE
from pyquil.quil import Program
from pyquil.quilbase import Pragma
from pyquil.paulis import PauliTerm

from forest.benchmarking.utils import transform_pauli_moments_to_bit
from forest.benchmarking.analysis.fitting import fit_decay_constant_param_decay, \
    fit_decaying_sinusoid, fit_shifted_cosine
from forest.benchmarking.operator_estimation import ObservablesExperiment, ExperimentResult, \
    ExperimentSetting, estimate_observables, zeros_state, minusZ, plusX


MILLISECOND = 1e-6  # A millisecond (ms) is an SI unit of time
MICROSECOND = 1e-6  # A microsecond (us) is an SI unit of time
NANOSECOND = 1e-9  # A nanosecond (ns) is an SI unit of time
USEC_PER_DEPTH = .1

# A Hertz (Hz) is a derived unit of frequency in SI Units; 1 Hz is defined as one cycle per second.
KHZ = 1e3  # kHz
MHZ = 1e6  # MHz
GHZ = 1e9  # GHz


def get_stats_by_qubit(expt_results: List[List[ExperimentResult]]):
    """
    Organize the mean and std_err of an experiment by qubit.

    :param expt_results:
    :return:
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


def generate_t1_experiment(qubits: Sequence[int], times: Sequence[float],
                           make_simultaneous = False) -> List[ObservablesExperiment]:
    """
    Return a ObservablesExperiment containing programs which constitute a t1 experiment to
    measure the decay time from the excited state to ground state.

    :param qubits: list of qubits to measure.
    :param times: The times at which to measure, given in seconds.
    :param make_simultaneous: generate len(times) simultaneous experiments, each consisting of a
        program which measures all qubits.
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

        if make_simultaneous:
            settings = [settings]

        expts.append(ObservablesExperiment(settings, program))

    return expts


def acquire_t1_data(qc: QuantumComputer, experiments: Sequence[ObservablesExperiment], num_shots) \
        -> List[List[ExperimentResult]]:
    """
    Acquire data to measure the T1 decay time for each of the input experiments.

    :param qc: The QuantumComputer to run the experiment on
    :param experiments:
    :param num_shots
    :return:
    """
    results = []
    for expt in experiments:
        results.append(list(estimate_observables(qc, expt, num_shots)))
    return results


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
    if z_std_errs is not None:
        probability_one, var = transform_pauli_moments_to_bit(np.asarray(z_expectations),
                                                              np.asarray(z_std_errs)**2)
        err = np.sqrt(var)
        min_non_zero = min([v for v in err if v > 0])
        # TODO: does this handle 0 var appropriately? Incorporate unbiased prior into std_err estimate?
        non_zero_err = np.asarray([v if v > 0 else min_non_zero for v in err])

        weights = 1 / non_zero_err
    else:
        probability_one = (np.asarray(z_expectations) + 1) / 2
        weights = None

    return fit_decay_constant_param_decay(np.asarray(times), probability_one, weights,
                                          param_guesses)

# TODO: remove this
# def plot_t1_estimate_over_data(experiments: Union[ObservablesExperiment,
#                                                   Sequence[ObservablesExperiment]],
#                                expts_fit_params,
#                                expts_fit_params_errs, # TODO: plot err bars, make like rb
#                                filename: str = None) -> None:
#     """
#     Plot T1 experimental data and estimated value of T1 as an exponential decay curve.
#
#     :param experiments: A list of experiments with T1 data.
#     :param filename: if provided, the file where the plot is saved
#     :return: None
#     """
#     if isinstance(experiments, ObservablesExperiment):
#         experiments = [experiments]
#     if isinstance(expts_fit_params[0], float):
#         expts_fit_params = [expts_fit_params]
#         expts_fit_params_errs = [expts_fit_params_errs]
#
#     for expt, fit_params, fit_params_errs in zip(experiments, expts_fit_params,
#                                                  expts_fit_params_errs):
#         q = expt.qubits[0]
#
#         times = [layer.depth * USEC_PER_DEPTH for layer in expt.layers]  # times in u-seconds
#         one_survival = [layer.estimates["Fraction One"][0] for layer in expt.layers]
#
#         plt.plot(times, one_survival, 'o-', label=f"q{q} T1 data")
#         plt.plot(times, exponential_decay_curve(np.array(times), *fit_params),
#                  label=f"q{q} fit: T1={fit_params[1]:.2f}us")
#
#     plt.xlabel("Time [us]")
#     plt.ylabel(r"Pr($|1\rangle$)")
#     plt.title("$T_1$ decay")
#
#     plt.legend(loc='best')
#     plt.tight_layout()
#     if filename is not None:
#         plt.savefig(filename)
#     plt.show()


# ==================================================================================================
#   T2 star and T2 echo functions
# ==================================================================================================
def generate_t2_star_experiment(qubit: int, times: Sequence[float], detuning: float = 5e6) \
        -> ObservablesExperiment:
    """
    Return a ObservablesExperiment containing programs which ran in sequence constitute a T2 star
    experiment to measure the T2 star coherence decay time.

    :param qubit: Which qubit to measure.
    :param times: The times at which to measure.
    :param detuning: The additional detuning frequency about the z axis.
    :return:
    """
    layers = []
    for t in times:
        # TODO: avoid aliasing while being mindful of the 20ns resolution in the QCS stack
        t = round(t, 7)  # enforce 100ns boundaries
        # delay and measure
        sequence = (Program(Pragma('DELAY', [qubit], str(t)))
                            + RZ(2 * pi * t * detuning, qubit) + RX(pi / 2, qubit), )
        settings = (ExperimentSetting(plusX(qubit), PauliTerm('Z', qubit)),)
        t_in_us = round(t / MICROSECOND, 1)

        # the depth is time in units of [100ns]
        layers.append(Layer(int(round(t_in_us / USEC_PER_DEPTH)), sequence, settings, (qubit,),
                            "T" + str(t_in_us) + "us", continuous_param=t))

    return ObservablesExperiment(tuple(layers), (qubit,), "T2star", metadata={'Detuning': detuning})


def generate_t2_echo_experiment(qubit: int, times: Sequence[float], detuning: float = 5e6) \
        -> ObservablesExperiment:
    """
    Return a ObservablesExperiment containing programs which ran in sequence constitute a T2 star
    experiment to measure the T2 star coherence decay time.

    :param qubit: Which qubit to measure.
    :param times: The times at which to measure.
    :param detuning: The additional detuning frequency about the z axis, specified in Hz
    :return:
    """
    layers = []
    for t in times:
        # TODO: avoid aliasing while being mindful of the 20ns resolution in the QCS stack
        t = round(t, 7)  # enforce 100ns boundaries

        half_delay = Pragma('DELAY', [qubit], str(t/2))
        echo_prog = Program([half_delay, RX(pi / 2, qubit), RX(pi / 2, qubit), half_delay])
        sequence = ([Program(RX(pi / 2, qubit)), # prep
                     # delay/echo/delay and measure
                     echo_prog + RZ(2 * pi * t * detuning, qubit) + RX(pi / 2, qubit)])

        settings = (ExperimentSetting(zeros_state([qubit]), PauliTerm('Z', qubit)),)
        t_in_us = round(t / MICROSECOND, 1)

        # the depth is time in units of [100ns]
        layers.append(Layer(int(round(t_in_us / USEC_PER_DEPTH)), sequence, settings, (qubit,),
                            "T" + str(t_in_us) + "us", continuous_param=t_in_us))

    return ObservablesExperiment(tuple(layers), (qubit,), "T2echo", metadata={'Detuning': detuning})


def acquire_t2_data(qc: QuantumComputer, experiments: Sequence[ObservablesExperiment], num_shots) \
        -> Sequence[ObservablesExperiment]:
    """
    Execute experiments to measure the T2 time of one or more qubits.

    :param qc: The QuantumComputer to run the experiment on
    :param experiments:
    :param num_shots
    :return:
    """
    return acquire_qubit_spectroscopy_data(qc, experiments, num_shots)


def fit_t2_results(experiment: ObservablesExperiment, param_guesses: tuple = None) \
        -> ModelResult:
    """
    Wrapper for fitting the results of a ObservablesExperiment; simply extracts key parameters
    and passes on to the standard fit.

    The estimate for T2 can be found in the returned fit.params['decay_constant']

    :param experiment: the T2star or T2echo ObservablesExperiment with results on which to fit a
        T2 decay.
    :param param_guesses: guesses for the (amplitude, decay_constant, offset, baseline,
        frequency) parameters where decay_constant is in microseconds and frequency is in MHZ
    :return: a ModelResult fit with estimates of the Model parameters, including the T2
        'decay_constant'
    """
    x_data = []
    y_data = []
    weights = []
    for layer in experiment.layers:
        x_data.append(layer.depth * USEC_PER_DEPTH) # times in u-seconds
        y_data.append(layer.estimates["Fraction One"][0])
        err = layer.estimates["Fraction One"][1]
        # TODO: improve handling of inf weights
        if err == 0:
            weight = 100
        else:
           weight = 1/layer.estimates["Fraction One"][1]
        weights.append(weight)

    detuning = experiment.metadata['Detuning']

    if param_guesses is None:  # make some standard reasonable guess
        param_guesses = (.5, 10, 0.5, 0., detuning / MHZ)

    return fit_decaying_sinusoid(np.asarray(x_data), np.asarray(y_data), param_guesses,
                                 np.asarray(weights))


# TODO: remove
# def plot_t2_estimate_over_data(experiments: Union[ObservablesExperiment,
#                                                   Sequence[ObservablesExperiment]],
#                                expts_fit_params,
#                                expts_fit_params_errs, # TODO: plot err bars, make like rb
#                                filename: str = None) -> None:
#     """
#     Plot T1 experimental data and estimated value of T1 as an exponential decay curve.
#
#     :param experiments: A list of experiments with T1 data.
#     :param filename: if provided, the file where the plot is saved
#     :return: None
#     """
#     if isinstance(experiments, ObservablesExperiment):
#         experiments = [experiments]
#     if isinstance(expts_fit_params[0], float):
#         expts_fit_params = [expts_fit_params]
#         expts_fit_params_errs = [expts_fit_params_errs]
#
#     for expt, fit_params, fit_params_errs in zip(experiments, expts_fit_params,
#                                                  expts_fit_params_errs):
#         q = expt.qubits[0]
#
#         times = [layer.depth * USEC_PER_DEPTH for layer in expt.layers]  # times in u-seconds
#         one_survival = [layer.estimates["Fraction One"][0] for layer in expt.layers]
#
#         plt.plot(times, one_survival, 'o-', label=f"q{q} T2 data")
#         plt.plot(times, exponentially_decaying_sinusoidal_curve(np.array(times), *fit_params),
#                  label=f"q{q} fit: freq={fit_params[2] / MHZ:.2f}MHz, "
#                        f""f"T2={fit_params[1] / MICROSECOND:.2f}us")
#
#     plt.xlabel("Time [us]")
#     plt.ylabel(r"Pr($|1\rangle$)")
#     expt_types = [expt.expt_type for expt in experiments]
#     if 'T2star' in expt_types and 'T2echo' in expt_types:
#         plt.title("$T_2$ (mixed type) decay")
#     elif 'T2star' in expt_types:
#         plt.title("$T_2^*$ (Ramsey) decay")
#     elif 'T2echo' in expt_types:
#         plt.title("$T_2$ (Echo) decay")
#     else:
#         plt.title("Unknown Type decay")
#
#     plt.legend(loc='best')
#     plt.tight_layout()
#     if filename is not None:
#         plt.savefig(filename)
#     plt.show()


# ==================================================================================================
#   TODO CPMG
# ==================================================================================================


# ==================================================================================================
#   Rabi
# ==================================================================================================
def generate_rabi_experiment(qubit: int, angles: Sequence[float]) -> ObservablesExperiment:
    """
    Return a DataFrame containing programs which, when run in sequence, constitute a Rabi
    experiment.

    Rabi oscillations are observed by applying successively larger rotations to the same initial
    state.

    :param qubit: Which qubit to measure.
    :param angles: A list of angles at which to make a measurement
    :return:
    """
    layers = []
    for angle in angles:
        sequence = ([Program(RX(angle, qubit))])
        settings = (ExperimentSetting(zeros_state([qubit]), PauliTerm('Z', qubit)), )

        layers.append(Layer(1, sequence, settings, (qubit,), f"{round(angle,2)}rad",
                            continuous_param=angle))
    # TODO: avoid warnings from basic_compile about RX(not +/-pi)
    return ObservablesExperiment(tuple(layers), (qubit,), "Rabi")


def acquire_rabi_data(qc: QuantumComputer, experiments: Sequence[ObservablesExperiment], num_shots) \
        -> Sequence[ObservablesExperiment]:
    """
    Execute Rabi experiments.

    :param qc: The QuantumComputer to run the experiment on
    :param experiments:
    :param num_shots
    :return:
    """
    return acquire_qubit_spectroscopy_data(qc, experiments, num_shots)


def fit_rabi_results(experiment: ObservablesExperiment, param_guesses: tuple = None) \
        -> ModelResult:
    """
    Wrapper for fitting the results of a ObservablesExperiment; simply extracts key parameters
    and passes on to the standard fit.

    Note the following interpretation of the model fit parameters:
    x: the independent variable is the angle that we specify when writing a gate instruction. If
        our gates are incorrectly calibrated then a given control angle will result in a different
        angle than intended by the multiplicative 'frequency' of the model
    amplitude: this is equivalently (p1_given_1 - p1_given_0) / 2 where
        p1_given_1 is the probability of measuring 1 when the qubit is in the |1> state.
        p1_given_0 is the probability of measuring 1 when the qubit is in the |0> state.
    offset: This is the offset phase, in radians, with respect to the true rotation frequency.
        e.g. if our gate is mis-calibrated resulting in an offset 'off' then we require a control
        angle of RX(-off / frequency) to correct the offset
    baseline: this is the amplitude + p1_given_0
    frequency: The ratio of the actual angle rotated over the intended control angle
        e.g. If our gates are incorrectly calibrated to apply an over-rotation then
        frequency will be greater than 1; the intended control angle will be smaller than the
        true angle rotated.

    :param experiment: the Rabi ObservablesExperiment with results on which to fit a Rabi flop
    :param param_guesses: guesses for the (amplitude, offset, baseline, frequency) parameters
    :return: a ModelResult fit with estimates of the Model parameters, including the frequency
        which gives the ratio of actual angle over intended control angle
    """
    x_data = []
    y_data = []
    weights = []
    for layer in experiment.layers:
        x_data.append(layer.continuous_param) # control angles in radians
        y_data.append(layer.estimates["Fraction One"][0])
        err = layer.estimates["Fraction One"][1]
        # TODO: improve handling of inf weights
        if err == 0:
            weight = 100
        else:
           weight = 1/layer.estimates["Fraction One"][1]
        weights.append(weight)

    if param_guesses is None:  # make some standard reasonable guess
        param_guesses = (.5, 0, .5, 1.)

    return fit_shifted_cosine(np.asarray(x_data), np.asarray(y_data), param_guesses,
                              np.asarray(weights))

# TODO: remove
# def plot_rabi_estimate_over_data(experiments: Union[ObservablesExperiment,
#                                                   Sequence[ObservablesExperiment]],
#                                  expts_fit_params,
#                                  expts_fit_params_errs, # TODO: plot err bars, make like rb
#                                  filename: str = None) -> None:
#     """
#     Plot Rabi oscillation experimental data and estimated curve.
#
#     :param experiments: A list of experiments with rabi data.
#     :param filename: if provided, the file where the plot is saved
#     :return: None
#     """
#     if isinstance(experiments, ObservablesExperiment):
#         experiments = [experiments]
#     if isinstance(expts_fit_params[0], float):
#         expts_fit_params = [expts_fit_params]
#         expts_fit_params_errs = [expts_fit_params_errs]
#
#     for expt, fit_params, fit_params_errs in zip(experiments, expts_fit_params,
#                                                  expts_fit_params_errs):
#         q = expt.qubits[0]
#
#         angles = [layer.continuous_param for layer in expt.layers]
#         one_survival = [layer.estimates["Fraction One"][0] for layer in expt.layers]
#
#         plt.plot(angles, one_survival, 'o-', label=f"qubit {q} Rabi data")
#         plt.plot(angles, shifted_cosine(np.array(angles), *fit_params),
#                  label=f"qubit {q} fitted line")
#
#     plt.xlabel("RX angle [rad]")
#     plt.ylabel(r"Pr($|1\rangle$)")
#     plt.title("Rabi flop")
#     plt.legend(loc='best')
#     plt.tight_layout()
#     if filename is not None:
#         plt.savefig(filename)
#     plt.show()


# ==================================================================================================
#   CZ phase Ramsey
# ==================================================================================================
def generate_cz_phase_ramsey_experiment(cz_qubits: Sequence[int], measure_qubit: int,
                                        angles: Sequence[float]) -> ObservablesExperiment:
    """
    Return a ObservablesExperiment containing programs programs that constitute a CZ phase ramsey
    experiment.

    :param cz_qubits: the qubits participating in the cz gate
    :param measure_qubit: Which qubit to measure.
    :param angles: A list of angles at which to make a measurement
    :return:
    """
    qubits = tuple(set(cz_qubits).union([measure_qubit]))
    layers = []
    for angle in angles:
        # TODO: replace Z expectation with X expectation, remove prep and pre-measure?
        send_to_equator = Program(RX(pi/2, measure_qubit))
        apply_phase = Program(RZ(angle, measure_qubit))
        sequence = ([send_to_equator + CZ(*cz_qubits) + apply_phase + send_to_equator.dagger()])
        settings = (ExperimentSetting(zeros_state([measure_qubit]), PauliTerm('Z', measure_qubit)),)

        layers.append(Layer(1, sequence, settings, qubits, f"{round(angle,2)}rad",
                            continuous_param=angle))

    return ObservablesExperiment(tuple(layers), qubits, "CZ Ramsey")


def acquire_cz_phase_ramsey_data(qc: QuantumComputer, experiments: Sequence[ObservablesExperiment],
                                 num_shots: int):
    """
    Execute experiments to measure the RZ incurred as a result of a CZ gate.

    :param qc: The QuantumComputer to run the experiment on
    :param experiments:
    :param num_shots
    :return:
    """
    return acquire_qubit_spectroscopy_data(qc, experiments, num_shots)


def fit_cz_phase_ramsey_results(experiment: ObservablesExperiment, param_guesses: tuple = None) \
        -> ModelResult:
    """
    Wrapper for fitting the results of a ObservablesExperiment; simply extracts key parameters
    and passes on to the standard fit.

    Note the following interpretation of the model fit:
    x: the independent variable is the angle that we specify when writing a gate instruction. If
        our gates are incorrectly calibrated then a given control angle will result in a different
        angle than intended by the multiplicative 'frequency' of the model
    amplitude: this is equivalently (p1_given_1 - p1_given_0) / 2 where
        p1_given_1 is the probability of measuring 1 when the qubit is in the |1> state.
        p1_given_0 is the probability of measuring 1 when the qubit is in the |0> state.
    offset: This is the offset phase, in radians, with respect to the true rotation frequency.
        e.g. say that our RZ gate is perfectly calibrated and the cz gate imparts an effective
        RZ(pi/5) rotation to the qubit; in this case f_ideal_phase is pi/5, offset is 0,
        and this phase could be corrected by applying the gate RZ(-pi/5, qubit) after cz. If our RZ
        gate was instead found to be mis-calibrated, a correction using our mis-calibrated RZ gate
        would require a control angle of RZ(-frequency * pi/5, qubit)
    baseline: this is the amplitude + p1_given_0
    frequency: The ratio of the actual qubit frequency over the potentially mis-calibrated control
        frequency e.g. If our gates are incorrectly calibrated to apply an over-rotation then
        frequency will be greater than 1; the control frequency will be smaller than the true
        frequency so we interpret a given desired angle to require more time, and that control
        time (multiplied by the larger true frequency) results in a larger angle than the
        intended control angle.

    :param experiment: the CZ Ramsey ObservablesExperiment with results on which to fit a shifted
        cosine.
    :param param_guesses: guesses for the (amplitude, offset, baseline, frequency) parameters
    :return: a ModelResult fit with estimates of the Model parameters, including the frequency
        which gives the ratio of actual qubit frequency over calibrated control frequency
    """
    x_data = []
    y_data = []
    weights = []
    for layer in experiment.layers:
        x_data.append(layer.continuous_param) # control angles in radians
        y_data.append(layer.estimates["Fraction One"][0])
        err = layer.estimates["Fraction One"][1]
        # TODO: improve handling of inf weights
        if err == 0:
            weight = 100
        else:
           weight = 1/layer.estimates["Fraction One"][1]
        weights.append(weight)

    if param_guesses is None:  # make some standard reasonable guess
        param_guesses = (.5, 0, .5, 1.)

    return fit_shifted_cosine(np.asarray(x_data), np.asarray(y_data), param_guesses,
                              np.asarray(weights))

# TODO: remove
# def plot_cz_ramsey_estimate_over_data(experiment: ObservablesExperiment, fit_params,
#                                       fit_params_errs, # TODO: plot err bars, make like rb
#                                       filename: str = None) -> None:
#     """
#     Plot CZ phase ramsey oscillation experimental data and estimated curve.
#
#     :param experiment: A list of experiments with cz phase ramsey data.
#     :param filename: if provided, the file where the plot is saved
#     :return: None
#     """
#     # TODO: store measure qubits in layer?
#     q = experiment.layers[0].settings[0].in_state[0].qubit
#     cz_qubits = [qubit for qubit in experiment.qubits if qubit != q]
#     if len(cz_qubits) < 2:
#         cz_qubits.append(q)
#
#     angles = [layer.continuous_param for layer in experiment.layers]
#     one_survival = [layer.estimates["Fraction One"][0] for layer in experiment.layers]
#
#     plt.plot(angles, one_survival, 'o-', label=f"qubit{q} CZ Ramsey data")
#     plt.plot(angles, shifted_cosine(np.array(angles), *fit_params),
#              label=f"qubit {q} fitted line")
#
#     estimated_phase, phase_err = experiment.estimates["Phase"]
#     # TODO: is it important to plot the line at the peak?
#     plt.axvline(pi - estimated_phase,
#                 label=f"pi - q{q} imparted phase={pi - estimated_phase:.3f}+/-{phase_err:.3f} rad")
#
#     # TODO: support plotting of multiple experiments
#     # if len(edges) == 1:
#     #     # this deals with the one edge case, then plot will have an empty row
#     #     # if you don't do this you get `axes.shape = (2,)`
#     #     fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24, 30))
#     # else:
#     #     fig, axes = plt.subplots(nrows=len(edges), ncols=2, figsize=(24, 10 * len(edges)))
#     #
#     # for id_row, edge in enumerate(edges):
#     #     for id_col, qubit in enumerate(edge):
#     #
#     #         if row['Fit_params'].values[0] is None:
#     #             print(f"Rabi estimate did not succeed for qubit {q}")
#     #         else:
#     #             fit_params = row['Fit_params'].values[0]
#     #             max_ESV = row['max_ESV'].values[0]
#     #             max_ESV_err = row['max_ESV_err'].values[0]
#
#     plt.xlabel("RZ phase [rad]")
#     plt.ylabel(r"Pr($|1\rangle$)")
#     plt.title(f"CZ Phase Ramsey fringes on q{q} from CZ({cz_qubits[0]},{cz_qubits[1]})")
#     plt.legend(loc='best')
#     plt.tight_layout()
#     if filename is not None:
#         plt.savefig(filename)
#     plt.show()
