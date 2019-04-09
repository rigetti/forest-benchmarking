from typing import Union, List, Tuple, Sequence

import numpy as np
from numpy import pi
from scipy import optimize
from matplotlib import pyplot as plt

from pyquil.api import QuantumComputer
from pyquil.gates import RX, RZ, CZ, MEASURE
from pyquil.quil import Program
from pyquil.quilbase import Pragma
from pyquil.paulis import PauliTerm
from pyquil.operator_estimation import ExperimentSetting, zeros_state

from forest.benchmarking.utils import transform_pauli_moments_to_bit
from forest.benchmarking.stratified_experiment import StratifiedExperiment, Layer, \
    acquire_stratified_data

MILLISECOND = 1e-6  # A millisecond (ms) is an SI unit of time
MICROSECOND = 1e-6  # A microsecond (us) is an SI unit of time
NANOSECOND = 1e-9  # A nanosecond (ns) is an SI unit of time
USEC_PER_DEPTH = .1

# A Hertz (Hz) is a derived unit of frequency in SI Units; 1 Hz is defined as one cycle per second.
KHZ = 1e3  # kHz
MHZ = 1e6  # MHz
GHZ = 1e9  # GHz


# ==================================================================================================
#   T1
# ==================================================================================================


def generate_t1_experiment(qubit: int, times: Sequence[float]) -> StratifiedExperiment:
    """
    Return a StratifiedExperiment containing programs which constitute a t1 experiment to
    measure the decay time from the excited state to ground state.

    :param qubit: Which qubit to measure.
    :param times: The times at which to measure, given in seconds.
    :return: A dataframe with columns: time, t1 program
    """
    layers = []
    for t in times:
        t = round(t, 7)  # enforce 100ns boundaries
        sequence = ([Program(RX(pi, qubit)), Program(Pragma('DELAY', [qubit], str(t)))])
        settings = (ExperimentSetting(zeros_state([qubit]), PauliTerm('Z', qubit)), )
        t_in_us = round(t/MICROSECOND, 1)

        # the depth is time in units of [100ns]
        layers.append(Layer(int(round(t_in_us / USEC_PER_DEPTH)), sequence, settings, (qubit,),
                            "T"+str(t_in_us)+"us", continuous_param = t))

    return StratifiedExperiment(tuple(layers), (qubit,), "T1")


def acquire_t1_data(qc: QuantumComputer, experiments: Sequence[StratifiedExperiment], num_shots):
    """
    Execute experiments to measure the T1 decay time of 1 or more qubits.

    :param qc: The QuantumComputer to run the experiment on
    :param experiments:
    :param num_shots
    :return:
    """
    if not isinstance(experiments, Sequence):
        experiments = [experiments]
    acquire_stratified_data(qc, experiments, num_shots)
    for expt in experiments:
        for layer in expt.layers:
            z_expectation = layer.results[0].expectation
            var = layer.results[0].stddev**2
            prob0, bit_var = transform_pauli_moments_to_bit(z_expectation, var)
            # TODO: allow addition to estimates or always over-write?
            layer.estimates = {"Fraction One": (1- prob0, np.sqrt(bit_var))}


def estimate_t1(experiment: StratifiedExperiment):
    """
    Estimate T1 from experimental data.

    :param experiment:
    :return:
    """
    x_data = [layer.depth * USEC_PER_DEPTH for layer in experiment.layers]  # times in u-seconds
    y_data = [layer.estimates["Fraction One"][0] for layer in experiment.layers]

    fit_params, fit_params_errs = fit_to_exponential_decay_curve(np.array(x_data), np.array(y_data))
    #TODO: check if estimates exists?
    experiment.estimates = {"T1": (fit_params[1], fit_params_errs[1])}
    return fit_params, fit_params_errs


def plot_t1_estimate_over_data(experiments: Union[StratifiedExperiment,
                                                  Sequence[StratifiedExperiment]],
                               expts_fit_params,
                               expts_fit_params_errs, # TODO: plot err bars, make like rb
                               filename: str = None) -> None:
    """
    Plot T1 experimental data and estimated value of T1 as an exponential decay curve.

    :param experiments: A list of experiments with T1 data.
    :param filename: if provided, the file where the plot is saved
    :return: None
    """
    if isinstance(experiments, StratifiedExperiment):
        experiments = [experiments]
    if isinstance(expts_fit_params[0], float):
        expts_fit_params = [expts_fit_params]
        expts_fit_params_errs = [expts_fit_params_errs]

    for expt, fit_params, fit_params_errs in zip(experiments, expts_fit_params,
                                                 expts_fit_params_errs):
        q = expt.qubits[0]

        times = [layer.depth * USEC_PER_DEPTH for layer in expt.layers]  # times in u-seconds
        one_survival = [layer.estimates["Fraction One"][0] for layer in expt.layers]

        plt.plot(times, one_survival, 'o-', label=f"q{q} T1 data")
        plt.plot(times, exponential_decay_curve(np.array(times), *fit_params),
                 label=f"q{q} fit: T1={fit_params[1]:.2f}us")

    plt.xlabel("Time [us]")
    plt.ylabel(r"Pr($|1\rangle$)")
    plt.title("$T_1$ decay")

    plt.legend(loc='best')
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


# ==================================================================================================
#   T2 star and T2 echo functions
# ==================================================================================================
def generate_t2_star_experiment(qubit: int, times: Sequence[float], detuning: float = 5e6) \
        -> StratifiedExperiment:
    """
    Return a StratifiedExperiment containing programs which ran in sequence constitute a T2 star
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
        sequence = ([Program(RX(pi / 2, qubit)),  # prep
                     Program(Pragma('DELAY', [qubit], str(t))) # delay and measure
                     + RZ(2 * pi * t * detuning, qubit) + RX(pi / 2, qubit)])
        settings = (ExperimentSetting(zeros_state([qubit]), PauliTerm('Z', qubit)),)
        t_in_us = round(t / MICROSECOND, 1)

        # the depth is time in units of [100ns]
        layers.append(Layer(int(round(t_in_us / USEC_PER_DEPTH)), sequence, settings, (qubit,),
                            "T" + str(t_in_us) + "us", continuous_param=t))

    return StratifiedExperiment(tuple(layers), (qubit,), "T2star", meta_data={'Detuning': detuning})


def generate_t2_echo_experiment(qubit: int, times: Sequence[float], detuning: float = 5e6) \
        -> StratifiedExperiment:
    """
    Return a StratifiedExperiment containing programs which ran in sequence constitute a T2 star
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

    return StratifiedExperiment(tuple(layers), (qubit,), "T2echo", meta_data={'Detuning': detuning})


def acquire_t2_data(qc: QuantumComputer, experiments: Sequence[StratifiedExperiment], num_shots):
    """
    Execute experiments to measure the T2 time of one or more qubits.

    :param qc: The QuantumComputer to run the experiment on
    :param experiments:
    :param num_shots
    :return:
    """
    if not isinstance(experiments, Sequence):
        experiments = [experiments]
    acquire_stratified_data(qc, experiments, num_shots)
    for expt in experiments:
        for layer in expt.layers:
            z_expectation = layer.results[0].expectation
            var = layer.results[0].stddev**2
            prob0, bit_var = transform_pauli_moments_to_bit(z_expectation, var)
            # TODO: allow addition to estimates or always over-write?
            layer.estimates = {"Fraction One": (1- prob0, np.sqrt(bit_var))}


def estimate_t2(experiment: StratifiedExperiment):
    """
    Estimate T2 star or T2 echo from experimental data.

    :param experiment:
    :return:
    """
    x_data = [layer.depth * USEC_PER_DEPTH for layer in experiment.layers]  # times in u-seconds
    y_data = [layer.estimates["Fraction One"][0] for layer in experiment.layers]
    detuning = experiment.meta_data['Detuning']

    fit_params, fit_params_errs = fit_to_exponentially_decaying_sinusoidal_curve(np.array(x_data),
                                                                                 np.array(y_data),
                                                                                 detuning)
    # TODO: check and perhaps change untis. Was done in seconds before...
    experiment.estimates = {"T2": fit_params[1], "Freq": fit_params[2]}
    return fit_params, fit_params_errs


def plot_t2_estimate_over_data(experiments: Union[StratifiedExperiment,
                                                  Sequence[StratifiedExperiment]],
                               expts_fit_params,
                               expts_fit_params_errs, # TODO: plot err bars, make like rb
                               filename: str = None) -> None:
    """
    Plot T1 experimental data and estimated value of T1 as an exponential decay curve.

    :param experiments: A list of experiments with T1 data.
    :param filename: if provided, the file where the plot is saved
    :return: None
    """
    if isinstance(experiments, StratifiedExperiment):
        experiments = [experiments]
    if isinstance(expts_fit_params[0], float):
        expts_fit_params = [expts_fit_params]
        expts_fit_params_errs = [expts_fit_params_errs]

    for expt, fit_params, fit_params_errs in zip(experiments, expts_fit_params,
                                                 expts_fit_params_errs):
        q = expt.qubits[0]

        times = [layer.depth * USEC_PER_DEPTH for layer in expt.layers]  # times in u-seconds
        one_survival = [layer.estimates["Fraction One"][0] for layer in expt.layers]

        plt.plot(times, one_survival, 'o-', label=f"q{q} T2 data")
        plt.plot(times, exponentially_decaying_sinusoidal_curve(np.array(times), *fit_params),
                 label=f"q{q} fit: freq={fit_params[2] / MHZ:.2f}MHz, "
                       f""f"T2={fit_params[1] / MICROSECOND:.2f}us")

    plt.xlabel("Time [us]")
    plt.ylabel(r"Pr($|1\rangle$)")
    expt_types = [expt.expt_type for expt in experiments]
    if 'T2star' in expt_types and 'T2echo' in expt_types:
        plt.title("$T_2$ (mixed type) decay")
    elif 'T2echo' in expt_types:
        plt.title("$T_2^*$ (Ramsey) decay")
    elif 'T2echo' in expt_types:
        plt.title("$T_2$ (Echo) decay")
    else:
        plt.title("Unknown Type decay")

    plt.legend(loc='best')
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


# ==================================================================================================
#   TODO CPMG
# ==================================================================================================


# ==================================================================================================
#   Rabi
# ==================================================================================================
def generate_rabi_experiment(qubit: int, angles: Sequence[float]) -> StratifiedExperiment:
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

    return StratifiedExperiment(tuple(layers), (qubit,), "Rabi")


def acquire_rabi_data(qc: QuantumComputer, experiments: Sequence[StratifiedExperiment], num_shots):
    """
    Execute Rabi experiments.

    :param qc: The QuantumComputer to run the experiment on
    :param experiments:
    :param num_shots
    :return:
    """
    if not isinstance(experiments, Sequence):
        experiments = [experiments]
    acquire_stratified_data(qc, experiments, num_shots)
    for expt in experiments:
        for layer in expt.layers:
            z_expectation = layer.results[0].expectation
            var = layer.results[0].stddev**2
            prob0, bit_var = transform_pauli_moments_to_bit(z_expectation, var)
            # TODO: allow addition to estimates or always over-write?
            layer.estimates = {"Fraction One": (1- prob0, np.sqrt(bit_var))}


def estimate_rabi(experiment: StratifiedExperiment):
    """
    Estimate Rabi oscillation from experimental data.

    :param experiment: Rabi experiment with results
    :return:
    """
    x_data = [layer.continuous_param for layer in experiment.layers] # angles
    y_data = [layer.estimates["Fraction One"][0] for layer in experiment.layers]

    # fit to sinusoid
    fit_params, fit_params_errs = fit_to_shifted_cosine(np.array(x_data), np.array(y_data))
    #TODO: check if estimates exists?
    param_labels = ["p(1|1)", "p(1|0)", "f_ideal/f_control", "Phase"]
    experiment.estimates = {label: (param, err) for label, param, err in zip(param_labels,
                                                                           fit_params,
                                                                           fit_params_errs)}

    return fit_params, fit_params_errs


def plot_rabi_estimate_over_data(experiments: Union[StratifiedExperiment,
                                                  Sequence[StratifiedExperiment]],
                                 expts_fit_params,
                                 expts_fit_params_errs, # TODO: plot err bars, make like rb
                                 filename: str = None) -> None:
    """
    Plot Rabi oscillation experimental data and estimated curve.

    :param experiments: A list of experiments with rabi data.
    :param filename: if provided, the file where the plot is saved
    :return: None
    """
    if isinstance(experiments, StratifiedExperiment):
        experiments = [experiments]
    if isinstance(expts_fit_params[0], float):
        expts_fit_params = [expts_fit_params]
        expts_fit_params_errs = [expts_fit_params_errs]

    for expt, fit_params, fit_params_errs in zip(experiments, expts_fit_params,
                                                 expts_fit_params_errs):
        q = expt.qubits[0]

        angles = [layer.continuous_param for layer in expt.layers]
        one_survival = [layer.estimates["Fraction One"][0] for layer in expt.layers]

        plt.plot(angles, one_survival, 'o-', label=f"qubit {q} Rabi data")
        plt.plot(angles, shifted_cosine(np.array(angles), *fit_params),
                 label=f"qubit {q} fitted line")

    plt.xlabel("RX angle [rad]")
    plt.ylabel(r"Pr($|1\rangle$)")
    plt.title("Rabi flop")
    plt.legend(loc='best')
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


# ==================================================================================================
#   CZ phase Ramsey
# ==================================================================================================
def generate_cz_phase_ramsey_experiment(cz_qubits: Sequence[int], measure_qubit: int,
                                        angles: Sequence[float]) -> StratifiedExperiment:
    """
    Return a StratifiedExperiment containing programs programs that constitute a CZ phase ramsey
    experiment.

    :param cz_qubits: the qubits participating in the cz gate
    :param measure_qubit: Which qubit to measure.
    :param angles: A list of angles at which to make a measurement
    :return:
    """
    qubits = tuple(set(cz_qubits).union([measure_qubit]))
    layers = []
    for angle in angles:
        send_to_equator = Program(RX(pi/2, measure_qubit))
        apply_phase = Program(RZ(angle, measure_qubit))
        sequence = ([send_to_equator + CZ(*cz_qubits) + apply_phase + send_to_equator.dagger()])
        settings = (ExperimentSetting(zeros_state([measure_qubit]), PauliTerm('Z', measure_qubit)),)

        layers.append(Layer(1, sequence, settings, qubits, f"{round(angle,2)}rad",
                            continuous_param=angle))

    return StratifiedExperiment(tuple(layers), qubits, "CZ Ramsey")


def acquire_cz_phase_ramsey_data(qc: QuantumComputer, experiments: Sequence[StratifiedExperiment],
                                 num_shots: int):
    """
    Execute experiments to measure the RZ incurred as a result of a CZ gate.

    :param qc: The QuantumComputer to run the experiment on
    :param experiments:
    :param num_shots
    :return:
    """
    if isinstance(experiments, StratifiedExperiment):
        experiments = [experiments]
    acquire_stratified_data(qc, experiments, num_shots)
    for expt in experiments:
        for layer in expt.layers:
            z_expectation = layer.results[0].expectation
            var = layer.results[0].stddev**2
            prob0, bit_var = transform_pauli_moments_to_bit(z_expectation, var)
            # TODO: allow addition to estimates or always over-write?
            layer.estimates = {"Fraction One": (1- prob0, np.sqrt(bit_var))}


def estimate_cz_phase_ramsey(experiment: StratifiedExperiment):
    """
    Estimate CZ phase ramsey experimental data.

    :param experiment: CZ phase ramsey experiment with results
    :return:
    """
    x_data = [layer.continuous_param for layer in experiment.layers] # angles
    y_data = [layer.estimates["Fraction One"][0] for layer in experiment.layers]

    # fit to (1-cos(angle))/2
    fit_params, fit_params_errs = fit_to_shifted_cosine(np.array(x_data), np.array(y_data))

    #TODO: check if estimates exists?
    # TODO: check that these parameters are correct after re-factoring model
    param_labels = ["p(1|1)", "p(1|0)", "f_ideal/f_control", "Phase"]
    experiment.estimates = {label: (param, err) for label, param, err in zip(param_labels,
                                                                           fit_params,
                                                                           fit_params_errs)}

    return fit_params, fit_params_errs


def plot_cz_ramsey_estimate_over_data(experiment: StratifiedExperiment, fit_params,
                                      fit_params_errs, # TODO: plot err bars, make like rb
                                      filename: str = None) -> None:
    """
    Plot CZ phase ramsey oscillation experimental data and estimated curve.

    :param experiment: A list of experiments with cz phase ramsey data.
    :param filename: if provided, the file where the plot is saved
    :return: None
    """
    # TODO: store measure qubits in layer?
    q = experiment.layers[0].settings[0].in_state[0].qubit
    cz_qubits = [qubit for qubit in experiment.qubits if qubit != q]
    if len(cz_qubits) < 2:
        cz_qubits.append(q)

    angles = [layer.continuous_param for layer in experiment.layers]
    one_survival = [layer.estimates["Fraction One"][0] for layer in experiment.layers]

    plt.plot(angles, one_survival, 'o-', label=f"qubit{q} CZ Ramsey data")
    plt.plot(angles, shifted_cosine(np.array(angles), *fit_params),
             label=f"qubit {q} fitted line")

    estimated_phase, phase_err = experiment.estimates["Phase"]
    # TODO: is it important to plot the line at the peak?
    plt.axvline(pi - estimated_phase,
                label=f"pi - q{q} imparted phase={pi - estimated_phase:.3f}+/-{phase_err:.3f} rad")

    # TODO: support plotting of multiple experiments
    # if len(edges) == 1:
    #     # this deals with the one edge case, then plot will have an empty row
    #     # if you don't do this you get `axes.shape = (2,)`
    #     fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24, 30))
    # else:
    #     fig, axes = plt.subplots(nrows=len(edges), ncols=2, figsize=(24, 10 * len(edges)))
    #
    # for id_row, edge in enumerate(edges):
    #     for id_col, qubit in enumerate(edge):
    #
    #         if row['Fit_params'].values[0] is None:
    #             print(f"Rabi estimate did not succeed for qubit {q}")
    #         else:
    #             fit_params = row['Fit_params'].values[0]
    #             max_ESV = row['max_ESV'].values[0]
    #             max_ESV_err = row['max_ESV_err'].values[0]

    plt.xlabel("RZ phase [rad]")
    plt.ylabel(r"Pr($|1\rangle$)")
    plt.title(f"CZ Phase Ramsey fringes on q{q} from CZ({cz_qubits[0]},{cz_qubits[1]})")
    plt.legend(loc='best')
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


# ==================================================================================================
#   Fits and so forth
# ==================================================================================================
def exponential_decay_curve(t: Union[float, np.ndarray],
                            amplitude: float,
                            time_decay_constant: float,
                            t_offset: float = 0.0) -> Union[float, np.ndarray]:
    """
    Calculate exponential decay at a series of points.

    :param t: The independent variable with respect to which decay is calculated.
    :param amplitude: The amplitude of the decay curve.
    :param time_decay_constant: The time decay constant - in this case T1 - of the decay curve.
    :param t_offset: The time offset of the curve, assumed to be 0.0.
    :return: The exponential decay at the point(s) in time.
    """
    return amplitude * np.exp(-1 * (t - t_offset) / time_decay_constant)


def fit_to_exponential_decay_curve(x_data: np.ndarray,
                                   y_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit experimental data to exponential decay curve.

    :param x_data: Independent data to fit to.
    :param y_data: Experimental, dependent data to fit to.
    :return: Arrays of fitted decay curve parameters and their errors
    """
    params, params_covariance = optimize.curve_fit(exponential_decay_curve,
                                                   x_data, y_data,
                                                   p0=[1.0, 15e-6, 0.0])

    # parameter error extraction from
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    params_errs = np.sqrt(np.diag(params_covariance))

    return params, params_errs


def shifted_cosine(control_angle: float, p1_given_1: float, p1_given_0: float,
                   f_ideal_over_f_control: float, f_ideal_phase: float) -> np.ufunc:
    """
    Calculate sinusoidal response at a series of points.

    :param control_angle: The independent variable; this is the angle that we specify in our
        gates. If our gates are incorrectly calibrated then a given control angle will result in
        a different angle with respect to the ideal qubit frequency by the factor
        f_ideal_over_f_control.
    :param p1_given_1: The probability of measuring 1 when the qubit is in the |1> state.
    :param p1_given_0: The probability of measuring 1 when the qubit is in the |0> state.
    :param f_ideal_over_f_control: The ratio of the true qubit frequency over the control
        frequency determined by calibration. e.g. If our gates are incorrectly calibrated to
        apply an over-rotation then f_ideal_over_f_control will be greater than 1; the control
        frequency will be smaller than the true frequency so we interpret a given desired angle to
        require more time, and that control time (multiplied by the larger true frequency) results
        in a larger angle than the intended control angle.
    :param f_ideal_phase: The offset phase, in radians, with respect to the ideal qubit frequency.
        e.g. in a cz Ramsey experiment say that our RZ gate is perfectly calibrated and the cz
        gate imparts an effective RZ(pi/5) rotation to the qubit; in this case f_ideal_phase is
        pi/5 and this phase could be corrected by applying the gate RZ(-pi/5) after cz. If our RZ
        gate was instead found to be mis-calibrated, note that the reported f_ideal_phase would
        remain the same but a correction using our mis-calibrated gates would require a control
        angle of RZ(-f_ideal_phase / f_ideal_over_f_control)
    :return: The sinusoidal response at the given phases(s).
    """
    amplitude = (p1_given_1 - p1_given_0) / 2
    baseline = amplitude + p1_given_0
    return -1 * amplitude * np.cos(f_ideal_over_f_control * control_angle + f_ideal_phase) + \
           baseline


def fit_to_shifted_cosine(x_data: np.ndarray,
                               y_data: List[float],
                               displayflag: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit experimental data to sinusoid.

    :param x_data: Independent data to fit to.
    :param y_data: Experimental, dependent data to fit to.
    :param displayflag: If True displays results from scipy curve fit analysis.
    :return: Arrays of fitted decay curve parameters and their standard deviations
    """
    params, params_covariance = optimize.curve_fit(shifted_cosine, x_data, y_data,
                                                   p0=[0.5, 0.5, 1.0, 0.])
    # parameter error extraction from
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    params_errs = np.sqrt(np.diag(params_covariance))

    # interleave params and params_errs
    print_params = []
    for idx in range(len(params)):
        print_params.append(params[idx])
        print_params.append(params_errs[idx])

    if displayflag:
        print("scipy curve fitting analysis returned\n"
              "p(1|1):\t{:.5f} +/- {:.5f}\n"
              "p(1|0):\t{:.5f} +/- {:.5f}\n"
              "f_ideal/f_control:\t{:.5f} +/- {:.5f}\n"
              "Phase:\t{:.5f} +/- {:.5f}".format(*print_params))

    return params, params_errs


def exponentially_decaying_sinusoidal_curve(t: Union[float, np.ndarray],
                                            amplitude: float,
                                            time_decay_constant: float,
                                            frequency: float,
                                            baseline: float,
                                            sin_t_offset: float = 0.0) -> Union[float, np.ndarray]:
    """
    Calculate exponentially decaying sinusoid at a series of points.

    :param t: The independent variable with respect to which decay is calculated.
    :param amplitude: The amplitude of the decay curve.
    :param time_decay_constant: The time decay constant - in this case T2 - of the decay curve.
    :param frequency: The frequency to fit to the Ramsey fringes.
    :param baseline: The baseline of the Ramsey fringes.
    :param sin_t_offset: The time offset of the sinusoidal curve, assumed to be 0.0.
    :return: The exponentially decaying sinusoid evaluated at the point(s) in time.
    """
    return amplitude * np.exp(-1 * t / time_decay_constant) * \
           np.sin(frequency * (t - sin_t_offset)) + baseline


def fit_to_exponentially_decaying_sinusoidal_curve(x_data: np.ndarray,
                                                   y_data: np.ndarray,
                                                   detuning: float = 5e6) -> Tuple[np.ndarray,
                                                                                   np.ndarray]:
    """
    Fit experimental data to exponential decay curve.

    :param x_data: Independent data to fit to.
    :param y_data: Experimental, dependent data to fit to.
    :param detuning: Detuning frequency used in experiment creation.
    :return: Arrays of fitted decay curve parameters and their errors
    """
    params, params_covariance = optimize.curve_fit(exponentially_decaying_sinusoidal_curve,
                                                   x_data, y_data,
                                                   p0=[0.5, 15e-6, detuning, 0.5, 0.0])

    # parameter error extraction from
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    params_errs = np.sqrt(np.diag(params_covariance))

    return params, params_errs
