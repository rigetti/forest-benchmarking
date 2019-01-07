from typing import Union, List, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

from pyquil.api import QuantumComputer
from pyquil.gates import RX, RZ, CZ, MEASURE
from pyquil.quil import Program
from pyquil.quilbase import Pragma


def generate_single_t1_experiment(qubits: Union[int, List[int]],
                                  time: float,
                                  n_shots: int = 1000) -> Program:
    """
    Return a t1 program in native Quil for a single time point.

    :param qubits: Which qubits to measure.
    :param time: The decay time before measurement.
    :param n_shots: The number of shots to average over for the data point.
    :return: A T1 Program.
    """
    program = Program()

    try:
        len(qubits)
    except TypeError:
        qubits = [qubits]

    ro = program.declare('ro', 'BIT', len(qubits))
    for q in qubits:
        program += RX(np.pi, q)
        program += Pragma('DELAY', [q], str(time))
    for i in range(len(qubits)):
        program += MEASURE(qubits[i], ro[i])
    program.wrap_in_numshots_loop(n_shots)
    return program


def generate_t1_experiments(qubits: Union[int, List[int]],
                            stop_time: float,
                            n_shots: int = 1000,
                            num_points: int = 15) -> List[Tuple[float, Program]]:
    """
    Return a list of programs which ran in sequence constitute a t1 experiment to measure the
    decay time from the excited state to ground state.

    :param qubits: Which qubits to measure.
    :param stop_time: The maximum decay time to measure at.
    :param n_shots: The number of shots to average over for each data point.
    :param num_points: The number of points for each t1 curve.
    :return: list of tuples in the form: (time, t1 program with decay of that time)
    """
    start_time = 0
    time_and_programs = []
    for t in np.linspace(start_time, stop_time, num_points):
        t = round(t, 7)  # try to keep time on 100ns boundaries
        time_and_programs.append((t, generate_single_t1_experiment(qubits, t, n_shots)))
    return time_and_programs


def run_t1(qc: QuantumComputer,
           qubits: Union[int, List[int]],
           stop_time: float,
           n_shots: int = 1000,
           num_points: int = 15,
           filename: str = None) -> pd.DataFrame:
    """
    Execute experiments to measure the t1 decay time of 1 or more qubits.

    :param qc: The QuantumComputer to run the experiment on.
    :param qubits: Which qubits to measure.
    :param stop_time: The maximum decay time to measure at.
    :param n_shots: The number of shots to average over for each data point.
    :param num_points: The number of points for each t1 curve.
    :param filename: The name of the file to write JSON-serialized results to.
    """
    results = []
    for t, program in generate_t1_experiments(qubits, stop_time, n_shots, num_points):
        executable = qc.compiler.native_quil_to_executable(program)
        bitstrings = qc.run(executable)

        for i in range(len(qubits)):
            avg = np.mean(bitstrings[:, i])
            results.append({
                'qubit': qubits[i],
                'time': t,
                'n_bitstrings': len(bitstrings),
                'avg': float(avg),
            })

    df = pd.DataFrame(results)
    if filename is not None:
        df.to_json(filename)
    return df


def generate_single_t2_experiment(qubits: Union[int, List[int]],
                                  time: float,
                                  detuning: float,
                                  n_shots: int = 1000) -> Program:
    """
    Return a t2 program in native Quil for a single time point.

    :param qubits: Which qubits to measure.
    :param time: The decay time before measurement.
    :param detuning: The additional detuning frequency about the z axis.
    :param n_shots: The number of shots to average over for the data point.
    :return: A T2 Program.
    """
    program = Program()

    try:
        len(qubits)
    except TypeError:
        qubits = [qubits]

    ro = program.declare('ro', 'BIT', len(qubits))
    for q in qubits:
        program += RX(np.pi / 2, q)
        program += Pragma('DELAY', [q], str(time))
        program += RZ(2 * np.pi * time * detuning, q)
        program += RX(np.pi / 2, q)
    for i in range(len(qubits)):
        program += MEASURE(qubits[i], ro[i])
    program.wrap_in_numshots_loop(n_shots)
    return program


def generate_t2_experiments(qubits: Union[int, List[int]],
                            stop_time: float,
                            detuning: float = 5e6,
                            n_shots: int = 1000,
                            num_points: int = 15) -> List[Tuple[float, Program]]:
    """
    Return a list of programs which ran in sequence constitute a t2 experiment to measure the
    t2 coherence decay time.

    :param qubits: Which qubits to measure.
    :param stop_time: The maximum decay time to measure at.
    :param detuning: The additional detuning frequency about the z axis.
    :param n_shots: The number of shots to average over for each data point.
    :param num_points: The number of points for each t2 curve.
    :return: list of tuples in the form: (time, t2 program with decay of that time)
    """
    start_time = 0
    time_and_programs = []
    for t in np.linspace(start_time, stop_time, num_points):
        # TODO: avoid aliasing while being mindful of the 20ns resolution in the QCS stack
        time_and_programs.append((t, generate_single_t2_experiment(qubits, t, detuning,
                                                                   n_shots=n_shots)))
    return time_and_programs


def run_t2(qc: QuantumComputer,
           qubits: Union[int, List[int]],
           stop_time: float,
           detuning: float = 5e6,
           n_shots: int = 1000,
           num_points: int = 50,
           filename: str = None) -> Tuple[pd.DataFrame, float]:
    """
    Execute experiments to measure the t2 decay time of 1 or more qubits.

    :param qc: The QuantumComputer to run the experiment on.
    :param qubits: Which qubits to measure.
    :param stop_time: The maximum decay time to measure at.
    :param detuning: The additional detuning frequency about the z axis.
    :param n_shots: The number of shots to average over for each data point.
    :param num_points: The number of points for each t2 curve.
    :param filename: The name of the file to write JSON-serialized results to.
    :return: T2 results, detuning used in creating experiments for those results
    """
    results = []
    for t, program in generate_t2_experiments(qubits, stop_time, detuning=detuning,
                                              n_shots=n_shots, num_points=num_points):
        executable = qc.compiler.native_quil_to_executable(program)
        bitstrings = qc.run(executable)

        for i in range(len(qubits)):
            avg = np.mean(bitstrings[:, i])
            results.append({
                'qubit': qubits[i],
                'time': t,
                'n_bitstrings': len(bitstrings),
                'avg': float(avg),
            })

    if filename:
        pd.DataFrame(results).to_json(filename)
    return pd.DataFrame(results), detuning


def generate_single_rabi_experiment(qubits: Union[int, List[int]],
                                    theta: float,
                                    n_shots: int = 1000) -> Program:
    """
    Return a Rabi program in native Quil rotated through the given angle.

    Rabi oscillations are observed by applying successively larger rotations to the same initial
    state.

    :param qubits: Which qubits to measure.
    :param theta: The angle of the Rabi RX rotation.
    :param n_shots: The number of shots to average over for the data point.
    :return: A Program that rotates through a given angle about the X axis.
    """
    program = Program()

    try:
        len(qubits)
    except TypeError:
        qubits = [qubits]

    ro = program.declare('ro', 'BIT', len(qubits))
    for q in qubits:
        program += RX(theta, q)
    for i in range(len(qubits)):
        program += MEASURE(qubits[i], ro[i])
    program.wrap_in_numshots_loop(n_shots)
    return program


def generate_rabi_experiments(qubits: Union[int, List[int]],
                              n_shots: int = 1000,
                              num_points: int = 15) -> List[Tuple[float, Program]]:
    """
    Return a list of programs which, when run in sequence, constitute a Rabi experiment.

    Rabi oscillations are observed by applying successively larger rotations to the same initial
    state.

    :param qubits: Which qubits to measure.
    :param n_shots: The number of shots to average over for each data point.
    :param num_points: The number of points for each Rabi curve.
    :return: list of tuples in the form: (angle, program for Rabi rotation of that angle)
    """
    angle_and_programs = []
    for theta in np.linspace(0.0, 2 * np.pi, num_points):
        angle_and_programs.append((theta, generate_single_rabi_experiment(qubits,
                                                                          theta,
                                                                          n_shots)))
    return angle_and_programs


def run_rabi(qc: QuantumComputer,
             qubits: Union[int, List[int]],
             n_shots: int = 1000,
             num_points: int = 15,
             filename: str = None) -> pd.DataFrame:
    """
    Execute experiments to measure Rabi flop one or more qubits.

    :param qc: The QuantumComputer to run the experiment on.
    :param qubits: Which qubits to measure.
    :param n_shots: The number of shots to average over for each data point.
    :param num_points: The number of points for each Rabi curve.
    :param filename: The name of the file to write JSON-serialized results to.
    :return: DataFrame with Rabi results.
    """
    results = []
    for theta, program in generate_rabi_experiments(qubits, n_shots, num_points):
        executable = qc.compiler.native_quil_to_executable(program)
        bitstrings = qc.run(executable)

        for i in range(len(qubits)):
            avg = np.mean(bitstrings[:, i])
            results.append({
                'qubit': qubits[i],
                'angle': theta,
                'n_bitstrings': len(bitstrings),
                'avg': float(avg),
            })

    if filename:
        pd.DataFrame(results).to_json(filename)
    return pd.DataFrame(results)


def generate_parametric_cz_phase_ramsey_program(qcid: int,
                                                other_qcid: int) -> Program:
    """
    Generate a single CZ phase Ramsey experiment at a given phase.

    :param qcid: The qubit to move around the Bloch sphere and measure the incurred RZ on.
    :param other_qcid: The other qubit that constitutes a two-qubit pair along with `qcid`.
    :param phase: The phase kick to supply after playing the CZ pulse on the equator.
    :param num_shots: The number of shots to average over for the data point.
    :return: A parametric Program for performing a CZ Ramsey experiment.
    """
    program = Program()
    # NOTE: only need readout register for `qcid` not `other_qcid` since `other_qcid` is only
    #       needed to identify which CZ gate we're using
    ro = program.declare('ro', 'BIT', 1)
    theta = program.declare('theta', 'REAL')

    # go to the equator
    program += Program(RX(np.pi / 2, qcid))
    # apply the CZ gate - note that CZ is symmetric, so the order of qubits doesn't matter
    program += Program(CZ(qcid, other_qcid))
    # go to |1> after a phase kick
    program += Program(RZ(theta, qcid), RX(np.pi / 2, qcid))

    program += MEASURE(qcid, ro[0])

    return program


def run_cz_phase_ramsey(qc: QuantumComputer,
                        qubits: Tuple[int, int],
                        start_phase: float = 0.0,
                        stop_phase: float = 2 * np.pi,
                        num_points: int = 15,
                        num_shots: int = 1000,
                        filename: str = None) -> pd.DataFrame:
    """
    Execute experiments to measure the RZ incurred as a result of a CZ pulse.

    :param qc: The qubit to move around the Bloch sphere and measure the incurred RZ on.
    :param qubits: A list of two connected qubits to perform CZ phase Ramsey experiments on.
    :param start_phase: The starting phase for the CZ phase Ramsey experiment.
    :param stop_phase: The stopping phase for the CZ phase Ramsey experiment.
    :param num_points: The number of points to sample at between the starting and stopping phase.
    :param num_shots: The number of shots to average over for each data point.
    :param filename: The name of the file to write JSON-serialized results to.
    :return: The JSON-serialized results from CZ phase Ramsey experiment.
    """
    # TODO: verify that `qubits` is a list of two connected qubits and raise an exception otherwise
    results = []

    for qubit in qubits:
        other_qubit = remove_qubits_from_qubit_list(list(qubits), qubit)

        parametric_ramsey_prog = generate_parametric_cz_phase_ramsey_program(qubit, other_qubit)
        binary = compile_parametric_program(qc, parametric_ramsey_prog, num_shots=num_shots)
        qc.qam.load(binary)

        for theta in np.linspace(start_phase, stop_phase, num_points):
            qc.qam.write_memory(region_name='theta', value=theta)
            qc.qam.run()
            qc.qam.wait()
            bitstrings = qc.qam.read_from_memory_region(region_name="ro")

            avg = np.mean(bitstrings[:, 0])
            results.append({
                'qubit': qubit,
                'phase': theta,
                'n_bitstrings': len(bitstrings),
                'avg': float(avg),
            })

    if filename:
        pd.DataFrame(results).to_json(filename)
    return pd.DataFrame(results)


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


def sinusoidal_waveform(x: float,
                        amplitude: float,
                        baseline: float,
                        frequency: float,
                        x_offset: float) -> np.ufunc:
    """
    Calculate sinusoidal response at a series of points.

    :param x: The independent variable with respect to which the sinusoidal response is calculated.
    :param amplitude: The amplitude of the sinusoid.
    :param baseline: The baseline of the sinusoid.
    :param frequency: The frequency of the sinusoid.
    :param x_offset: The x offset of the sinusoid.
    :return: The sinusoidal response at the given phases(s).
    """
    return amplitude * np.sin(frequency * x + x_offset) + baseline


def fit_to_sinusoidal_waveform(x_data: np.ndarray,
                               y_data: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit experimental data to sinusoid.

    :param x_data: Independent data to fit to.
    :param y_data: Experimental, dependent data to fit to.
    :return: Arrays of fitted decay curve parameters and their standard deviations
    """
    params, params_covariance = optimize.curve_fit(sinusoidal_waveform, x_data, y_data,
                                                   p0=[0.5, 0.5, 1.0, np.pi / 2])
    # parameter error extraction from
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    params_errs = np.sqrt(np.diag(params_covariance))

    # interleave params and params_errs
    print_params = []
    for idx in range(len(params)):
        print_params.append(params[idx])
        print_params.append(params_errs[idx])

    print("scipy curve fitting analysis returned\n"
          "amplitude:\t{:.5f} +/- {:.5f}\n"
          "baseline:\t{:.5f} +/- {:.5f}\n"
          "frequency:\t{:.5f} +/- {:.5f}\n"
          "x offset:\t{:.5f} +/- {:.5f}".format(*print_params))

    return params, params_errs


def get_peak_from_fit_params(fit_params: np.ndarray,
                             fit_params_errs: np.ndarray) -> Tuple[float, float]:
    """
    Extract peak from the fit parameters returned by scipy.optimize.curve_fit.

    :param fit_params: fit parameters out of scipy.optimize.curve_fit
    :param fit_params_errs: standard deviations on the fit parameters from scipy.optimize.curve_fit
    :return: The phase corresponding the to the maximum excited state visibility and its st. dev.
    """
    # TODO: do away with hard-coded indices for fit params
    x0 = fit_params[-1]
    x0_err = fit_params_errs[-1]
    freq = fit_params[-2]
    freq_err = fit_params_errs[-2]

    print("propagating error using x_0 = {} +/- {} and freq = {} +/- {}".format(x0, x0_err,
                                                                                freq, freq_err))

    # find the phase corresponding to maximum excited state visibility (ESV) using the fit params
    max_ESV = (np.pi / 2 - x0) / freq
    # max_ESV_err obtained by applying error propagation formula to max_ESV
    max_ESV_err = np.sqrt((x0_err / freq) ** 2 + ((np.pi / 2 - x0) * (freq_err / freq ** 2)) ** 2)

    print("\nmaximum excited state visibility observed at x = {} +/- {}".format(max_ESV,
                                                                                max_ESV_err))

    return max_ESV, max_ESV_err


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


def compile_parametric_program(qc: QuantumComputer,
                               parametric_prog: Program,
                               num_shots: int = 1000) -> None:
    """
    Compile the parametric program, and transfer the binary to the quantum device.

    :param qc: The QuantumComputer to run the experiment on.
    :param parametric_prog: The parametric program to compile and transfer to the quantum device.
    :param num_shots: The number of shots to average over for each data point.
    :return: The binary from the compiled parametric program.
    """
    parametric_prog.wrap_in_numshots_loop(shots=num_shots)
    binary = qc.compiler.native_quil_to_executable(parametric_prog)
    return binary


def remove_qubits_from_qubit_list(qubit_list: List[int],
                                  qubits_to_remove: Union[int, List[int]]) -> Union[int, List[int]]:
    """
    Remove the selected qubits from the given list and return the pruned list.

    :param qubit_list: The qubit list to remove the selected qubits from.
    :param qubits_to_remove: The qubits to remove from the selected list.
    :return: The given qubit list with the selected qubits removed
    """
    # cast qubits_to_remove as a list
    try:
        len(qubits_to_remove)
    except TypeError:
        qubits_to_remove = [qubits_to_remove]

    # remove list of qubits_to_remove
    new_qubit_list = list(set(qubit_list) - set(qubits_to_remove))

    # return an int or a list, as appropriate
    if len(new_qubit_list) == 1:
        return new_qubit_list[0]
    else:
        return new_qubit_list
