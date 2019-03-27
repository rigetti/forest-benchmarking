from typing import Union, List, Tuple

import numpy as np
import pandas as pd
from scipy import optimize
from matplotlib import pyplot as plt

from pyquil.api import QuantumComputer
from pyquil.gates import RX, RZ, CZ, MEASURE
from pyquil.quil import Program
from pyquil.quilbase import Pragma

MILLISECOND = 1e-6  # A millisecond (ms) is an SI unit of time
MICROSECOND = 1e-6  # A microsecond (us) is an SI unit of time
NANOSECOND = 1e-9  # A nanosecond (ns) is an SI unit of time

# A Hertz (Hz) is a derived unit of frequency in SI Units; 1 Hz is defined as one cycle per second.
KHZ = 1e3  # kHz
MHZ = 1e6  # MHz
GHZ = 1e9  # GHz


# ==================================================================================================
#   T1
# ==================================================================================================

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
                            n_points: int = 15) -> pd.DataFrame:
    """
    Return a DataFrame containing programs which ran in sequence constitute a t1 experiment to
    measure the decay time from the excited state to ground state.

    :param qubits: Which qubits to measure.
    :param stop_time: The maximum decay time to measure at.
    :param n_shots: The number of shots to average over for each data point.
    :param num_points: The number of points for each t1 curve.
    :return: A dataframe with columns: time, t1 program
    """
    start_time = 0
    time_and_programs = []

    for t in np.linspace(start_time, stop_time, n_points):
        t = round(t, 7)  # try to keep time on 100ns boundaries
        time_and_programs.append({
            'Time': t,
            'Program': generate_single_t1_experiment(qubits, t, n_shots)
        })
    return pd.DataFrame(time_and_programs)


def acquire_t1_data(qc: QuantumComputer,
                    t1_experiment: pd.DataFrame,
                    ) -> pd.DataFrame:
    """
    Execute experiments to measure the T1 decay time of 1 or more qubits.

    :param qc: The QuantumComputer to run the experiment on
    :param t1_experiment: A pandas DataFrame with columns: time, t1 program
    :return: pandas DataFrame
    """
    results = []

    for index, row in t1_experiment.iterrows():
        t = row['Time']
        program = row['Program']

        executable = qc.compiler.native_quil_to_executable(program)
        bitstrings = qc.run(executable)

        qubits = list(program.get_qubits())
        for i in range(len(qubits)):
            avg = np.mean(bitstrings[:, i])
            results.append({
                'Qubit': qubits[i],
                'Time': t,
                'Num_bitstrings': len(bitstrings),
                'Average': float(avg),
                'Program': program,
            })

    df = pd.DataFrame(results)
    return df


def estimate_t1(df: pd.DataFrame):
    """
    Estimate T1 from experimental data.

    :param df: A pandas DataFrame of experimental T1 results to plot
    :return: pandas DataFrame
    """
    results = []

    for q in df['Qubit'].unique():
        df2 = df[df['Qubit'] == q].sort_values('Time')
        x_data = df2['Time']
        y_data = df2['Average']

        try:
            fit_params, fit_params_errs = fit_to_exponential_decay_curve(x_data, y_data)
            results.append({
                'Qubit': q,
                'T1': fit_params[1] / MICROSECOND,
                'Fit_params': fit_params,
                'Fit_params_errs': fit_params_errs,
                'Message': None,
            })
        except RuntimeError:
            print(f"Could not fit to experimental data for qubit {q}")
            results.append({
                'Qubit': q,
                'T1': None,
                'Fit_params': None,
                'Fit_params_errs': None,
                'Message': 'Could not fit to experimental data for qubit' + str(q),
            })
    return pd.DataFrame(results)


def plot_t1_estimate_over_data(df: pd.DataFrame,
                               df_est: pd.DataFrame,
                               qubits: list = None,
                               filename: str = None) -> None:
    """
    Plot T1 experimental data and estimated value of T1 as and exponential decay curve.

    :param df: A pandas DataFrame experimental results to plot
    :param df_est: A pandas DataFrame with estimates of T1.
    :param qubits: A list of qubits that you actually want plotted. The default is all qubits.
    :param qc_type: String indicating whether QVM or QPU was used to collect data.
    :return: None
    """
    if qubits is None:
        qubits = df['Qubit'].unique().tolist()

    # check the user specified valid qubits
    for qbx in qubits:
        if qbx not in df['Qubit'].unique():
            raise ValueError("The list of qubits does not match the ones you experimented on.")

    for q in qubits:
        df2 = df[df['Qubit'] == q].sort_values('Time')
        x_data = df2['Time']
        y_data = df2['Average']

        plt.plot(x_data / MICROSECOND, y_data, 'o-', label=f"QC{q} T1 data")

        row = df_est[df_est['Qubit'] == q]

        if row['Fit_params'].values[0] is None:
            print(f"T1 estimate did not succeed for qubit {q}")
        else:
            fit_params = (row['Fit_params'].values)[0]
            plt.plot(x_data / MICROSECOND, exponential_decay_curve(x_data, *fit_params),
                     label=f"QC{q} fit: T1={fit_params[1] / MICROSECOND:.2f}us")

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
def generate_single_t2_star_experiment(qubits: Union[int, List[int]],
                                       time: float,
                                       detuning: float,
                                       n_shots: int = 1000) -> Program:
    """
    Return a T2 star program in native Quil for a single time point.

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


def generate_t2_star_experiments(qubits: Union[int, List[int]],
                                 stop_time: float,
                                 detuning: float = 5e6,
                                 n_shots: int = 1000,
                                 num_points: int = 15)  -> pd.DataFrame:
    """
    Return a DataFrame containing programs which ran in sequence constitute a T2 star
    experiment to measure the T2 star coherence decay time.

    :param qubits: Which qubits to measure.
    :param stop_time: The maximum decay time to measure at.
    :param detuning: The additional detuning frequency about the z axis.
    :param n_shots: The number of shots to average over for each data point.
    :param num_points: The number of points for each T2 curve.
    :return: pandas DataFrame with columns: time, program, detuning
    """
    start_time = 0
    time_and_programs = []
    for t in np.linspace(start_time, stop_time, num_points):
        # TODO: avoid aliasing while being mindful of the 20ns resolution in the QCS stack
        time_and_programs.append({
            'Time': t,
            'Program': generate_single_t2_star_experiment(qubits, t, detuning, n_shots=n_shots),
            'Detuning': detuning,
        })
    return pd.DataFrame(time_and_programs)


def generate_single_t2_echo_experiment(qubits: Union[int, List[int]],
                                       time: float,
                                       detuning: float,
                                       n_shots: int = 1000) -> Program:
    """
    Return a T2 echo program in native Quil for a single time point.

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
        # prepare plus state |+>
        program += RX(np.pi / 2, q)
        # wait half of the delay
        program += Pragma('DELAY', [q], str(time / 2))
        # apply an X gate compiled out of RX(90)
        program += RX(np.pi / 2, q)
        program += RX(np.pi / 2, q)
        # wait the other half of the delay
        program += Pragma('DELAY', [q], str(time / 2))
        program += RZ(2 * np.pi * time * detuning, q)
        program += RX(np.pi / 2, q)
    for i in range(len(qubits)):
        program += MEASURE(qubits[i], ro[i])
    program.wrap_in_numshots_loop(n_shots)
    return program


def generate_t2_echo_experiments(qubits: Union[int, List[int]],
                                 stop_time: float,
                                 detuning: float = 5e6,
                                 n_shots: int = 1000,
                                 num_points: int = 15) -> pd.DataFrame:
    """
    Return a DataFrame containing programs which ran in sequence constitute a T2 echo
    experiment to measure the T2 echo coherence decay time.

    :param qubits: Which qubits to measure.
    :param stop_time: The maximum decay time to measure at.
    :param detuning: The additional detuning frequency about the z axis.
    :param n_shots: The number of shots to average over for each data point.
    :param num_points: The number of points for each T2 curve.
    :return: pandas DataFrame with columns: time, program, detuning
    """
    start_time = 0
    time_and_programs = []
    for t in np.linspace(start_time, stop_time, num_points):
        # TODO: avoid aliasing while being mindful of the 20ns resolution in the QCS stack
        time_and_programs.append({
            'Time': t,
            'Program': generate_single_t2_echo_experiment(qubits, t, detuning, n_shots=n_shots),
            'Detuning': detuning,
        })
    return pd.DataFrame(time_and_programs)


def acquire_t2_data(qc: QuantumComputer,
                    t2_experiment: pd.DataFrame) -> pd.DataFrame:
    """
    Execute experiments to measure the T2 star or T2 echo decay time of 1 or more qubits.

    :param qc: The QuantumComputer to run the experiment on
    :param t2_experiment: A pandas DataFrame containing: time, T2 program
    :return: pandas DataFrame containing T2 results, and detuning used in creating experiments for
    those results.
    """
    results = []

    for index, row in t2_experiment.iterrows():
        t = row['Time']
        program = row['Program']
        detuning = row['Detuning']
        executable = qc.compiler.native_quil_to_executable(program)
        bitstrings = qc.run(executable)

        qubits = list(program.get_qubits())
        for i in range(len(qubits)):
            avg = np.mean(bitstrings[:, i])
            results.append({
                'Qubit': qubits[i],
                'Time': t,
                'Num_bitstrings': len(bitstrings),
                'Average': float(avg),
                'Detuning': float(detuning),
            })

    return pd.DataFrame(results)


def estimate_t2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate T2 star or T2 echo from experimental data.

    :param df: A pandas DataFrame with experimental T2 results
    :param detuning: Detuning frequency used in experiment creation
    :return: pandas DataFrame
    """
    results = []
    for q in df['Qubit'].unique():
        df2 = df[df['Qubit'] == q].sort_values('Time')
        x_data = df2['Time']
        y_data = df2['Average']
        detuning = df2['Detuning'].values[0]

        try:
            fit_params, fit_params_errs = fit_to_exponentially_decaying_sinusoidal_curve(x_data,
                                                                                         y_data,
                                                                                         detuning)
            results.append({
                'Qubit': q,
                'T2': fit_params[1] / MICROSECOND,
                'Freq': fit_params[2] / MHZ,
                'Fit_params': fit_params,
                'Fit_params_errs': fit_params_errs,
                'Message': None,
            })
        except RuntimeError:
            print(f"Could not fit to experimental data for qubit {q}")
            results.append({
                'Qubit': q,
                'T2': None,
                'Freq': None,
                'Fit_params': None,
                'Fit_params_errs': None,
                'Message': 'Could not fit to experimental data for qubit' + str(q),
            })

    return pd.DataFrame(results)


def plot_t2_estimate_over_data(df: pd.DataFrame,
                               df_est: pd.DataFrame,
                               qubits: list = None,
                               t2_type: str = 'unknown',
                               filename: str = None) -> None:
    """
    Plot T2 star or T2 echo experimental data and estimated value of T1 as and exponential decay
    curve.

    :param df: A pandas DataFrame containing experimental results to plot.
    :param df_est: A pandas DataFrame containing estimates of T2.
    :param qubits: A list of qubits that you actually want plotted. The default is all qubits.
    :param detuning: Detuning frequency used in experiment creation.
    :param t2_type: String either 'star' or 'echo'.
    :param filename: String.
    :return: None
    """
    if qubits is None:
        qubits = df['Qubit'].unique().tolist()

    # check the user specified valid qubits
    for qbx in qubits:
        if qbx not in df['Qubit'].unique():
            raise ValueError("The list of qubits does not match the ones you experimented on.")

    for q in qubits:
        df2 = df[df['Qubit'] == q].sort_values('Time')
        x_data = df2['Time']
        y_data = df2['Average']

        plt.plot(x_data / MICROSECOND, y_data, 'o-', label=f"Qubit {q} T2 data")

        row = df_est[df_est['Qubit'] == q]

        if row['Fit_params'].values[0] is None:
            print(f"T2 estimate did not succeed for qubit {q}")
        else:
            fit_params = row['Fit_params'].values[0]
            plt.plot(x_data / MICROSECOND,
                     exponentially_decaying_sinusoidal_curve(x_data, *fit_params),
                     label=f"QC{q} fit: freq={fit_params[2] / MHZ:.2f}MHz, "
                           f""f"T2={fit_params[1] / MICROSECOND:.2f}us")

    plt.xlabel("Time [µs]")
    plt.ylabel("Pr(1)")
    if t2_type.lower() == 'star':
        plt.title("$T_2^*$ (Ramsey) decay")
    elif t2_type.lower() == 'echo':
        plt.title("$T_2$ (Echo) decay")
    else:
        plt.title("$T_2$ (unknown) decay")

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
                              num_points: int = 15) -> pd.DataFrame:
    """
    Return a DataFrame containing programs which, when run in sequence, constitute a Rabi
    experiment.

    Rabi oscillations are observed by applying successively larger rotations to the same initial
    state.

    :param qubits: Which qubits to measure.
    :param n_shots: The number of shots to average over for each data point
    :param num_points: The number of points for each Rabi curve
    :return: pandas DataFrame with columns: angle, program
    """
    angle_and_programs = []
    for theta in np.linspace(0.0, 2 * np.pi, num_points):
        angle_and_programs.append({
            'Angle': theta,
            'Program': generate_single_rabi_experiment(qubits, theta, n_shots),
        })
    return pd.DataFrame(angle_and_programs)


def acquire_rabi_data(qc: QuantumComputer,
                      rabi_experiment: pd.DataFrame,
                      filename: str = None) -> pd.DataFrame:
    """
    Execute experiments to measure Rabi flop one or more qubits.

    :param qc: The QuantumComputer to run the experiment on
    :param rabi_experiment: pandas DataFrame: (theta, Rabi program)
    :param filename: The name of the file to write JSON-serialized results to
    :return: DataFrame with Rabi results
    """
    results = []
    for index, row in rabi_experiment.iterrows():
        theta = row['Angle']
        program = row['Program']
        executable = qc.compiler.native_quil_to_executable(program)
        bitstrings = qc.run(executable)

        qubits = list(program.get_qubits())
        for i in range(len(qubits)):
            avg = np.mean(bitstrings[:, i])
            results.append({
                'Qubit': qubits[i],
                'Angle': theta,
                'Num_bitstrings': len(bitstrings),
                'Average': float(avg),
            })

    if filename:
        pd.DataFrame(results).to_json(filename)
    return pd.DataFrame(results)


def estimate_rabi(df: pd.DataFrame):
    """
    Estimate Rabi oscillation from experimental data.

    :param df: Experimental Rabi results to estimate
    :return: pandas DataFrame
    """
    results = []

    for q in df['Qubit'].unique():
        df2 = df[df['Qubit'] == q].sort_values('Angle')
        angles = df2['Angle']
        prob_of_one = df2['Average']

        try:
            # fit to sinusoid
            fit_params, fit_params_errs = fit_to_sinusoidal_waveform(angles, prob_of_one)

            results.append({
                'Qubit': q,
                'Angle': fit_params[1],
                'Prob_of_one': fit_params[2],
                'Fit_params': fit_params,
                'Fit_params_errs': fit_params_errs,
                'Message': None,
            })
        except RuntimeError:
            print(f"Could not fit to experimental data for qubit {q}")
            results.append({
                'Qubit': q,
                'Angle': None,
                'Prob_of_one': None,
                'Fit_params': None,
                'Fit_params_errs': None,
                'Message': 'Could not fit to experimental data for qubit' + str(q),
            })
    return pd.DataFrame(results)


def plot_rabi_estimate_over_data(df: pd.DataFrame,
                                 df_est: pd.DataFrame,
                                 qubits: list = None,
                                 filename: str = None) -> None:
    """
    Plot Rabi oscillation experimental data and estimated curve.

    :param df: Experimental results to plot and fit curve to.
    :param df_est: Estimates of Rabi oscillation.
    :param qubits: A list of qubits that you actually want plotted. The default is all qubits.
    :param filename: The name of the file to write JSON-serialized results to
    :return: None
    """
    if qubits is None:
        qubits = df['Qubit'].unique().tolist()

    # check the user specified valid qubits
    for qbx in qubits:
        if qbx not in df['Qubit'].unique():
            raise ValueError("The list of qubits does not match the ones you experimented on.")

    for q in qubits:
        df2 = df[df['Qubit'] == q].sort_values('Angle')
        angles = df2['Angle']
        prob_of_one = df2['Average']

        # plot raw data
        plt.plot(angles, prob_of_one, 'o-', label=f"qubit {q} Rabi data")

        row = df_est[df_est['Qubit'] == q]

        if row['Fit_params'].values[0] is None:
            print(f"Rabi estimate did not succeed for qubit {q}")
        else:
            fit_params = row['Fit_params'].values[0]
            # overlay fitted sinusoidal curve
            plt.plot(angles, sinusoidal_waveform(angles, *fit_params),
                     label=f"qubit {q} fitted line")

    plt.xlabel("RX angle [rad]")
    plt.ylabel("Pr($|1\langle)")
    plt.title("Rabi flop")
    plt.legend(loc='best')
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


# ==================================================================================================
#   CZ phase Ramsey
# ==================================================================================================

def generate_cz_phase_ramsey_program(qb: int, other_qb: int, n_shots: int = 1000) -> Program:
    """
    Generate a single CZ phase Ramsey experiment at a given phase.

    :param qb: The qubit to move around the Bloch sphere and measure the incurred RZ on.
    :param other_qb: The other qubit that constitutes a two-qubit pair along with `qb`.
    :param n_shots: The number of shots to average over for each data point.
    :param phase: The phase kick to supply after playing the CZ pulse on the equator.
    :param n_shots: The number of shots to average over for the data point.
    :return: A parametric Program for performing a CZ Ramsey experiment.
    """
    program = Program()
    # NOTE: only need readout register for `qb` not `other_qb` since `other_qb` is only
    #       needed to identify which CZ gate we're using
    ro = program.declare('ro', 'BIT', 1)
    theta = program.declare('theta', 'REAL')

    # go to the equator
    program += Program(RX(np.pi / 2, qb))
    # apply the CZ gate - note that CZ is symmetric, so the order of qubits doesn't matter
    program += Program(CZ(qb, other_qb))
    # go to |1> after a phase kick
    program += Program(RZ(theta, qb), RX(np.pi / 2, qb))

    program += MEASURE(qb, ro[0])

    program.wrap_in_numshots_loop(n_shots)
    return program


def generate_cz_phase_ramsey_experiment(edges: List[Tuple[int, int]],
                                        start_phase: float = 0.0,
                                        stop_phase: float = 2 * np.pi,
                                        num_points: int = 15,
                                        num_shots: int = 1000):
    """
    Returns a DataFrame of parameters and programs that constitute a CZ phase ramsey experiment.

    :param edges: List of Tuples containing edges that one can perform a CZ on.
    :param start_phase: The starting phase for the CZ phase Ramsey experiment.
    :param stop_phase: The stopping phase for the CZ phase Ramsey experiment.
    :param num_points: The number of points to sample at between the starting and stopping phase.
    :param num_shots: The number of shots to average over for each data point.
    :return: pandas DataFrame
    """

    cz_expriment = []
    for edge in edges:
        qubit, other_qubit = edge

        # first qubit gets RZ
        cz_expriment.append({
            'Edge': tuple(edge),
            'Rz_qubit': qubit,
            'Program': generate_cz_phase_ramsey_program(qubit, other_qubit, num_shots),
            'Start_phase': start_phase,
            'Stop_phase': stop_phase,
            'Num_points': num_points,
            'Num_shots': num_shots,
        })

        # second qubit gets RZ
        cz_expriment.append({
            'Edge': tuple(edge),
            'Rz_qubit': other_qubit,
            'Program': generate_cz_phase_ramsey_program(other_qubit, qubit, num_shots),
            'Start_phase': start_phase,
            'Stop_phase': stop_phase,
            'Num_points': num_points,
            'Num_shots': num_shots,
        })

    return pd.DataFrame(cz_expriment)


def acquire_cz_phase_ramsey_data(qc: QuantumComputer,
                                 cz_experiment: pd.DataFrame,
                                 filename: str = None) -> pd.DataFrame:
    """
    Execute experiments to measure the RZ incurred as a result of a CZ gate.

    :param qc: The qubit to move around the Bloch sphere and measure the incurred RZ on
    :param cz_experiment: pandas DataFrame
    :param filename: The name of the file to write JSON-serialized results to
    :return: pandas DataFrame
    """
    results = []

    for index, row in cz_experiment.iterrows():
        parametric_ramsey_prog = row['Program']
        edge = row['Edge']
        rz_qb = row['Rz_qubit']
        start_phase = row['Start_phase']
        stop_phase = row['Stop_phase']
        num_points = row['Num_points']
        num_shots = row['Num_shots']

        binary = compile_parametric_program(qc, parametric_ramsey_prog, num_shots=num_shots)

        qc.qam.load(binary)

        for theta in np.linspace(start_phase, stop_phase, num_points):
            qc.qam.write_memory(region_name='theta', value=theta)
            qc.qam.run()
            qc.qam.wait()
            bitstrings = qc.qam.read_from_memory_region(region_name="ro")

            avg = np.mean(bitstrings[:, 0])
            results.append({
                'Edge': edge,
                'Rz_qubit': rz_qb,
                'Phase': theta,
                'Num_bitstrings': len(bitstrings),
                'Average': float(avg),
            })

    if filename:
        pd.DataFrame(results).to_json(filename)
    return pd.DataFrame(results)


def estimate_cz_phase_ramsey(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate CZ phase ramsey experimental data.

    :param df: Experimental results to plot and fit exponential decay curve to.
    :return: List of dicts.
    """
    results = []
    edges = df['Edge'].unique()

    for id_row, edge in enumerate(edges):

        for id_col, qubit in enumerate(edge):
            qubit_df = df[(df['Rz_qubit'] == qubit) & (df['Edge'] == edge)].sort_values('Phase')
            phases = qubit_df['Phase']
            prob_of_one = qubit_df['Average']
            rz_qb = qubit_df['Rz_qubit'].values[0]

            try:
                # fit to sinusoid
                fit_params, fit_params_errs = fit_to_sinusoidal_waveform(phases, prob_of_one)
                # find max excited state visibility (ESV) and propagate error from fit params
                max_ESV, max_ESV_err = get_peak_from_fit_params(fit_params, fit_params_errs)

                results.append({
                    'Edge': edge,
                    'Rz_qubit': rz_qb,
                    'Angle': fit_params[1],
                    'Prob_of_one': fit_params[2],
                    'Fit_params': fit_params,
                    'Fit_params_errs': fit_params_errs,
                    'max_ESV': max_ESV,
                    'max_ESV_err': max_ESV_err,
                    'Message': None,
                })
            except RuntimeError:
                print(f"Could not fit to experimental data for edge {edge}")
                results.append({
                    'Edge': edge,
                    'Rz_qubit': rz_qb,
                    'Angle': None,
                    'Prob_of_one': None,
                    'Fit_params': None,
                    'Fit_params_errs': None,
                    'max_ESV': None,
                    'max_ESV_err': None,
                    'Message': 'Could not fit to experimental data for edge' + str(edge),
                })
    return pd.DataFrame(results)


def plot_cz_phase_estimate_over_data(df: pd.DataFrame,
                                     df_est: pd.DataFrame,
                                     filename: str = None) -> None:
    """
    Plot Ramsey experimental data, the fitted sinusoid, and the maximum of that sinusoid.

    :param df: Experimental results to plot and fit exponential decay curve to.
    :param df_est: estimates of CZ Ramsey experiments
    :param filename: The name of the file to write JSON-serialized results to
    :return: None
    """
    edges = df['Edge'].unique()
    if len(edges) == 1:
        # this deals with the one edge case, then plot will have an empty row
        # if you don't do this you get `axes.shape = (2,)`
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(24, 30))
    else:
        fig, axes = plt.subplots(nrows=len(edges), ncols=2, figsize=(24, 10 * len(edges)))

    for id_row, edge in enumerate(edges):

        for id_col, qubit in enumerate(edge):
            qubit_df = df[(df['Rz_qubit'] == qubit) & (df['Edge'] == edge)].sort_values('Phase')
            phases = qubit_df['Phase']
            prob_of_one = qubit_df['Average']

            # plot raw data
            axes[id_row, id_col].plot(phases, prob_of_one, 'o',
                                      label=f"qubit{qubit} CZ Ramsey data")

            row = df_est[df_est['Rz_qubit'] == qubit]

            if row['Fit_params'].values[0] is None:
                print(f"Rabi estimate did not succeed for qubit {q}")
            else:
                fit_params = row['Fit_params'].values[0]
                max_ESV = row['max_ESV'].values[0]
                max_ESV_err = row['max_ESV_err'].values[0]

                # overlay fitted curve and vertical line at maximum ESV
                axes[id_row, id_col].plot(phases, sinusoidal_waveform(phases, *fit_params),
                                          label=f"QC{qubit} fitted line")
                axes[id_row, id_col].axvline(max_ESV,
                                             label=f"QC{qubit} max ESV={max_ESV:.3f}+/-{max_ESV_err:.3f} rad")

            axes[id_row, id_col].set_xlabel("Phase on second +X/2 gate [rad]")
            axes[id_row, id_col].set_ylabel("Pr($|1\langle)")
            axes[id_row, id_col].set_title(f"CZ Phase Ramsey fringes on QC{qubit}\n"
                                           f"due to CZ_{edge[0]}_{edge[1]} application")
            axes[id_row, id_col].legend(loc='best')
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
                               y_data: List[float],
                               displayflag: bool = False,
                               ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit experimental data to sinusoid.

    :param x_data: Independent data to fit to.
    :param y_data: Experimental, dependent data to fit to.
    :param displayflag: If True displays results from scipy curve fit analysis.
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

    if displayflag:
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
