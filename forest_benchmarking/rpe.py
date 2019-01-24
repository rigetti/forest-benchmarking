from typing import Tuple, List

import numpy as np
from numpy import pi
from pandas import DataFrame, Series

from pyquil.gates import I, RX, RY, RZ, X
from pyquil.quil import Program
from pyquil.api import QuantumComputer
from forest_benchmarking.compilation import basic_compile
from forest_benchmarking.utils import transform_bit_moments_to_pauli, local_pauli_eig_prep, local_pauli_eig_meas
import warnings

import matplotlib.pyplot as plt


def prepare_state(experiment: Program, qubit: int, axis: Tuple = None) -> None:
    """
    Initialize the state for the given experiment. If the experiment is to estimate angle for
    RZ(angle), the initial state would be the plus 1 eigenstate of X, or |+> = RY(pi/2) |0>. For
    rotation about an arbitrary axis, the initial state is equivalently RZ(phi)RY(theta + pi/2)|0>
    where axis=(theta, phi) specifies the axis of rotation.

    :param prog: the program comprising the experiment, which begins with state preparation.
    :param qubit: the qubit whose state is being prepared
    :param axis: axis of rotation specified as (theta, phi) in typical spherical coordinates.
    (Mutates the experiment program)
    """
    if axis:
        experiment.inst(RY(pi / 2 + axis[0], qubit))
        experiment.inst(RZ(axis[1], qubit))
    else:
        local_pauli_eig_prep(experiment, 'X', qubit)


def generate_single_depth_experiment(rotation: Program, depth: int, exp_type: str, axis: Tuple = None) -> Program:
    """
    Generate an experiment for a single depth where the type specifies a final measurement of either
    X or Y. The rotation program is repeated depth number of times, and we assume the rotation is
    about the axis (theta, phi).

    :param rotation: the program specifying the gate whose angle of rotation we wish to estimate.
    :param depth: the number of times we apply the rotation in the experiment
    :param exp_type: X or Y, specifying which operator to measure at the end of the experiment
    :param axis: the axis of rotation. If none is specified, axis is assumed to be the Z axis. (rotation should be RZ)
    :return: a program specifying the entire experiment of a single iteration of the RPE protocol in [RPE]
    """
    experiment = Program()
    ro_bit = experiment.declare("ro", "BIT", 1)
    qubit = list(rotation.get_qubits())[0]
    prepare_state(experiment, qubit, axis)
    for _ in range(depth):
        experiment.inst(rotation)
    if axis:
        experiment.inst(RZ(-axis[1], qubit))
        experiment.inst(RY(-axis[0], qubit))
    local_pauli_eig_meas(experiment, exp_type, qubit)
    experiment.measure(qubit, ro_bit)
    return experiment


def generate_2q_single_depth_experiment(rotation: Program, depth: int, exp_type: str,
                                        measurement_qubit: int, init_one: bool = False, axis: Tuple = None) -> Program:
    """
    A special variant of the 1q method that is specifically designed to calibrate a CPHASE gate. The
    ideal CPHASE is of the following form
        CPHASE(\phi) = diag(1,1,1,Exp[-i \phi]
    The imperfect CPHASE has two local Z rotations and a possible over (or under) rotation on the
    phase phi. Thus we have
        CPHASE(\Phi, \Theta_1, \Theta_2) = diag( exp(-a -b), exp(-a + b), exp(a-b), exp(a+b+c) )
        a = i \Theta_1 / 2,     b = i \Theta_2 / 2,     c = i \Phi

    The following experiments isolate the three angles using the state preparations |0>|+>, |+>|0>,
    |1>|+>, |+>|1> where the incurred phase is measured on the qubit initialized to the plus state.
    The four measurements are specified by setting the measurement qubit to either q1 or q2, and
    setting init_one to True indicating that the non-measurement qubit be prepared in the one state
    |1>.

    :param rotation: the program specifying the gate whose angle of rotation we wish to estimate.
    :param depth: the number of times we apply the rotation in the experiment
    :param exp_type: X or Y, specifying which operator to measure at the end of the experiment
    :param measurement_qubit: the qubit to be measured in this variant of the experiment
    :param axis: the axis of rotation. If none is specified, axis is assumed to be the Z axis. (rotation should be RZ)
    :param init_one: True iff the non-measurement qubit should be prepared in the 1 state.
    :param axis: the axis of rotation. If none is specified, axis is assumed to be the Z axis. (rotation should be RZ)
    :return: An estimate of some aspect of the CPHASE gate which depends on the measurement variant.
    """
    prog = Program()
    ro_bit = prog.declare("ro", "BIT", 1)
    qubits = rotation.get_qubits()
    non_measurement_qubit = list(qubits - {measurement_qubit})[0]
    prepare_state(prog, measurement_qubit)
    if init_one:
        prog.inst(X(non_measurement_qubit))
    for _ in range(depth):
        prog.inst(rotation)
    if axis:
        prog.inst(RZ(-axis[1], measurement_qubit))
        prog.inst(RY(-axis[0], measurement_qubit))
    local_pauli_eig_meas(prog, exp_type, measurement_qubit)
    prog.measure(measurement_qubit, ro_bit)
    return prog


def generate_rpe_experiments(rotation: Program, num_depths: int = 5, axis: Tuple[float, float] = None,
                             measurement_qubit=None, init_one=False) -> DataFrame:
    """
    Generate a dataframe containing all the experiments needed to perform robust phase estimation of
    a gate.

    The algorithm is due to:

    [RPE]  Robust Calibration of a Universal Single-Qubit Gate-Set via Robust Phase Estimation
            Kimmel et al., Phys. Rev. A 92, 062315 (2015)
            https://doi.org/10.1103/PhysRevA.92.062315

    [RPE2] Experimental Demonstration of a Cheap and Accurate Phase Estimation
            Rudinger et al., Phys. Rev. Lett. 118, 190502 (2017)
            https://doi.org/10.1103/PhysRevLett.118.190502

    :param rotation: the program or gate whose angle of rotation is to be estimated
    :param num_depths: the number of depths in the protocol described in [RPE]. Max depth = 2**(num_depths-1)
    :param axis: the axis of rotation specified by (theta, phi). Assumed to be the Z axis if none is specified.
    :param measurement_qubit: Pertinent only to CPHASE experiment. See generate_2q_single_depth_experiment above
    :param init_one: Pertinent only to CPHASE experiment. See generate_2q_single_depth_experiment above
    :return:
    """
    if axis is None and \
            (len(Program(rotation).instructions) > 1 or Program(rotation).instructions[0].name not in ['Z', 'RZ']):
        warnings.warn("If the rotation provided is not about the Z axis, "
                      "remember to specify an axis of rotation in polar coordinates (theta, phi) radians")

    def df_dict():
        for exponent in range(num_depths):
            depth = 2 ** exponent
            for exp_type in ['X', 'Y']:
                if measurement_qubit is None:
                    yield {"Depth": depth,
                           "Exp_Type": exp_type,
                           "Experiment": generate_single_depth_experiment(rotation, depth, exp_type, axis)}
                else:
                    # Pertinent only to CPHASE experiment. See generate_2q_single_depth_experiment above
                    yield {"Depth": depth,
                           "Exp_Type": exp_type,
                           "Experiment": generate_2q_single_depth_experiment(rotation, depth, exp_type,
                                                                             measurement_qubit,
                                                                             init_one=init_one, axis=axis)}

    # TODO: Put dtypes on this DataFrame in the right way
    return DataFrame(df_dict())


def get_additive_error_factor(M_j: float, max_additive_error: float) -> float:
    """
    Calculate the factor in Equation V.17 of [RPE] that multiplies the number of trials at the jth
    iteration in order to maintain Heisenberg scaling with the same variance upper bound as with no
    additive error. This holds as long as the actual max_additive_error in the procedure is no more
    than 1/sqrt(8) ~=~ .354 error present in the procedure

    :param M_j: the number of shots in the jth iteration of RPE
    :param max_additive_error: the assumed maximum of the additive errors you hope to adjust for
    :return: A factor that when multiplied by M_j yields a number of shots which preserves Heisenberg Scaling
    """
    return np.log(.5 * (1 - np.sqrt(8) * max_additive_error) ** (1 / M_j)) \
           / np.log(1 - .5 * (1 - np.sqrt(8) * max_additive_error) ** 2)


def num_trials(depth, max_depth, alpha, beta, multiplicative_factor: float = 1.0, additive_error: float = None) -> int:
    """
    Calculate the optimal number of shots per experiment with a given depth, as described by
    equations V.11 and V.17 in [RPE]

    :param depth: the depth of the experiment whose number of trials is calculated
    :param max_depth: maximum depth of the experiments
    :param alpha: a hyper-parameter in equation V.11 of [RPE], suggested to be 5/2
    :param beta: a hyper-parameter in equation V.11 of [RPE], suggested to be 1/2
    :param multiplicative_factor: An additional add-hoc factor that multiplies the optimal number of shots
    :param additive_error: an estimate of the maximum additive error in the experiment, eq. V.15 of [RPE]
    :return: Mj, the number of shots for experiment with depth 2**(j-1) in iteration j of RPE
    """
    j = np.log2(depth) + 1
    K = np.log2(max_depth) + 1
    Mj = (alpha * (K - j) + beta)
    if additive_error:
        multiplicative_factor *= get_additive_error_factor(Mj, additive_error)
    return int(np.ceil(Mj * multiplicative_factor).astype(int))


def acquire_rpe_data(experiments: DataFrame, qc: QuantumComputer, multiplicative_factor: float = 1.0,
                     additive_error: float = None, results_label="Results") -> DataFrame:
    """
    Run each experiment in the experiments data frame a number of times which is specified by
    num_trials. Store the raw shot values in a column labeled by results_label.

    :param experiments: dataframe containing experiments generated by a call to generate_rpe_experiments
    :param qc: a quantum computer, e.g. QVM or QPU, that runs the experiments
    :param multiplicative_factor: an ad-hoc factor to multiply the number of shots at each iteration. See num_trials
    :param additive_error: an estimate of the maximum additive error in the experiment, eq. V.15 of [RPE]
    :param results_label: label for the column with results that is added to the copied experiments data frame.
    :return: A copy of the experiments data frame with the results in a new column.
    """

    def run(qc: QuantumComputer, exp: Program, n_trials: int) -> np.ndarray:
        exp.wrap_in_numshots_loop(n_trials)
        executable = qc.compiler.native_quil_to_executable(basic_compile(exp))
        return qc.run(executable)

    alpha = 5 / 2  # should be > 2
    beta = 1 / 2  # should be > 0
    max_depth = max(experiments["Depth"].values)
    results = [run(qc, experiment,
                   num_trials(depth, max_depth, alpha, beta, multiplicative_factor, additive_error))
               for (depth, experiment) in zip(experiments["Depth"].values, experiments["Experiment"].values)]
    experiments = experiments.copy()
    experiments[results_label] = Series(results)
    return experiments


#########
# Analysis
#########


def p_max(M_j: int) -> float:
    """
    Calculate an upper bound on the probability of error in the estimate on the jth iteration.
    Equation V.6 in [RPE]

    :param M_j: The number of shots for the jth iteration of RPE
    :return: p_max(M_j), an upper bound on the probability of error on the estimate k_j * Angle
    """
    return (1 / np.sqrt(2 * pi * M_j)) * (2 ** -M_j)


def xci(h: int) -> float:
    """
    Calculate the maximum error in the estimate after h iterations given that no errors occurred in
    all previous iterations. Equation V.7 in [RPE]

    :param h: the iteration before which we assume no errors have occured in our estimation.
    :return: the maximum error in our estimate, given h
    """
    return 2 * pi / (2 ** h)


def get_variance_upper_bound(experiments: DataFrame, results_label='Results') -> float:
    """
    Equation V.9 in [RPE]

    :param experiments: a dataframe with RPE results. Importantly the bound follows from the number
    of shots at each iteration of the experiment, so experiments needs to be populated with the
    desired number of shots results.
    :param results_label: label for the column with results from which the variance is estimated
    :return: An upper bound of the variance of the angle estimate corresponding to the input
    experiments.
    """
    max_depth = max(experiments["Depth"].values)
    K = np.log2(max_depth).astype(int) + 1

    M_js = []
    # 1 <= j <= K, where j is the one-indexed iteration number
    for j in range(1, K + 1):
        single_depth = experiments.groupby(["Depth"]).get_group(2 ** (j - 1)).set_index('Exp_Type')
        M_j = len(single_depth.loc['X', results_label])
        M_js += [M_j]

    # note that M_js is 0 indexed but 1 <= j <= K, so M_j = M_js[j-1]
    return (1 - p_max(M_js[K - 1])) * xci(K + 1) ** 2 + sum(
        [xci(i + 1) ** 2 * p_max(M_j) for i, M_j in enumerate(M_js)])


def find_expectation_values(experiments: DataFrame, results_label='Results') -> \
        Tuple[List, List, List, List]:
    """
    Calculate expectation values and standard deviation of the mean for each depth and
    experiment type.

    :param experiments: a dataframe with RPE results populated by a call to acquire_rpe_data
    :param results_label: label for the column with results from which the variance is estimated
    """
    xs = []
    ys = []
    x_stds = []
    y_stds = []

    for depth, group in experiments.groupby(["Depth"]):
        N = len(group[group['Exp_Type'] == 'X'][results_label].values[0])

        p_x = group[group['Exp_Type'] == 'X'][results_label].values[0].mean()
        p_y = group[group['Exp_Type'] == 'Y'][results_label].values[0].mean()
        # standard deviation of the mean of the probabilities
        p_x_std = group[group['Exp_Type'] == 'X'][results_label].values[0].std() / np.sqrt(N)
        p_y_std = group[group['Exp_Type'] == 'Y'][results_label].values[0].std() / np.sqrt(N)
        # convert probabilities to expectation values of X and Y
        exp_x, var_x = transform_bit_moments_to_pauli(1-p_x, p_x_std**2)
        exp_y, var_y = transform_bit_moments_to_pauli(1-p_y, p_y_std**2)
        xs.append(exp_x)
        ys.append(exp_y)
        # standard deviations need the scaling but not the shifting
        x_stds.append(np.sqrt(var_x))
        y_stds.append(np.sqrt(var_y))

    return xs, ys, x_stds, y_stds


def robust_phase_estimate(xs: List, ys: List, x_stds: List, y_stds: List,
                          bloch_data: List = None) -> float:
    """
    Estimate the phase in an iterative fashion as described in section V. of [RPE]
    Note: in the realistic case that additive errors are present, the estimate is biased.
    See Appendix B of [RPE] for discussion/comparison to other techniques.

    :param xs: expectation value <X> operator for each iteration
    :param ys: expectation value <Y> operator for each iteration
    :param x_std: standard deviation of the mean for 'xs'
    :param y_std: standard deviation of the mean for 'ys'
    :param bloch_data: when provided, list is mutated to store the radius and angle of each iteration
    :return: An estimate of the phase of the rotation program passed into generate_rpe_experiments
    """

    theta_est = 0
    for j, (x, y, x_std, y_std) in enumerate(zip(xs, ys, x_stds, y_stds)):
        # k is both the depth and the portion of the circle constrained by each iteration
        k = 2 ** j
        r = np.sqrt(x ** 2 + y ** 2)
        r_std = np.sqrt(x_std ** 2 + y_std ** 2)
        if r < r_std:
            # cannot reliably place the vector an any quadrant of the circle, so terminate
            break

        # get back an estimate between -pi and pi
        theta_j_est = np.arctan2(y, x) / k
        plus_or_minus = pi / k  # the principal range bound from previous estimate
        # update the estimate given that it falls within plus_or_minus of the last estimate
        offset = (theta_j_est - (theta_est - plus_or_minus)) % (2 * plus_or_minus)
        theta_est += offset - plus_or_minus

        if bloch_data is not None:
            bloch_data.append((r, theta_est * k))

    return theta_est % (2 * pi)  # return value between 0 and 2pi


#########
# Plotting
#########


def plot_RPE_iterations(experiments: DataFrame, expected_positions: List = None) -> plt.Axes:
    """
    Creates a polar plot of the estimated location of the state in the plane perpendicular to the
    axis of rotation for each iteration of RPE.

    :param experiments: a dataframe with RPE results populated by a call to acquire_rpe_data
    :param expected_positions: a list of expected (radius, angle) pairs for each iteration
    :return: a matplotlib subplot visualizing each iteration of the RPE experiment
    """
    positions = []
    xs, ys, x_stds, y_stds = find_expectation_values(experiments)
    result = robust_phase_estimate(xs, ys, x_stds, y_stds, positions)
    rs = [pos[0] for pos in positions]
    angles = [pos[1] for pos in positions]

    ax = plt.subplot(111, projection='polar')

    # observed
    ax.scatter(angles, rs)
    for j, (radius, angle) in enumerate(positions):
        ax.annotate("Ob" + str(j), (angle, radius), color='blue')

    # expected
    if expected_positions:
        expected_rs = [pos[0] for pos in expected_positions]
        expected_angles = [pos[1] for pos in expected_positions]
        ax.scatter(expected_angles, expected_rs)
        for j, (radius, angle) in enumerate(expected_positions):
            ax.annotate("Ex" + str(j), (angle, radius), color='orange')
        ax.set_title("RPE Iterations Observed(O) and Expected(E)", va='bottom')
    else:
        ax.set_title("Observed Position per RPE Iteration")

    ax.set_rmax(1.0)
    ax.set_rticks([0.5, 1])  # radial ticks
    ax.set_rlabel_position(-22.5)  # offset radial labels to lower right quadrant
    ax.grid(True)

    return ax
