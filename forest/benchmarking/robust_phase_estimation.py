from typing import Tuple, List

import numpy as np
from numpy import pi
from pandas import DataFrame, Series

from pyquil.gates import I, RX, RY, RZ, X
from pyquil.quil import Program
from pyquil.quilbase import Gate
from pyquil.api import QuantumComputer
from forest.benchmarking.compilation import basic_compile
from forest.benchmarking.utils import transform_bit_moments_to_pauli, local_pauli_eig_meas

import matplotlib.pyplot as plt


def rpe_dataframe(subgraph: List[Tuple], num_depths: int, ):

    def df_dict():
        for exponent in range(num_depths):
            depth = 2 ** exponent
            for meas_dir in ['X', 'Y']:
                yield {"Subgraph": subgraph,
                       "Depth": depth,
                       "Meas_Direction": meas_dir,
                       "Experiment": generate_single_depth_rpe_experiment(rotation, axis, depth,
                                                                          meas_dir,
                                                                          measurement_qubit,
                                                                          custom_prep)}

    # TODO: Put dtypes on this DataFrame in the right way
    return DataFrame(df_dict())


def prepare_state_about_axis(qubit: int, axis: Tuple[float, float]) -> Program:
    """
    Generates a program that prepares a state perpendicular to the given (theta, phi) axis on
    the Bloch sphere.

    In the context of an RPE experiment, the supplied axis is the axis of rotation of the gate
    whose magnitude of rotation the experimenter is trying to estimate. The first entry of axis
    is the polar angle, theta, in radians from the pauli Z axis (north pole or zero state on the
    Bloch sphere). The second entry is the azimuthal angle, phi, of the axis from the X-Z plane
    of the Bloch sphere. The prepared state is a point on the sphere whose radial line is
    perpendicular to the supplied axis; the state is pi radians in the theta direction from axis.

    For example, the axis (0, 0) corresponds to an RPE experiment estimating the angle parameter
    of the rotation RZ(angle). The initial state of this experiment would be the plus one
    eigenstate of X, or |+> = RY(pi/2) |0> since this state is perpendicular to the axis of
    rotation of RZ. For rotation about an arbitrary axis=(theta, phi), the initial state is
    equivalently RZ(phi)RY(theta + pi/2)|0>

    :param qubit: the qubit whose state is being prepared
    :param axis: axis of rotation specified as (theta, phi) in typical spherical coordinates.
    :return: A preparation program that prepares the qubit in a state perpendicular to axis.
    """
    prep = Program()
    prep += RY(pi / 2 + axis[0], qubit)
    prep += RZ(axis[1], qubit)
    return prep


def generate_single_depth_rpe_experiment(rotation: Program, axis: Tuple[float, float], depth: int,
                                         meas_dir: str, measurement_qubit: int,
                                         custom_prep: Program = None) -> Program:
    """
    Generate a single RPE experiment where rotation is applied depth many times before a final
    measurement in either the X or Y 'direction' relative to a frame with the rotation axis
    corresponding to the Z axis.

    Rotation should be the program implementing the gate whose magnitude of rotation about the
    supplied axis we wish to estimate. A single RPE experiment iteration comprises preparing a
    state perpendicular to the axis of rotation, rotating this state depth many times,
    and measuring in the plane perpendicular to the axis of rotation along one of two orthogonal
    directions; to specify which of these two directions, we adopt a frame where the axis of
    rotation corresponds to the Z axis and meas_dir takes the value either 'X' or 'Y'. For
    conceptual clarity, the rotated state is physically returned to the X-Y plane before being
    measured in the X or Y basis. Note that only Z-basis measurements are actually implemented,
    so even measurement in the X or Y basis still requires some pre-measurement gates.

    To summarize, we start in a frame where a state is prepared somewhere in a plane P perpendicular
    to the supplied axis of rotation. The rotation program is applied to this state depth many
    times, spinning the state in this plane P of rotation. There is an intermediate rotation
    which shifts the frame so that P coincides with the X-Y plane, the axis of rotation lies
    along the Z axis, and the initial state prepared would have coincided with the +X state.
    Finally, the state is measured in the X or Y basis, which itself involves some final gate
    applied before a measurement in the Z basis.

    Concretely, if the axis of rotation is the Z axis (0, 0) then the initial state is +X,
    the plane of rotation is simply the X-Y plane, and the final measurement provides either the
    X or Y expectation of the rotated state, which places the rotated state in the X-Y plane.

    :param rotation: the program specifying the gate whose angle of rotation we wish to estimate.
    :param axis: the axis of rotation corresponding to the rotation program, specified in
        radians as (theta, phi) in typical spherical coordinates, with standard Bloch sphere
        orientation (Z axis vertical with |0> at top, +X cross +Y = +Z using right-hand rule)
    :param depth: the number of times we apply the rotation in the experiment
    :param meas_dir: X or Y, specifying which operator to measure following the depth many
        rotations and after the plane of rotation has been brought to the X-Y plane.
    :param measurement_qubit:
    :param custom_prep: an optional preparation program to run before the standard preparation
    :return: a program implementing a single iteration of the RPE protocol in [RPE]
    """
    experiment = Program()

    # if a custom preparation is supplied, do that first
    if custom_prep is not None:
        experiment += custom_prep

    # prepare the measurement qubit in a state perpendicular to the axis of rotation
    experiment += prepare_state_about_axis(measurement_qubit, axis)

    # rotate the state depth many times
    for _ in range(depth):
        experiment.inst(rotation)

    # return state to X-Y plane
    experiment += RZ(-axis[1], measurement_qubit)
    experiment += RY(-axis[0], measurement_qubit)

    # measure either in either X or Y basis
    experiment += local_pauli_eig_meas(meas_dir, measurement_qubit)

    ro_bit = experiment.declare("ro", "BIT", 1)
    experiment.measure(measurement_qubit, ro_bit[0])

    return experiment


def generate_single_rpe_experiment(rotation: Program, axis: Tuple[float, float],
                                   num_depths: int = 5, measurement_qubit: int = None,
                                   custom_prep: Program = None) -> DataFrame:
    """
    Generate a dataframe containing all the experiments needed to perform robust phase estimation
    to estimate the angle of rotation about the given axis performed by the given rotation program.

    The algorithm is due to:

    [RPE]  Robust Calibration of a Universal Single-Qubit Gate-Set via Robust Phase Estimation
           Kimmel et al.,
           Phys. Rev. A 92, 062315 (2015)
           https://doi.org/10.1103/PhysRevA.92.062315
           https://arxiv.org/abs/1502.02677

    [RPE2] Experimental Demonstration of a Cheap and Accurate Phase Estimation
           Rudinger et al.,
           Phys. Rev. Lett. 118, 190502 (2017)
           https://doi.org/10.1103/PhysRevLett.118.190502
           https://arxiv.org/abs/1702.01763

    :param rotation: the program or gate whose angle of rotation is to be estimated. Note that
        this program will be run through forest_benchmarking.compilation.basic_compile().
    :param axis: the axis of rotation corresponding to the rotation program, specified in
        radians as (theta, phi) in typical spherical coordinates, with standard Bloch sphere
        orientation (Z axis vertical with |0> at top, +X cross +Y = +Z using right-hand rule)
    :param num_depths: the number of depths in the protocol described in [RPE]. A depth is the
        number of consecutive applications of the rotation in a single iteration. The maximum
        depth is 2**(num_depths-1)
    :param measurement_qubit: the qubit whose angle of rotation, as a result of the action of
        the rotation program, RPE will attempt to estimate. This is the only qubit measured.
    :param custom_prep: an optional prep program that will be run before the standard preparation
        of the measurement qubit. This could, for example, be supplied to initialize the control
        qubit of a CZ to |1>, with the target qubit as the measurement qubit.
    :return: a dataframe populated with all of experiments necessary for the RPE protocol in
        [RPE] with the necessary depth, measurement_direction, and program.
    """
    if isinstance(rotation, Gate):
        rotation = Program(rotation)

    if measurement_qubit is None:
        qubits = rotation.get_qubits()
        if len(qubits) == 1:
            measurement_qubit = qubits.pop()  # measure the relevant qubit
        else:
            raise ValueError("A measurement qubit must be specified.")

    def df_dict():
        for exponent in range(num_depths):
            depth = 2 ** exponent
            for meas_dir in ['X', 'Y']:
                yield {"Depth": depth,
                       "Meas_Direction": meas_dir,
                       "Experiment": generate_single_depth_rpe_experiment(rotation, axis, depth,
                                                                          meas_dir,
                                                                          measurement_qubit,
                                                                          custom_prep)}

    # TODO: Put dtypes on this DataFrame in the right way
    return DataFrame(df_dict())


def generate_rpe_experiments()


def get_additive_error_factor(M_j: float, max_additive_error: float) -> float:
    """
    Calculate the factor in Equation V.17 of [RPE].

    This factor multiplies the number of trials at the jth iteration in order to maintain
    Heisenberg scaling with the same variance upper bound as if there were no additive error
    present. This holds as long as the actual max_additive_error in the procedure is no more than
    1/sqrt(8) ~=~ .354 error present in the procedure

    :param M_j: the number of shots in the jth iteration of RPE
    :param max_additive_error: the assumed maximum of the additive errors you hope to adjust for
    :return: A factor that multiplied by M_j yields a number of shots preserving Heisenberg Scaling
    """
    return np.log(.5 * (1 - np.sqrt(8) * max_additive_error) ** (1 / M_j)) \
        / np.log(1 - .5 * (1 - np.sqrt(8) * max_additive_error) ** 2)


def num_trials(depth, max_depth, alpha, beta, multiplicative_factor: float = 1.0,
               additive_error: float = None) -> int:
    """
    Calculate the optimal number of shots per experiment with a given depth.

    The calculation is given by equations V.11 and V.17 in [RPE]. A non-default multiplicative
    factor breaks the optimality guarantee.

    :param depth: the depth of the experiment whose number of trials is calculated
    :param max_depth: maximum depth of the experiments
    :param alpha: a hyper-parameter in equation V.11 of [RPE], suggested to be 5/2
    :param beta: a hyper-parameter in equation V.11 of [RPE], suggested to be 1/2
    :param multiplicative_factor: extra add-hoc factor that multiplies the optimal number of shots
    :param additive_error: estimate of the max additive error in the experiment, eq. V.15 of [RPE]
    :return: Mj, the number of shots for experiment with depth 2**(j-1) in iteration j of RPE
    """
    j = np.log2(depth) + 1
    K = np.log2(max_depth) + 1
    Mj = (alpha * (K - j) + beta)
    if additive_error:
        multiplicative_factor *= get_additive_error_factor(Mj, additive_error)
    return int(np.ceil(Mj * multiplicative_factor))


def acquire_rpe_data(experiments: DataFrame, qc: QuantumComputer,
                     multiplicative_factor: float = 1.0, additive_error: float = None,
                     results_label="Results") -> DataFrame:
    """
    Run each experiment in the experiments data frame a number of times which is specified by
    num_trials().

    The experiments df is copied, and raw shot outputs are stored in a column labeled by
    results_label, which defaults to "Results". The number of shots run at each depth can be
    modified indirectly by adjusting multiplicative_factor and additive_error.

    :param experiments: dataframe containing experiments, generated by generate_rpe_experiments()
    :param qc: a quantum computer, e.g. QVM or QPU, that runs the experiments
    :param multiplicative_factor: ad-hoc factor to multiply the number of shots per iteration. See
        num_trials() which computes the optimal number of shots per iteration.
    :param additive_error: estimate of the max additive error in the experiment, see num_trials()
    :param results_label: label for the column of the returned df to be populated with results
    :return: A copy of the experiments data frame with the raw shot results in a new column.
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


def _p_max(M_j: int) -> float:
    """
    Calculate an upper bound on the probability of error in the estimate on the jth iteration.
    Equation V.6 in [RPE]

    :param M_j: The number of shots for the jth iteration of RPE
    :return: p_max(M_j), an upper bound on the probability of error on the estimate k_j * Angle
    """
    return (1 / np.sqrt(2 * pi * M_j)) * (2 ** -M_j)


def _xci(h: int) -> float:
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
        single_depth = experiments.groupby(["Depth"]).get_group(2 ** (j - 1)).set_index(
            'Meas_Direction')
        M_j = len(single_depth.loc['X', results_label])
        M_js += [M_j]

    # note that M_js is 0 indexed but 1 <= j <= K, so M_j = M_js[j-1]
    return (1 - _p_max(M_js[K - 1])) * _xci(K + 1) ** 2 + sum(
        [_xci(i + 1) ** 2 * _p_max(M_j) for i, M_j in enumerate(M_js)])


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
        N = len(group[group['Meas_Direction'] == 'X'][results_label].values[0])

        p_x = group[group['Meas_Direction'] == 'X'][results_label].values[0].mean()
        p_y = group[group['Meas_Direction'] == 'Y'][results_label].values[0].mean()
        # standard deviation of the mean of the probabilities
        p_x_std = group[group['Meas_Direction'] == 'X'][results_label].values[0].std() / np.sqrt(N)
        p_y_std = group[group['Meas_Direction'] == 'Y'][results_label].values[0].std() / np.sqrt(N)
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
    :param bloch_data: if provided, list is mutated to store the radius and angle of each iteration
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


def plot_rpe_iterations(experiments: DataFrame, expected_positions: List = None) -> plt.Axes:
    """
    Creates a polar plot of the estimated location of the state in the plane perpendicular to the
    axis of rotation for each iteration of RPE.

    :param experiments: a dataframe with RPE results populated by a call to acquire_rpe_data
    :param expected_positions: a list of expected (radius, angle) pairs for each iteration
    :return: a matplotlib subplot visualizing each iteration of the RPE experiment
    """
    positions = []
    xs, ys, x_stds, y_stds = find_expectation_values(experiments)
    # mutate positions, do not need the actual estimate
    robust_phase_estimate(xs, ys, x_stds, y_stds, positions)
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
