from typing import Tuple, List, Sequence, Union

import numpy as np
from numpy import pi
from pandas import DataFrame, Series
import pandas

from pyquil.quil import Program, merge_programs
from pyquil.gates import RZ, RX
from pyquil.quilbase import Gate
from pyquil.api import QuantumComputer
from forest.benchmarking.compilation import basic_compile
from forest.benchmarking.utils import transform_bit_moments_to_pauli, local_pauli_eig_meas, \
    determine_simultaneous_grouping, prepare_state_on_bloch_sphere, standard_basis_to_bloch_vector

import matplotlib.pyplot as plt


def eigenvector_superposition_prep(qubit, e1: np.ndarray, e2: np.ndarray):
    """
    Provides a program which, from the |0> state, prepares the equal superposition of the given
    orthogonal vectors.

    Note that for a single qubit Hilbert space, specifying a single eigenvector e1 of an operator A
    necessarily determines the other eigenvector e2 since these vectors are necessarily
    orthogonal and span the Hilbert space. However, each eigenvector in isolation is only defined
    up to some global phase. Hence, for the superposition to be un-ambiguously determined,
    we require both eigenvectors be supplied, which specifies a specific relative phase.

    :param qubit: the qubit to prepare in the given state.
    :param e1: a vector [alpha, beta]
    :param e2: a vector [alpha2, beta2] specifying a vector orthogonal to e1
    :return: a program preparing the state (e1 + e2)/sqrt(2)
    """
    superposition = (e1 + e2) / np.sqrt(2)
    theta, phi = standard_basis_to_bloch_vector(superposition)

    return prepare_state_on_bloch_sphere(qubit, theta, phi)


def orthogonal_to_axis_prep(qubit: int, axis: Tuple[float, float]) -> Program:
    """
    Generates a program that prepares a state perpendicular to the given (theta, phi) axis on
    the Bloch sphere.

    In the context of an RPE experiment, the supplied axis is the axis of rotation of the gate
    whose magnitude of rotation the experimenter is trying to estimate. Equivalently, axis is the
    plus eigenvector of the rotation. The prepared state is a point on the sphere whose radial
    line is perpendicular to the supplied axis; the state is pi radians in the theta direction
    from axis. Equivalently, the final state is simply some equal magnitude (perhaps with a
    relative phase) superposition of the two eigenstates operator which rotates about the given
    rotation axis.

    For example, the axis (0, 0) corresponds to an RPE experiment estimating the angle parameter
    of the rotation RZ(angle). The initial state of this experiment would be the plus one
    eigenstate of X, or |+> = RY(pi/2) |0> since this state is perpendicular to the axis of
    rotation of RZ. For rotation about an arbitrary axis=(theta, phi), the initial state is
    equivalently RZ(phi)RY(theta + pi/2)|0>

    :param qubit: the qubit whose state is being prepared
    :param axis: axis of rotation specified as (theta, phi) in typical spherical coordinates.
    :return: A preparation program that prepares the qubit in a state perpendicular to axis.
    """
    return prepare_state_on_bloch_sphere(qubit, axis[0] + pi / 2, axis[1])


def orthogonal_to_axis_measures(qubit: int, axis: Tuple[float, float]) -> Program:
    """
    Return the measure programs for the RPE experiment on a rotation about axis.

    This method simply treats the state at axis as if it is the +Z state and returns
    programs
    that measure along the induced "X" and "Y" axes, where we follow the convention of
    orthogonal_to_axis_prep that the +X state is pi/2 radians from axis in the theta direction.

    :param qubit:
    :param axis:
    :return:
    """
    # the "X" measurement is simply in the direction of the initial preparation
    x_measurment = orthogonal_to_axis_prep(qubit, axis).dagger()

    # for "Y" measurement, first undo phi rotation
    y_measurement = Program(RZ(-axis[1], qubit))
    y_measurement += RX(pi / 2, qubit)  # then rotate +Y up to +Z

    return x_measurment, y_measurement


def prep_and_measures_for_cz(q1: int, q2: int, e1: np.ndarray, e2: np.ndarray):
    """

    We consider the qubits to be ordered |q1 q2> and use the standard representation, for example
        |0 1>  -->  [[0, 1, 0, 0]].T

    :param q1:
    :param q2:
    :param e1:
    :param e2:
    :return:
    """
    # standardize as single row
    e1 = np.asarray(e1).reshape(4,)
    e2 = np.asarray(e2).reshape(4,)

    if e1[0] == e2[3] == 1 or e1[1] == e2[2] == 1:
        raise ValueError("Estimation of the relative phase between these particular eigenvectors "
                         "is not supported, as the gates required for state prep and measurement "
                         "are not local.")

    sup = e1 + e2

    alpha1 = sup[0] + sup[1]
    beta1 = sup[2] + sup[3]
    alpha2 = sup[0] + sup[2]
    beta2 = sup[1] + sup[3]

    q1_state = np.array([alpha1, beta1])
    q2_state = np.array([alpha2, beta2])
    q1_state = q1_state / np.linalg.norm(q1_state)
    q2_state = q2_state / np.linalg.norm(q2_state)

    prep1 = prepare_state_on_bloch_sphere(q1, standard_basis_to_bloch_vector(q1_state))
    prep2 = prepare_state_on_bloch_sphere(q2, standard_basis_to_bloch_vector(q2_state))
    prep = prep1 + prep2

    x_meas = prep.dagger()
    y_meas =


def generate_rpe_experiment(state_prep: Program, rotation: Program,
                            measurement_preps: Tuple[Program, Program],
                            measure_qubits: Sequence[int] = None, num_depths: int = 5) -> DataFrame:
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

    :param state_prep:
    :param rotation: the program or gate whose angle of rotation is to be estimated. Note that
        this program will be run through forest_benchmarking.compilation.basic_compile().
    :param measurement_preps:
    :param num_depths: the number of depths in the protocol described in [RPE]. A depth is the
        number of consecutive applications of the rotation in a single iteration. The maximum
        depth is 2**(num_depths-1)
    :param measure_qubits: the qubits whose angle of rotation, as a result of the action of
        the rotation program, RPE will attempt to estimate. These are the only qubits measured.
    :return: a dataframe populated with all of programs necessary for the RPE protocol in
        [RPE] with the necessary depth, measurement_direction, and program.
    """
    if isinstance(rotation, Gate):
        rotation = Program(rotation)

    prep_qubits = state_prep.get_qubits()
    rotation_qubits = rotation.get_qubits()
    qubits = prep_qubits.union(rotation_qubits)

    # assume that both measurements make use of the same qubits
    measurement_qubits = measurement_preps[0].get_qubits()
    qubits = qubits.union(measurement_qubits)

    if measure_qubits is None:
        measure_qubits = measurement_qubits

    qubits = qubits.union(measure_qubits)

    def df_dict():
        for exponent in range(num_depths):
            depth = 2 ** exponent
            for meas_dir in [0, 1]:
                depth_rotations = sum([rotation for _ in range(depth)], Program())
                yield {"Qubits": qubits,
                       "Measure Qubits": measure_qubits,
                       "Depth": depth,
                       "Meas_Direction": meas_dir,
                       "Program": sum([state_prep, depth_rotations, measurement_preps[meas_dir]],
                                      Program())}

    # TODO: Put dtypes on this DataFrame in the right way
    return DataFrame(df_dict())


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
    Calculate the optimal number of shots per program with a given depth.

    The calculation is given by equations V.11 and V.17 in [RPE]. A non-default multiplicative
    factor breaks the optimality guarantee.

    :param depth: the depth of the program whose number of trials is calculated
    :param max_depth: maximum depth of programs in the experiment
    :param alpha: a hyper-parameter in equation V.11 of [RPE], suggested to be 5/2
    :param beta: a hyper-parameter in equation V.11 of [RPE], suggested to be 1/2
    :param multiplicative_factor: extra add-hoc factor that multiplies the optimal number of shots
    :param additive_error: estimate of the max additive error in the experiment, eq. V.15 of [RPE]
    :return: Mj, the number of shots for program with depth 2**(j-1) in iteration j of RPE
    """
    j = np.log2(depth) + 1
    K = np.log2(max_depth) + 1
    Mj = (alpha * (K - j) + beta)
    if additive_error:
        multiplicative_factor *= get_additive_error_factor(Mj, additive_error)
    return int(np.ceil(Mj * multiplicative_factor))


def _run_rpe_program(qc: QuantumComputer, prog: Program, measure_qubits: Sequence[Sequence[int]],
                     num_shots: int) -> np.ndarray:
    """
    Simple helper to run a program with appropriate number of shots and return result.

    Note that the program is first compiled with basic_compile.

    :param qc: quantum computer to run program on
    :param prog: program to run
    :param num_shots: number of shots to repeat the program
    :return: the results of the program
    """
    meas_qubits = [qubit for qubits in measure_qubits for qubit in qubits]
    ro_bit = prog.declare("ro", "BIT", len(meas_qubits))
    for idx, q in enumerate(meas_qubits):
        prog.measure(q, ro_bit[idx])
    prog.wrap_in_numshots_loop(num_shots)
    executable = qc.compiler.native_quil_to_executable(basic_compile(prog))
    return qc.run(executable)


def run_single_rpe_experiment(experiment: DataFrame, qc: QuantumComputer,
                              multiplicative_factor: float = 1.0, additive_error: float = None,
                              results_label="Results") -> DataFrame:
    """
    Run each program in the experiment data frame a number of times which is specified by
    num_trials().

    The experiment df is copied, and raw shot outputs are stored in a column labeled by
    results_label, which defaults to "Results". The number of shots run at each depth can be
    modified indirectly by adjusting multiplicative_factor and additive_error.

    :param experiment: dataframe generated by generate_rpe_experiment()
    :param qc: a quantum computer, e.g. QVM or QPU, that runs each program in the experiment
    :param multiplicative_factor: ad-hoc factor to multiply the number of shots per iteration. See
        num_trials() which computes the optimal number of shots per iteration.
    :param additive_error: estimate of the max additive error in the experiment, see num_trials()
    :param results_label: label for the column of the returned df to be populated with results
    :return: A copy of the experiment data frame with the raw shot results in a new column.
    """
    alpha = 5 / 2  # should be > 2
    beta = 1 / 2  # should be > 0
    max_depth = max(experiment["Depth"].values)
    results = [_run_rpe_program(qc, program,
                                num_trials(depth, max_depth, alpha, beta, multiplicative_factor,
                                           additive_error))
               for (depth, program) in zip(experiment["Depth"].values,
                                           experiment["Program"].values)]
    exp_with_results = experiment.copy()
    exp_with_results[results_label] = Series(results)
    return exp_with_results


def acquire_rpe_data(qc: QuantumComputer, experiments: Union[DataFrame, Sequence[DataFrame]],
                     multiplicative_factor: float = 1.0, additive_error: float = None,
                     grouping: Sequence[Tuple[int]] = None,
                     results_label="Results") -> DataFrame:
    """
    Run each experiment in the sequence of experiments.

    Each individual experiment df is copied, and raw shot outputs are stored in a column labeled by
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

    if isinstance(experiments, DataFrame):
        return run_single_rpe_experiment(qc, experiments, multiplicative_factor, additive_error,
                                         results_label)

    if grouping is None:
        grouping = determine_simultaneous_grouping(experiments)

    results = []
    for group in grouping:
        grouped_expts = [experiments[idx] for idx in group]

        depths = [expt["Depth"] for expt in grouped_expts]
        for d1, d2 in zip(depths[:-1], depths[1:]):
            assert d1.equals(d2), "Depths must be equivalent to run experiments simultaneously."

        programs_df = pandas.concat([expt["Program"] for expt in grouped_expts], axis=1)
        programs = programs_df.apply(merge_programs, axis=1)

        measure_qubits = [list(expt["Measure Qubits"].values[0]) for expt in grouped_expts]

        max_depth = max(depths[0])
        alpha = 5 / 2  # should be > 2
        beta = 1 / 2  # should be > 0

        results = [_run_rpe_program(qc, program, measure_qubits,
                                    num_trials(depth, max_depth, alpha, beta, multiplicative_factor,
                                               additive_error))
                   for (depth, program) in zip(depths[0], programs.values)]

    experiments = experiments.copy()
    experiments[results_label] = Series(results)
    return experiments


# def separate_rpe_results_by_id(df: DataFrame) -> Sequence[DataFrame]:
#     """
#     Takes a single DataFrame with a number of different RPE experiments which were run
#     simultaneously and returns a list of separate data frames for each experiment.
#
#     :param df: a data frame output by a run of acquire_rpe_data on a list of experiments.
#     :return: a list of separate experiment data frames with results.
#     """
#     ids = df["Exp. ID"].unique()
#
#     experiments = []
#     for id in ids:
#         single_experiment = df.loc[df["Exp. ID"] == id]
#         experiments.append(single_experiment)
#
#
#     return experiments


# def acquire_rpe_data(experiments: DataFrame, qc: QuantumComputer,
#                      multiplicative_factor: float = 1.0, additive_error: float = None,
#                      results_label="Results") -> DataFrame:
#     """
#     Run each experiment in the experiments data frame a number of times which is specified by
#     num_trials().
#
#     The experiments df is copied, and raw shot outputs are stored in a column labeled by
#     results_label, which defaults to "Results". The number of shots run at each depth can be
#     modified indirectly by adjusting multiplicative_factor and additive_error.
#
#     :param experiments: dataframe containing experiments, generated by generate_rpe_experiments()
#     :param qc: a quantum computer, e.g. QVM or QPU, that runs the experiments
#     :param multiplicative_factor: ad-hoc factor to multiply the number of shots per iteration. See
#         num_trials() which computes the optimal number of shots per iteration.
#     :param additive_error: estimate of the max additive error in the experiment, see num_trials()
#     :param results_label: label for the column of the returned df to be populated with results
#     :return: A copy of the experiments data frame with the raw shot results in a new column.
#     """
#
#     def run(qc: QuantumComputer, exp: Program, n_trials: int) -> np.ndarray:
#         exp.wrap_in_numshots_loop(n_trials)
#         executable = qc.compiler.native_quil_to_executable(basic_compile(exp))
#         return qc.run(executable)
#
#     alpha = 5 / 2  # should be > 2
#     beta = 1 / 2  # should be > 0
#     max_depth = max(experiments["Depth"].values)
#     results = [run(qc, experiment,
#                    num_trials(depth, max_depth, alpha, beta, multiplicative_factor, additive_error))
#                for (depth, experiment) in zip(experiments["Depth"].values, experiments["Experiment"].values)]
#     experiments = experiments.copy()
#     experiments[results_label] = Series(results)
#     return experiments


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


def get_variance_upper_bound(experiment: DataFrame, results_label='Results') -> float:
    """
    Equation V.9 in [RPE]

    :param experiment: a dataframe with RPE results. Importantly the bound follows from the number
        of shots at each iteration of the experiment, so the data frame needs to be populated with
        the desired number-of-shots-many results.
    :param results_label: label for the column with results from which the variance is estimated
    :return: An upper bound of the variance of the angle estimate corresponding to the input
        experiments.
    """
    max_depth = max(experiment["Depth"].values)
    K = np.log2(max_depth).astype(int) + 1

    M_js = []
    # 1 <= j <= K, where j is the one-indexed iteration number
    for j in range(1, K + 1):
        single_depth = experiment.groupby(["Depth"]).get_group(2 ** (j - 1)).set_index(
            'Meas_Direction')
        M_j = len(single_depth.loc['X', results_label])
        M_js += [M_j]

    # note that M_js is 0 indexed but 1 <= j <= K, so M_j = M_js[j-1]
    return (1 - _p_max(M_js[K - 1])) * _xci(K + 1) ** 2 + sum(
        [_xci(i + 1) ** 2 * _p_max(M_j) for i, M_j in enumerate(M_js)])


def find_expectation_values(experiment: DataFrame, results_label='Results') -> \
        Tuple[List, List, List, List]:
    """
    Calculate expectation values and standard deviation of the mean for each depth and
    experiment type.

    :param experiment: a dataframe with RPE results populated by a call to acquire_rpe_data
    :param results_label: label for the column with results from which the variance is estimated
    """
    xs = []
    ys = []
    x_stds = []
    y_stds = []

    for depth, group in experiment.groupby(["Depth"]):
        N = len(group[group['Meas_Direction'] == 'X'][results_label].values[0])

        p_x = group[group['Meas_Direction'] == 'X'][results_label].values[0].mean()
        p_y = group[group['Meas_Direction'] == 'Y'][results_label].values[0].mean()
        # standard deviation of the mean of the probabilities
        p_x_std = group[group['Meas_Direction'] == 'X'][results_label].values[0].std() / np.sqrt(N)
        p_y_std = group[group['Meas_Direction'] == 'Y'][results_label].values[0].std() / np.sqrt(N)
        # convert probabilities to expectation values of X and Y
        exp_x, var_x = transform_bit_moments_to_pauli(1 - p_x, p_x_std ** 2)
        exp_y, var_y = transform_bit_moments_to_pauli(1 - p_y, p_y_std ** 2)
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
    :param x_stds: standard deviation of the mean for 'xs'
    :param y_stds: standard deviation of the mean for 'ys'
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
