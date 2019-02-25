from typing import Tuple, List, Sequence, Union

import numpy as np
from numpy import pi
from pandas import DataFrame, Series
import pandas

from pyquil.quil import Program, merge_programs, DefGate, Pragma
from pyquil.quilbase import Gate
from pyquil.gates import RX
from pyquil.api import QuantumComputer
from forest.benchmarking.compilation import basic_compile
from forest.benchmarking.utils import transform_bit_moments_to_pauli, local_pauli_eig_prep, \
    local_pauli_eig_meas, determine_simultaneous_grouping, bloch_vector_to_standard_basis


import matplotlib.pyplot as plt


def bloch_rotation_to_eigenvectors(theta: float, phi: float) -> Sequence[np.ndarray]:
    """
    Provides convenient conversion from a 1q rotation about some Bloch vector to the two
    eigenvectors of rotation that lay along the rotation axis.

    The standard right-hand convention would dictate that the Bloch vector is such that the
    rotation about this vector is counter-clockwise for a clock facing in the direction of
    the vector. In this case, the order of returned eigenvectors when passed through
    get_change_of_basis_from_eigvecs to an RPE experiment will result in a positive phase.

    :param theta: azimuthal angle given in radians
    :param phi:  polar angle given in radians
    :return: the eigenvectors of the rotation, listed in order consistent with the right-hand-rule
        and the conventions taken by get_change_of_basis_from_eigvecs and the RPE experiment
    """
    eig1 = np.array([bloch_vector_to_standard_basis(theta, phi)]).T
    eig2 = np.array([bloch_vector_to_standard_basis(pi - theta, pi + phi)]).T
    return eig1, eig2


def _is_pos_pow_two(x: int) -> bool:
    """
    Simple check that an integer is a positive power of two.
    :param x: number to check
    :return: whether x is a positive power of two
    """
    if x <= 0:
        return False
    while (x & 1) == 0:
        x = x >> 1
    return x == 1


def get_change_of_basis_from_eigvecs(eigenvectors: Sequence[np.ndarray]):
    """
    Generates a unitary matrix that sends each computational basis state to the corresponding
    eigenvector.

    :param eigenvectors: a sequence of dim-many length-dim eigenvectors. In the context of an RPE
        experiment, these should be eigenvectors of the rotation. For convenience, the function
        bloch_rotation_to_eigenvectors() will provide the appropriate list of eigenvectors for a
        rotation about a particular axis in the 1q Bloch sphere. Note that RPE experiment follows
        the convention of the right-hand rule. E.g. for a 1q rotation lets say eigenvectors {e1, e2}
        have eigenvalues with phase {p1, p2}. If the relative phase {p1-p2} is positive,
        then the rotation happens about e1 in the counter-clockwise direction for a clock facing
        the e1 direction.
    :return:
    """
    assert len(eigenvectors) > 1 and _is_pos_pow_two(len(eigenvectors)), \
        "Specification of all dim-many eigenvectors is required."

    # standardize the possible list, 1d or 2d-row-vector ndarray inputs to column vectors
    eigs = []
    for eig in eigenvectors:
        eig = np.asarray(eig)
        shape = eig.shape
        if len(shape) == 1:
            eig = eig[np.newaxis]
        eigs.append(eig.reshape(max(shape),1))

    dim = eigs[0].shape[0]
    id_dim = np.eye(dim)
    comp_basis = [row[np.newaxis] for row in id_dim]  # get computational basis in row vectors

    # this unitary will take the computational basis to a basis where our rotation is diagonal
    basis_change = sum([np.kron(ev, cb) for ev, cb in zip(eigs, comp_basis)])

    return basis_change


def generate_rpe_experiment(rotation: Program, change_of_basis: Union[np.ndarray, Program],
                            measure_qubits: Sequence[int] = None, num_depths: int = 6,
                            post_select_state: Sequence[int] = None) -> DataFrame:
    """
    Generate a dataframe containing all the experiments needed to perform robust phase estimation
    to estimate the angle of rotation about the given axis performed by the given rotation program.

    The single qubit algorithm is due to:

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
    :param change_of_basis:
    :param num_depths: the number of depths in the protocol described in [RPE]. A depth is the
        number of consecutive applications of the rotation in a single iteration. The maximum
        depth is 2**(num_depths-1)
    :param measure_qubits: the qubits whose angle of rotation, as a result of the action of
        the rotation program, RPE will attempt to estimate. These are the only qubits measured.
        In the case of a two qubit experiment this should include both qubits.
    :param post_select_state:
    :return: a dataframe populated with all of programs necessary for the RPE protocol in
        [RPE] with the necessary depth, measurement_direction, and program.
    """
    if isinstance(rotation, Gate):
        rotation = Program(rotation)

    if isinstance(change_of_basis, Gate):
        change_of_basis = Program(change_of_basis)

    rotation_qubits = rotation.get_qubits()

    if measure_qubits is None:
        measure_qubits = rotation_qubits  # assume interest in qubits being rotated.
        # If you wish to measure multiple single qubit phases e.g. induced by cross-talk from the
        # operation of a gate on other qubits, consider creating multiple "dummy" experiments
        # that implement the identity and measure the qubits of interest. Subsequently run these
        # "dummies" simultaneously with an experiment whose rotation is the cross-talky program.

    qubits = rotation_qubits.union(measure_qubits)

    def df_dict():
        for exponent in range(num_depths):
            depth = 2 ** exponent
            for meas_dir in ["X", "Y"]:
                if len(measure_qubits) > 1:
                    # this is a >1q RPE experiment; the qubit being rotated need be specified
                    for meas_qubit in measure_qubits:
                        yield {"Qubits": qubits,
                               "Rotation": rotation,
                               "Depth": depth,
                               "Measure Direction": meas_dir,
                               "Measure Qubits": list(measure_qubits),
                               "Non-Z-Basis Meas Qubit": meas_qubit,
                               "Change of Basis": change_of_basis,
                               }
                else:
                    # standard 1q experiment, no need for Non-Z-Basis Meas Qubit
                    yield {"Qubits": qubits,
                           "Rotation": rotation,
                           "Depth": depth,
                           "Measure Direction": meas_dir,
                           "Measure Qubits": list(measure_qubits),
                           "Change of Basis": change_of_basis,
                           }
    # TODO: Put dtypes on this DataFrame in the right way
    expt = DataFrame(df_dict())

    if post_select_state is not None:
        expt["Post Select State"] = [post_select_state for _ in range(expt.shape[0])]

    # change_of_basis is already specified as program, so add composed program column
    if isinstance(change_of_basis, Program):
        expt["Program"] = expt.apply(_make_prog_from_df, axis=1)

    return expt

#     """
#     Generates a program that prepares a state perpendicular to the given (theta, phi) axis on
#     the Bloch sphere.
#
#     In the context of an RPE experiment, the supplied axis is the axis of rotation of the gate
#     whose magnitude of rotation the experimenter is trying to estimate. Equivalently, axis is the
#     plus eigenvector of the rotation. The prepared state is a point on the sphere whose radial
#     line is perpendicular to the supplied axis; the state is pi radians in the theta direction
#     from axis. Equivalently, the final state is simply some equal magnitude (perhaps with a
#     relative phase) superposition of the two eigenstates operator which rotates about the given
#     rotation axis.
#
#     For example, the axis (0, 0) corresponds to an RPE experiment estimating the angle parameter
#     of the rotation RZ(angle). The initial state of this experiment would be the plus one
#     eigenstate of X, or |+> = RY(pi/2) |0> since this state is perpendicular to the axis of
#     rotation of RZ. For rotation about an arbitrary axis=(theta, phi), the initial state is
#     equivalently RZ(phi)RY(theta + pi/2)|0>


def add_program_to_df(qc: QuantumComputer, experiment: DataFrame):
    expt = experiment.copy()
    expt["Program"] = expt.apply(_make_prog_from_df, axis=1,  args=(qc,))
    return expt


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


def _run_rpe_program(qc: QuantumComputer, program: Program, measure_qubits: Sequence[Sequence[int]],
                     num_shots: int) -> np.ndarray:
    """
    Simple helper to run a program with appropriate number of shots and return result.

    Note that the program is first compiled with basic_compile.

    :param qc: quantum computer to run program on
    :param prog: program to run
    :param num_shots: number of shots to repeat the program
    :return: the results of the program
    """
    prog = Program() + program  # make a copy of program
    meas_qubits = [qubit for qubits in measure_qubits for qubit in qubits]
    ro_bit = prog.declare("ro", "BIT", len(meas_qubits))
    for idx, q in enumerate(meas_qubits):
        prog.measure(q, ro_bit[idx])
    prog.wrap_in_numshots_loop(num_shots)
    executable = qc.compiler.native_quil_to_executable(basic_compile(prog))
    return qc.run(executable)


def run_single_rpe_experiment(qc: QuantumComputer, experiment: DataFrame,
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
    expt = experiment.copy()
    if "Program" not in expt.columns.values:
        # pass the qc and each row from dataframe into helper to make programs
        expt["Program"] = expt.apply(_make_prog_from_df, axis=1, args=(qc,))

    alpha = 5 / 2  # should be > 2
    beta = 1 / 2  # should be > 0
    max_depth = max(experiment["Depth"].values)
    measure_qubits = experiment["Measure Qubits"].values[0]
    results = [_run_rpe_program(qc, program, [measure_qubits],
                                num_trials(depth, max_depth, alpha, beta, multiplicative_factor,
                                           additive_error))
               for (depth, program) in zip(expt["Depth"].values, expt["Program"].values)]

    expt[results_label] = Series(results)
    return expt


def change_of_basis_matrix_to_quil(qc: QuantumComputer, qubits: Sequence[int],
                                   change_of_basis: np.ndarray) -> Program:
    """
    Helper to return a native quil program for the given qc to implement the change_of_basis matrix.

    :param qc: Quantum Computer that will need to use the change of basis
    :param qubits: the qubits the program should act on
    :param change_of_basis: a unitary matrix acting on len(qubits)
    :return: a native quil program that implements change_of_basis on the qubits of qc.
    """
    prog = Program()
    # ensure the program is compiled onto the proper qubits
    prog += Pragma('INITIAL_REWIRING', ['"NAIVE"'])
    g_definition = DefGate("COB", change_of_basis)
    # get the gate constructor
    COB = g_definition.get_constructor()
    # add definition to program
    prog += g_definition
    # add gate to program
    prog += COB(*qubits)
    # compile to native quil
    nquil = qc.compiler.quil_to_native_quil(prog)

    # strip the program to only what we need, i.e. the gates themselves.
    only_gates = Program([inst for inst in nquil if isinstance(inst, Gate)])

    return only_gates


def _make_prog_from_df(row: Series, qc: QuantumComputer = None):
    """

    :param qc:
    :param qubits:
    :param measure_qubits:
    :param depth:
    :param meas_dir:
    :param meas_qubit:
    :param change_of_basis:
    :param rotation:
    :return:
    """
    cob = row["Change of Basis"]
    rotation = row["Rotation"]
    meas_qubits = row["Measure Qubits"]
    if len(meas_qubits) > 1:
        meas_qubit = row["Non-Z-Basis Meas Qubit"]
    else:
        meas_qubit = meas_qubits[0]
    post_select_state = None
    if "Post Select State" in row.index:
        post_select_state = row["Post Select State"]

    if not isinstance(cob, Program) and qc is not None:
        cob = change_of_basis_matrix_to_quil(qc, meas_qubits, cob)

    prog = Program()

    if post_select_state is None:
        # start in equal superposition of basis states, i.e. put each qubit in plus state
        prog = sum([local_pauli_eig_prep('X', q) for q in meas_qubits], Program())
    else:
        # only start the non-z-basis measurement qubit in the superposition
        prog += local_pauli_eig_prep('X', meas_qubit)
        # put all other qubits in the post-selection-state
        prog += sum([RX(pi, q) for idx, q in enumerate(meas_qubits) if q != meas_qubit and
                     post_select_state[idx] == 1], Program())
    # using change_of_basis, transform to equal superposition of rotation eigenvectors
    prog += cob
    # perform the rotation depth many times
    prog += sum([rotation for _ in range(row["Depth"])], Program())
    # return to computational basis before measurements
    prog += cob.dagger()
    # prepare the meas_qubit in the appropriate meas_direction
    prog += local_pauli_eig_meas(row["Measure Direction"], meas_qubit)
    return prog


def acquire_rpe_data(qc: QuantumComputer, experiments: Union[DataFrame, Sequence[DataFrame]],
                     multiplicative_factor: float = 1.0, additive_error: float = None,
                     grouping: Sequence[Tuple[int]] = None, results_label="Results") \
        -> Union[DataFrame, Sequence[DataFrame]]:
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

    expts = [expt.copy() for expt in experiments]

    # check that each experiment has programs generated for this qc; generate them if not
    for expt in expts:
        if "Program" not in expt.columns.values:
            # pass the qc and each row from dataframe into helper to make programs
            expt["Program"] = expt.apply(_make_prog_from_df, axis=1, args=(qc,))

    # try to group experiments to run simultaneously
    if grouping is None:
        grouping = determine_simultaneous_grouping(expts, "Depth")

    # For each group, merge the individual programs, run each merged program, and separately
    # record results. Record the grouping via a list of indices of experiments in the input list.
    for group in grouping:
        grouped_expts = [expts[idx] for idx in group]

        programs_df = pandas.concat([expt["Program"] for expt in grouped_expts], axis=1)
        merged = programs_df.apply(merge_programs, axis=1)

        measure_qubits = [expt["Measure Qubits"].values[0] for expt in grouped_expts]

        depths = grouped_expts[0]["Depth"].values
        max_depth = max(depths)
        alpha = 5 / 2  # should be > 2
        beta = 1 / 2  # should be > 0

        results = [_run_rpe_program(qc, program, measure_qubits,
                                    num_trials(depth, max_depth, alpha, beta, multiplicative_factor,
                                               additive_error))
                   for (depth, program) in zip(depths, merged.values)]
        offset = 0
        for idx, meas_qs in enumerate(measure_qubits):
            expt = grouped_expts[idx]
            expt[results_label] = [row[:, offset: offset + len(meas_qs)] for row in results]
            offset += len(meas_qs)
            expt["Simultaneous Group"] = [group for _ in range(expt.shape[0])]

    return expts


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


def get_variance_upper_bound(experiment: DataFrame, multiplicative_factor: float = 1.0,
                             additive_error: float = None, results_label='Results') -> float:
    """
    Equation V.9 in [RPE]

    :param experiment: a dataframe with RPE results. Importantly the bound follows from the number
        of shots at each iteration of the experiment, so the data frame needs to be populated with
        the desired number-of-shots-many results.
    :param multiplicative_factor: ad-hoc factor to multiply the number of shots per iteration. See
        num_trials() which computes the optimal number of shots per iteration.
    :param additive_error: estimate of the max additive error in the experiment, see num_trials()
    :param results_label: label for the column with results from which the variance is estimated
    :return: An upper bound of the variance of the angle estimate corresponding to the input
        experiments.
    """
    max_depth = max(experiment["Depth"].values)
    K = np.log2(max_depth).astype(int) + 1

    M_js = []
    # 1 <= j <= K, where j is the one-indexed iteration number
    for j in range(1, K + 1):
        depth = 2 ** (j - 1)
        single_depth = experiment.groupby(["Depth"]).get_group(depth).set_index(
            'Measure Direction')

        if results_label in experiment.columns.values:
            # use the actual nunber of shots taken
            M_j = len(single_depth.loc['X', results_label])
        else:
            # default prediction for standard parameters
            M_j = num_trials(depth, max_depth, 5/2, 1/2, multiplicative_factor, additive_error)

        M_js += [M_j]

    # note that M_js is 0 indexed but 1 <= j <= K, so M_j = M_js[j-1]
    return (1 - _p_max(M_js[K - 1])) * _xci(K + 1) ** 2 + sum(
        [_xci(i + 1) ** 2 * _p_max(M_j) for i, M_j in enumerate(M_js)])


def get_moments(experiment: DataFrame, results_label='Results') -> Tuple[List, List, List, List]:
    """
    Calculate expectation values and standard deviation for each row of the experiment.

    :param experiment: a dataframe with RPE results populated by a call to acquire_rpe_data
    :param results_label: label for the column with results from which the moments are estimated
    """
    xs = []
    ys = []
    x_stds = []
    y_stds = []

    for depth, group in experiment.groupby(["Depth"]):
        N = len(group[group['Measure Direction'] == 'X'][results_label].values[0])

        p_x = group[group['Measure Direction'] == 'X'][results_label].values[0].mean()
        p_y = group[group['Measure Direction'] == 'Y'][results_label].values[0].mean()
        # standard deviation of the mean of the probabilities
        p_x_std = group[group['Measure Direction'] == 'X'][results_label].values[0].std() / np.sqrt(N)
        p_y_std = group[group['Measure Direction'] == 'Y'][results_label].values[0].std() / np.sqrt(N)
        # convert probabilities to expectation values of X and Y
        exp_x, var_x = transform_bit_moments_to_pauli(1 - p_x, p_x_std ** 2)
        exp_y, var_y = transform_bit_moments_to_pauli(1 - p_y, p_y_std ** 2)
        xs.append(exp_x)
        ys.append(exp_y)
        x_stds.append(np.sqrt(var_x))
        y_stds.append(np.sqrt(var_y))

    return xs, ys, x_stds, y_stds


def add_moments_to_dataframe(experiment: DataFrame, results_label='Results'):
    """
    Adds the calculated expected value and standard deviation for each row of results.

    This method is provided only to store moments in the dataframe for the interested user;
    calling this method is not necessary for getting an estimate of the phase, since the moments
    are simply recalculated from results each time the robust_phase_estimate is called.

    :param experiment: an rpe experiment
    :param results_label: label for the column with results from which the moments are estimated
    :return: a copy of the experiment with Expectation and Std Deviation for results_label.
    """
    xs, ys, x_stds, y_stds = get_moments(experiment, results_label)

    expt = experiment.copy()

    expectations = [None for _ in range(len(xs) + len(ys))]
    expectations[::2] = xs
    expectations[1::2] = ys
    expt["Expectation"] = expectations

    stds = [None for _ in range(len(x_stds) + len(y_stds))]
    stds[::2] = x_stds
    stds[1::2] = y_stds
    expt["Std Deviation"] = stds

    return expt


def estimate_phase_from_moments(xs: List, ys: List, x_stds: List, y_stds: List,
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


def robust_phase_estimate(experiment: DataFrame, results_label="Results"):
    """
    Provides the estimate of the phase for an RPE experiment with results.

    This is simply a convenient wrapper around get_moments() and estimate_phase_from_moments()
    which do all of the analysis. See those methods above for details.

    :param experiment: an RPE experiment with results.
    :param results_label: label for the column with results from which the moments are estimated
    :return: an estimate of the phase of the rotation program passed into generate_rpe_experiments
    """
    xs, ys, x_stds, y_stds = get_moments(experiment, results_label)
    phase = estimate_phase_from_moments(xs, ys, x_stds, y_stds)
    return phase


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
    xs, ys, x_stds, y_stds = get_moments(experiments)
    # mutate positions, do not need the actual estimate
    estimate_phase_from_moments(xs, ys, x_stds, y_stds, positions)
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
