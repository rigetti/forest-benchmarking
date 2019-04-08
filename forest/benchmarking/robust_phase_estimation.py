from typing import Tuple, List, Sequence, Union
import warnings

import numpy as np
from numpy import pi
from functools import reduce
from operator import mul

from pyquil.quil import Program, merge_programs, DefGate, Pragma
from pyquil.quilbase import Gate
from pyquil.api import QuantumComputer
from pyquil.paulis import PauliTerm
from pyquil.operator_estimation import ExperimentSetting, zeros_state, plusZ, minusZ, \
    plusX, _OneQState, TensorProductState
from forest.benchmarking.utils import bloch_vector_to_standard_basis
from forest.benchmarking.stratified_experiment import StratifiedExperiment, Layer, \
    acquire_stratified_data

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


def get_change_of_basis_from_eigvecs(eigenvectors: Sequence[np.ndarray]) -> np.ndarray:
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
    :return: a matrix for the change of basis transformation which maps computational basis
        states to the given eigenvectors. Necessary for generate_rpe_experiments called on a 
        rotation with the given eigenvectors.
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


def change_of_basis_matrix_to_quil(qc: QuantumComputer, qubits: Sequence[int],
                                   change_of_basis: np.ndarray) -> Program:
    """
    Helper to return a native quil program for the given qc to implement the change_of_basis matrix.

    :param qc: Quantum Computer that will need to use the change of basis
    :param qubits: the qubits the program should act on
    :param change_of_basis: a unitary matrix acting on len(qubits) many qubits
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


def all_eigenvector_prep_meas_settings(qubits: Sequence[int], change_of_basis: Program):
    # experiment settings put initial state in superposition of computational basis
    # the prep and pre_meas programs simply convert between this and superposition of eigenvectors
    prep_prog = change_of_basis
    pre_meas_prog = change_of_basis.dagger()
    init_state = reduce(mul, [plusX(q) for q in qubits], TensorProductState())

    settings = []
    for xy_q in qubits:
        iz_qubits = [q for q in qubits if q != xy_q]
        xy_terms = [PauliTerm('X', xy_q), PauliTerm('Y', xy_q)]
        iz_terms = [[PauliTerm('I', xy_q)]]
        iz_terms += [[PauliTerm('I', q), PauliTerm('Z', q)] for q in iz_qubits]
        settings += [ExperimentSetting(init_state, xy_term * term) for xy_term in xy_terms
                     for term_pair in iz_terms for term in term_pair]
    return prep_prog, pre_meas_prog, settings


def fixed_and_rotate_prep_meas_settings(fix_qubit: Tuple[int, int], rotate_qubit: int,
                                        change_of_basis: Program = None):
    prep_prog = Program()
    if change_of_basis is not None:
        prep_prog += change_of_basis

    if fix_qubit[1] == 1:
        fixed_q_state = minusZ(fix_qubit[0])  # initialize fixed qubit to |1> state
    else:
        fixed_q_state = plusZ(fix_qubit[0])  # initialize fixed qubit to |0> state

    init_state = fixed_q_state * plusX(rotate_qubit)

    fixed_q_ops = [PauliTerm('I', fix_qubit[0]), PauliTerm('Z', fix_qubit[0])]
    rot_q_ops = [PauliTerm('X', rotate_qubit), PauliTerm('Y', rotate_qubit)]

    settings = [ExperimentSetting(init_state, term1*term2) for term1 in fixed_q_ops for term2 in
                rot_q_ops]

    # prepare superposition, return to z basis, then do the measurement settings.
    return prep_prog, prep_prog.dagger(), settings


def generate_rpe_experiment(rotation: Program, prep_prog: Program, pre_meas_prog: Program,
                            settings: Sequence[ExperimentSetting], num_depths: int = 6) \
        -> StratifiedExperiment:
    """
    Generate a dataframe containing all the experiments needed to perform robust phase estimation
    to estimate the angle of rotation of the given rotation program.

    In general, this experiment consists of multiple iterations of the following steps performed for
    different depths and measurement in different "directions":
        1) Prepare the equal superposition between computational basis states (i.e. the
            eigenvectors of a rotation about the Z axis)
        2) Perform a change of basis which maps the computational basis to eigenvectors of the
            rotation.
        3) Perform the rotation depth-many times, where depth=2^iteration number. Each
            eigenvector component picks up a phase from the rotation. In the 1-qubit case this
            means that the state rotates about the axis formed by the eigenvectors at a rate
            which is given by the relative phase between the two eigenvector components.
        4) Invert the change of basis to return to the computational basis.
        5) Prepare (one of) the qubit(s) for measurement along either the X or Y axis.
        6) Measure this qubit, and in the multi-qubit case other qubits participating in rotation.
    Measure_qubits can be used e.g. in the case of noisy cross-talk to measure the effective
    action of some "rotation program" that acts on completely different qubits but nonetheless
    rotates each measure_qubit.

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
    :param prep_program: typically, a program implementing the unitary change of basis
        transformation which maps the computational basis into the basis formed by eigenvectors
        of the rotation. The sign of the estimate will be determined by which computational basis
        states are mapped to which eigenvectors. Following the right-hand-rule convention,
        a rotation of RX(phi) for phi>0 about the +X axis should be paired with a change of basis
        mapping |0> to |+> and |1> to |-> . This is achieved by the gate RY(pi/2, qubit). This
        program should be provided in native gates, or gates which can be custom-compiled by
        basic_compile.
    :param num_depths: the number of depths in the protocol described in [RPE]. A depth is the
        number of consecutive applications of the rotation in a single iteration. The maximum
        depth is 2**(num_depths-1)
    :param measure_qubits: the qubits whose angle of rotation, as a result of the action of
        the rotation program, RPE will attempt to estimate. These are the only qubits measured.
    :param prepare_and_post_select: is a bitstring used only in the multi-qubit case where one
        wishes to prepare the given qubits in the given classical state, and in analysis
        discard any results where those qubits are not observed in that state. Thus for a given
        prepare_and_post_select and a given measure_qubit being measured in the X or Y basis,
        the phase being estimated is the relative phase between the eigenvectors mapped to by the
        computational basis states consistent with the post_select_state and the equal
        superposition of the measure_qubit. For two qubits, one of the qubits may be assigned a
        bit, which will yield an estimate of one of the four possible phases typically
        estimated without post_select_state specified.
    :return: a dataframe populated with all of data necessary for the RPE protocol in [RPE]
    """
    rotation_qubits = rotation.get_qubits()
    measure_qubits = [state.qubit for setting in settings for state in setting.in_state]
    qubits = tuple(rotation_qubits.union(measure_qubits))

    layers = []
    for exponent in range(num_depths):
        depth = 2 ** exponent
        depth_many_rot = (Program(rotation) for _ in range(depth))
        sequence = (Program(prep_prog), *depth_many_rot, Program(pre_meas_prog))
        layers.append(Layer(depth, sequence, tuple(settings), qubits, str(exponent)))

    return StratifiedExperiment(tuple(layers), qubits, "RPE")


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
    factor breaks the optimality guarantee. Larger additive_error leads to a longer experiment,
    but the variance bounds only apply if the additive_error sufficiently reflects reality.

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


def acquire_rpe_data(qc: QuantumComputer,
                     experiments: Union[StratifiedExperiment, Sequence[StratifiedExperiment]],
                     multiplicative_factor: float = 1.0, additive_error: float = None):
    """
    Run each experiment in the sequence of experiments.

    The number of shots run at each depth can be modified indirectly by adjusting
    multiplicative_factor and additive_error.

    :param experiments:
    :param qc: a quantum computer, e.g. QVM or QPU, that runs the experiments
    :param multiplicative_factor: ad-hoc factor to multiply the number of shots per iteration. See
        num_trials() which computes the optimal number of shots per iteration.
    :param additive_error: estimate of the max additive error in the experiment, see num_trials()
    :return:
    """
    if isinstance(experiments, StratifiedExperiment):
        experiments = [experiments]
    # TODO: provide separate helper for populating ideal num_shots?
    for expt in experiments:
        depths = [layer.depth for layer in expt.layers]
        max_depth = max(depths)
        alpha = 5 / 2  # should be > 2 for Heisenberg scaling. See eq. V.11 in [RPE]
        beta = 1 / 2  # should be > 0
        for layer in expt.layers:
            layer.num_shots = num_trials(layer.depth, max_depth, alpha, beta,
                                         multiplicative_factor, additive_error)
    acquire_stratified_data(qc, experiments)


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


# def get_variance_upper_bound(experiment: DataFrame, multiplicative_factor: float = 1.0,
#                              additive_error: float = None, results_label='Results') -> float:
#     """
#     Equation V.9 in [RPE]
#
#     :param experiment: a dataframe with RPE results. Importantly the bound follows from the number
#         of shots at each iteration of the experiment, so the data frame needs to be populated with
#         the desired number-of-shots-many results.
#     :param multiplicative_factor: ad-hoc factor to multiply the number of shots per iteration. See
#         num_trials() which computes the optimal number of shots per iteration.
#     :param additive_error: estimate of the max additive error in the experiment, see num_trials()
#     :param results_label: label for the column with results from which the variance is estimated
#     :return: An upper bound of the variance of the angle estimate corresponding to the input
#         experiments.
#     """
#     max_depth = max(experiment["Depth"].values)
#     K = np.log2(max_depth).astype(int) + 1
#
#     M_js = []
#     # 1 <= j <= K, where j is the one-indexed iteration number
#     for j in range(1, K + 1):
#         depth = 2 ** (j - 1)
#         single_depth = experiment.groupby(["Depth"]).get_group(depth).set_index(
#             'Measure Direction')
#
#         if results_label in experiment.columns.values:
#             # use the actual nunber of shots taken
#             M_j = len(single_depth.loc['X', results_label])
#         else:
#             # default prediction for standard parameters
#             M_j = num_trials(depth, max_depth, 5/2, 1/2, multiplicative_factor, additive_error)
#
#         M_js += [M_j]
#
#     # note that M_js is 0 indexed but 1 <= j <= K, so M_j = M_js[j-1]
#     return (1 - _p_max(M_js[K - 1])) * _xci(K + 1) ** 2 + sum(
#         [_xci(i + 1) ** 2 * _p_max(M_j) for i, M_j in enumerate(M_js)])


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
            warnings.warn("Decoherence limited estimate of phase {0:.3f} to depth {1:d}. You may "
                          "want to increase the additive_error and/or multiplicative_factor and "
                          "try again.".format(theta_est % (2 * pi), k//2))
            break

        # get back an estimate between -pi and pi
        theta_j_est = np.arctan2(y, x) / k

        plus_or_minus = pi / k  # the principal range bound from previous estimate
        restricted_range = [theta_est - plus_or_minus, theta_est + plus_or_minus]
        # get the offset of the new estimate from within the restricted range
        offset = (theta_j_est - restricted_range[0]) % (2 * plus_or_minus)
        # update the estimate
        theta_est = offset + restricted_range[0]
        assert restricted_range[0] <= theta_est < restricted_range[1]

        if bloch_data is not None:
            bloch_data.append((r, theta_est * k))

    return theta_est % (2 * pi)  # return value between 0 and 2pi


def robust_phase_estimate(experiment: StratifiedExperiment) -> Union[float, Sequence[float]]:
    """
    Provides the estimate of the phase for an RPE experiment with results.

    In the 1q case this is simply a convenient wrapper around get_moments() and
    estimate_phase_from_moments() which do all of the analysis; see those methods above for details.
    For multiple qubits this method determines which possible outputs are consistent with the
    post-selection-state and the possible non-z-basis measurement qubit. For each choice of the
    latter, all such possible outcomes correspond to measurement of a different relative phase.
    get_moments() is called on a dataframe with rows consistent with the particular non-z-basis
    measurement qubit and each outcome. If there is no post-select state then the number of
    relative phases estimated is equal to the dimension of the Hilbert space.

    :param experiment: an RPE experiment with results.
    :return: an estimate of the phase of the rotation program passed into generate_rpe_experiments
        If the rotation program is multi-qubit then there will be
            2**(len(meas_qubits) - len(post_select_state) - 1)
        different relative phases estimated and returned.
    """
    relative_phases = []
    for xy_q in experiment.qubits:
        expectations = []
        stddevs = []
        z_qubits = [q for q in experiment.qubits if q != xy_q]
        for label in ['X', 'Y']:
            # organize operator results by z_qubit; there are up to 2 phase estimates per z_qubit
            results_by_z_qubit = {q: [] for q in z_qubits}
            i_results = []  # there should be one i_result per layer for each label
            for layer in experiment.layers:
                results = [res for res in layer.results if res.setting.out_operator[xy_q] == label]

                for res in results:
                    for z_q in z_qubits:
                        if res.setting.out_operator[z_q] == 'Z':
                            results_by_z_qubit[z_q].append(res)
                            break
                        i_results.append(res)

            xy_expectations = []
            xy_stddevs = []
            for q, results, i_ress in zip(results_by_z_qubit.items(), i_results):
                init_state = i_ress[0].setting.in_state[q]

                # add i and z results, unless state was explicitly initialized to one (minus Z)
                if not init_state == _OneQState('Z', 1, q):
                    plus_phase_expectations = []
                    plus_phase_stddevs = []
                    for res, i_res in zip(results, i_ress):
                        plus_phase_expectations.append(i_res.expectation + res.expectation)
                        # TODO: check error propogation
                        rel_z = res.stddev/res.expectation
                        rel_i = i_res.stddev/i_res.expectation
                        plus_phase_stddevs.append(np.sqrt(rel_z**2 + rel_i**2))

                    xy_expectations.append(plus_phase_expectations)
                    xy_stddevs.append(plus_phase_stddevs)

                # subtract i and z results, unless state was explicitly initialized to zero (plus Z)
                if  not init_state ==  _OneQState('Z', 0, q):
                    minus_phase_expectations = []
                    minus_phase_stddevs = []
                    for res, i_res in zip(results, i_ress):
                        minus_phase_expectations.append(i_res.expectation - res.expectation)
                        # TODO: check error propogation
                        rel_z = res.stddev / res.expectation
                        rel_i = i_res.stddev / i_res.expectation
                        minus_phase_stddevs.append(np.sqrt(rel_z ** 2 + rel_i ** 2))

                    xy_expectations.append(minus_phase_expectations)
                    xy_stddevs.append(minus_phase_stddevs)

            expectations.append(xy_expectations)
            stddevs.append(xy_stddevs)

        for (x_exps, y_exps), (x_errs, y_errs) in zip(*zip(expectations), *zip(stddevs)):
            for x_exp, y_exp, x_err, y_err in zip(x_exps, y_exps, x_errs, y_errs):
                relative_phases.append(estimate_phase_from_moments(x_exp, y_exp, x_err, y_err))

    return relative_phases


#########
# Plotting
#########

#
# def plot_rpe_iterations(experiments: DataFrame, expected_positions: List = None) -> plt.Axes:
#     """
#     Creates a polar plot of the estimated location of the state in the plane perpendicular to the
#     axis of rotation for each iteration of RPE.
#
#     :param experiments: a dataframe with RPE results populated by a call to acquire_rpe_data
#     :param expected_positions: a list of expected (radius, angle) pairs for each iteration
#     :return: a matplotlib subplot visualizing each iteration of the RPE experiment
#     """
#     positions = []
#     xs, ys, x_stds, y_stds = get_moments(experiments)
#     # mutate positions, do not need the actual estimate
#     estimate_phase_from_moments(xs, ys, x_stds, y_stds, positions)
#     rs = [pos[0] for pos in positions]
#     angles = [pos[1] for pos in positions]
#
#     ax = plt.subplot(111, projection='polar')
#
#     # observed
#     ax.scatter(angles, rs)
#     for j, (radius, angle) in enumerate(positions):
#         ax.annotate("Ob" + str(j), (angle, radius), color='blue')
#
#     # expected
#     if expected_positions:
#         expected_rs = [pos[0] for pos in expected_positions]
#         expected_angles = [pos[1] for pos in expected_positions]
#         ax.scatter(expected_angles, expected_rs)
#         for j, (radius, angle) in enumerate(expected_positions):
#             ax.annotate("Ex" + str(j), (angle, radius), color='orange')
#         ax.set_title("RPE Iterations Observed(O) and Expected(E)", va='bottom')
#     else:
#         ax.set_title("Observed Position per RPE Iteration")
#
#     ax.set_rmax(1.0)
#     ax.set_rticks([0.5, 1])  # radial ticks
#     ax.set_rlabel_position(-22.5)  # offset radial labels to lower right quadrant
#     ax.grid(True)
#
#     return ax
