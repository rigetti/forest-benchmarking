import functools
import itertools
from dataclasses import dataclass
from operator import mul
from typing import Callable, Tuple, List, Optional, Union, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import logm, pinv, eigh

import forest.benchmarking.distance_measures as dm
import forest.benchmarking.operator_estimation as est
from forest.benchmarking.superoperator_conversion import vec, unvec
from forest.benchmarking.utils import prepare_prod_sic_state, n_qubit_pauli_basis, partial_trace
from pyquil import Program
from pyquil.api import QuantumComputer
from pyquil.operator_estimation import ExperimentSetting, \
    TomographyExperiment as PyQuilTomographyExperiment, ExperimentResult, SIC0, SIC1, SIC2, SIC3, \
    plusX, minusX, plusY, minusY, plusZ, minusZ, TensorProductState, zeros_state
from pyquil.paulis import sI, sX, sY, sZ, PauliSum, PauliTerm, is_identity
from pyquil.unitary_tools import lifted_pauli, lifted_state_operator

MAXITER = "maxiter"
OPTIMAL = "optimal"
FRO = 'fro'


@dataclass
class TomographyExperiment:
    """
    A description of tomography experiments, i.e. preparation then operations then measurements, but not
    the results of experiments.
    """

    in_ops: List[str]
    """The (optional) state preparation operations that precede execution of the `program`"""

    program: Program
    """The pyquil Program to perform tomography on"""

    out_ops: List[PauliTerm]
    """The output Pauli operators measured after the action (by conjugation in the Heisenberg picture) of the `program' 
    on the `in_op`"""


def _state_tomo_settings(qubits: Sequence[int]):
    """Yield settings over itertools.product(I, X, Y, Z).

    Used as a helper function for generate_state_tomography_experiment

    :param qubits: The qubits to tomographize.
    """
    n_qubits = len(qubits)
    for o_ops in itertools.product([sI, sX, sY, sZ], repeat=n_qubits):
        o_op = functools.reduce(mul, (op(q) for op, q in zip(o_ops, qubits)), sI())

        yield ExperimentSetting(
            in_state=zeros_state(qubits),
            out_operator=o_op,
        )


def generate_state_tomography_experiment(program: Program, qubits: List[int]):
    """Generate a (pyQuil) TomographyExperiment containing the experimental settings required
    to characterize a quantum state.

    To collect data, try::

        from pyquil.operator_estimation import measure_observables
        results = list(measure_observables(qc=qc, tomo_experiment=experiment, n_shots=100_000))

    :param program: The program to prepare a state to tomographize
    :param qubits: The qubits to tomographize
    """
    return PyQuilTomographyExperiment(settings=list(_state_tomo_settings(qubits)),
                                      program=program, qubits=qubits)


def _sic_process_tomo_settings(qubits: Sequence[int]):
    """Yield settings over SIC basis cross I,X,Y,Z operators

    Used as a helper function for generate_process_tomography_experiment

    :param qubits: The qubits to tomographize.
    """
    for in_sics in itertools.product([SIC0, SIC1, SIC2, SIC3], repeat=len(qubits)):
        i_state = functools.reduce(mul, (state(q) for state, q in zip(in_sics, qubits)),
                                   TensorProductState())
        for o_ops in itertools.product([sI, sX, sY, sZ], repeat=len(qubits)):
            o_op = functools.reduce(mul, (op(q) for op, q in zip(o_ops, qubits)), sI())

            if is_identity(o_op):
                continue

            yield ExperimentSetting(
                in_state=i_state,
                out_operator=o_op,
            )


def _pauli_process_tomo_settings(qubits):
    """Yield settings over +-XYZ basis cross I,X,Y,Z operators

    Used as a helper function for generate_process_tomography_experiment

    :param qubits: The qubits to tomographize.
    """
    for states in itertools.product([plusX, minusX, plusY, minusY, plusZ, minusZ],
                                    repeat=len(qubits)):
        i_state = functools.reduce(mul, (state(q) for state, q in zip(states, qubits)),
                                   TensorProductState())
        for o_ops in itertools.product([sI, sX, sY, sZ], repeat=len(qubits)):
            o_op = functools.reduce(mul, (op(q) for op, q in zip(o_ops, qubits)), sI())

            if is_identity(o_op):
                continue

            yield ExperimentSetting(
                in_state=i_state,
                out_operator=o_op,
            )


def generate_process_tomography_experiment(program: Program, qubits: List[int], in_basis='sic'):
    """
    Generate a (pyQuil) TomographyExperiment containing the experiment settings required to
    characterize a quantum process.

    To collect data, try::

        from pyquil.operator_estimation import measure_observables
        results = list(measure_observables(qc=qc, tomo_experiment=experiment, n_shots=100_000))

    :param program: The program describing the process to tomographize.
    :param qubits: The qubits to tomographize.
    :param in_basis: A string identifying the input basis. Either "sic" or "pauli". SIC requires
        a smaller number of experiment settings to be run.
    """
    if in_basis == 'sic':
        func = _sic_process_tomo_settings
    elif in_basis == 'pauli':
        func = _pauli_process_tomo_settings
    else:
        raise ValueError(f"Unknown basis {in_basis}")

    return PyQuilTomographyExperiment(settings=list(func(qubits)), program=program, qubits=qubits)


@dataclass
class TomographyData:
    """Experimental data from a tomography experiment"""
    in_ops: Optional[List[str]]
    """The (optional) state preparation operations that precede execution of the `program`"""

    program: Program
    """The pyquil Program to perform tomography on"""

    out_ops: List[PauliTerm]
    """The output Pauli operators measured after the action (by conjugation in the Heisenberg picture) of the `program' 
    on the `in_op`"""

    dimension: int
    """Dimension of the Hilbert space"""

    number_qubits: int
    """number of qubits"""

    expectations: List[float]
    """expectation values as reported from the QPU"""

    variances: List[float]
    """variances associated with the `expectation`"""

    counts: List[int]
    """number of shots used to calculate the `expectation`"""


def shim_pyquil_results_to_TomographyData(program, qubits, results: List[ExperimentResult]):
    return TomographyData(
        in_ops=[r.setting.in_operator for r in results[1:]],
        out_ops=[r.setting.out_operator for r in results[1:]],
        expectations=[r.expectation for r in results[1:]],
        variances=[r.stddev ** 2 for r in results[1:]],
        program=program,
        number_qubits=len(qubits),
        dimension=2 ** len(qubits),
        counts=[r.total_counts for r in results[1:]],
    )


def acquire_tomography_data(experiment: TomographyExperiment, qc: QuantumComputer, var: float = 0.01,
                            symmetrize=False) -> TomographyData:
    """
    Acquire tomographic data used to estimate a quantum state or process. If the experiment has no input operators
    then state tomography is assumed.

    :param symmetrize: dictates whether to symmetrize readout when estimating the Pauli expectations.
    :param experiment: TomographyExperiment for the desired state or process
    :param qc: quantum device used to collect data
    :param float var: maximum tolerable variance per observable
    :return: The "TomographyData" corresponding to the TomographyExperiment
    """
    # get qubit information
    qubits = experiment.program.get_qubits()
    n_qubits = len(qubits)
    dimension = 2 ** len(qubits)

    expectations = []
    variances = []
    counts = []

    if experiment.in_ops is None:
        # state tomography
        for op in experiment.out_ops:
            # data aqcuisition
            expectation, variance, count = est.estimate_locally_commuting_operator(experiment.program, PauliSum([op]),
                                                                                   var, qc, symmetrize=symmetrize)
            expectations.append(np.real(expectation[0]))
            variances.append(variance[0, 0].real)
            counts.append(count)
    else:
        # process tomography
        for in_op in experiment.in_ops:
            for op in experiment.out_ops:
                # data aqcuisition
                tot_prog = prepare_prod_sic_state(in_op) + experiment.program
                expectation, variance, count = est.estimate_locally_commuting_operator(tot_prog, PauliSum([op]), var,
                                                                                       qc, symmetrize=symmetrize)

                expectations.append(np.real(expectation[0]))
                variances.append(variance[0, 0].real)
                counts.append(count)

    exp_data = TomographyData(
        in_ops=experiment.in_ops,
        program=experiment.program,
        out_ops=experiment.out_ops,
        dimension=dimension,
        number_qubits=n_qubits,
        expectations=expectations,
        variances=variances,
        counts=counts
    )
    return exp_data


@dataclass
class StateTomographyEstimate:
    """State estimate from tomography experiment"""

    state_point_est: np.ndarray
    """A point estimate of the quantum state rho output from the program being tomographed"""

    type: str
    """Type of estimator used e.g. 'linear inversion' or 'hedged_MLE'"""

    beta: Optional[float]
    """The Hedging parameter"""

    entropy: Optional[float]
    """The entropy penalty parameter"""

    dilution: Optional[float]
    """A diluation parameter"""

    loglike: Optional[float]
    """The log likelihood at the current estimate"""


@dataclass
class ProcessTomographyEstimate:
    """Process estimate from tomography experiment"""

    process_choi_est: np.ndarray
    """A point estimate of the quantum process being tomographed represented as a choi matrix"""

    type: str
    """Type of estimator used e.g. 'pgdb'"""


@dataclass
class TomographyEstimate:
    """State/Process estimate from tomography experiment"""
    in_ops: Optional[List[str]]
    """The (optional) state preparation operations that precede execution of the `program`"""

    program: Program
    """The pyquil Program to perform DFE on"""

    out_ops: List[PauliTerm]
    """The output Pauli operators measured after the action (by conjugation in the Heisenberg picture) of the `program' 
    on the `in_op`"""

    dimension: int
    """Dimension of the Hilbert space"""

    number_qubits: int
    """number of qubits"""

    expectations: List[float]
    """expectation values as reported from the QPU"""

    variances: List[float]
    """variances associated with the `expectation`"""

    estimate: Union[StateTomographyEstimate, ProcessTomographyEstimate]
    """State or process estimate from tomography experiment"""


def linear_inv_state_estimate(results: List[ExperimentResult],
                              qubits: List[int]) -> np.ndarray:
    """
    Estimate a quantum state using linear inversion.

    This is the simplest state tomography post processing. To use this function,
    collect state tomography data with :py:func:`generate_state_tomography_experiment`
    and :py:func:`~pyquil.operator_estimation.measure_observables`.

    For more details on this post-processing technique,
    see https://en.wikipedia.org/wiki/Quantum_tomography#Linear_inversion or
    see section 3.4 of

    [WOOD] Initialization and characterization of open quantum systems
           C. Wood,
           PhD thesis from University of Waterloo, (2015).
           http://hdl.handle.net/10012/9557

    :param results: A tomographically complete list of results.
    :param qubits: All qubits that were tomographized. This specifies the order in
        which qubits will be kron'ed together.
    :return: A point estimate of the quantum state rho.
    """
    measurement_matrix = np.vstack([
        vec(lifted_pauli(result.setting.out_operator, qubits=qubits)).T.conj()
        for result in results
    ])
    expectations = np.array([result.expectation for result in results])
    rho = pinv(measurement_matrix) @ expectations
    return unvec(rho)


def construct_projection_operators_on_n_qubits(num_qubits) -> List[np.ndarray]:
    """
    """
    # Identity prop to the size of Hilbert space
    IdH = np.eye(2 ** num_qubits, 2 ** num_qubits)
    effects = []
    for i, operator in enumerate(n_qubit_pauli_basis(num_qubits).ops):
        if i == 0:
            continue
            # Might need to change for >1Q.
        effects.append((IdH + operator) / 2)
        effects.append((IdH - operator) / 2)
    return effects


def iterative_mle_state_estimate(results: List[ExperimentResult], qubits: List[int], dilution=.005,
                                 entropy_penalty=0.0, beta=0.0, tol=1e-9, maxiter=100_000) \
        -> TomographyEstimate:
    """
    Given tomography data, use one of three iterative algorithms to return an estimate of the
    state.
    
    "... [The iterative] algorithm is characterized by a very high convergence rate and features a
    simple adaptive procedure that ensures likelihood increase in every iteration and
    convergence to the maximum-likelihood state." [DIMLE1]
    
    For MLE only option, set:        entropy_penalty=0.0 and beta=0.0.
    For MLE + maximum entropy, set:  entropy_penalty=(non-zero) and beta=0.0.
    For MLE + hedging, set:          entropy_penalty=0.0 and beta=(non-zero).
    
    The basic algorithm is due to
    
    [DIMLE1] Diluted maximum-likelihood algorithm for quantum tomography
             Řeháček et al.,
             PRA 75, 042108 (2007)
             https://doi.org/10.1103/PhysRevA.75.042108
             https://arxiv.org/abs/quant-ph/0611244

    with improvements from

    [DIMLE2] Quantum-State Reconstruction by Maximizing Likelihood and Entropy
             Teo et al.,
             PRL 107, 020404 (2011)
             https://doi.org/10.1103/PhysRevLett.107.020404
             https://arxiv.org/abs/1102.2662
                
    [HMLE]   Hedged Maximum Likelihood Quantum State Estimation
             Blume-Kohout,
             PRL, 105, 200504 (2010)
             https://doi.org/10.1103/PhysRevLett.105.200504
             https://arxiv.org/abs/1001.2029

    [IHMLE]  Iterative Hedged MLE from Yong Siah Teo's PhD thesis, see Eqn. 1.5.13 on page 88:
             Numerical Estimation Schemes for Quantum Tomography
             Y. S. Teo
             PhD Thesis, from National University of Singapore, (2013)
             https://arxiv.org/pdf/1302.3399.pdf

    :param state_tomography_experiment_data data: namedtuple.
    :param dilution: delta  = 1 / epsilon where epsilon is the dilution parameter used in [DIMLE1].
        in practice epsilon= 1/N
    :param entropy_penalty: the entropy penalty parameter from [DIMLE2].
    :param beta: The Hedging parameter from [HMLE].
    :param tol: The largest difference in the frobenious norm between update steps that will cause
         the algorithm to conclude that it has converged.
    :param maxiter: The maximum number of iterations to perform before aborting the procedure.
    :return: A TomographyEstimate whose estimate is a StateTomographyEstimate
    """
    data = shim_pyquil_results_to_TomographyData(
        program=None,
        qubits=qubits,
        results=results
    )
    exp_type = 'iterative_MLE'
    if (entropy_penalty != 0.0) and (beta != 0.0):
        raise ValueError("One can't sensibly do entropy penalty and hedging. Do one or the other"
                         " but not both.")
    else:
        if entropy_penalty != 0.0:
            exp_type = 'max_entropy_MLE'
        if beta != 0.0:
            exp_type = 'heged_MLE'

    # Identity prop to the size of Hilbert space
    dim = 2**len(qubits)
    IdH = np.eye(dim, dim)

    freq = []
    for expectation, count in zip(data.expectations, data.counts):
        num_plus_one = int((expectation + 1) / 2 * count)
        freq.append(num_plus_one)
        freq.append(count - num_plus_one)

    effects = construct_projection_operators_on_n_qubits(data.number_qubits)

    rho = IdH / data.dimension
    epsilon = 1 / dilution  # Dilution parameter used in [DIMLE1].
    iteration = 1
    status = OPTIMAL
    while True:
        rho_temp = rho
        if iteration >= maxiter:
            status = MAXITER
            break
        # Vanilla Iterative MLE
        Tk = _R(rho, effects, freq) - IdH  # Eq 6 of [DIMLE2] with \lambda = 0.

        # MaxENT Iterative MLE
        if entropy_penalty > 0.0:
            constraint = (logm(rho) - IdH * np.trace(rho.dot(logm(rho))))
            Tk -= (entropy_penalty * constraint)  # Eq 6 of [DIMLE2] with \lambda \neq 0.

        # Hedged Iterative MLE
        if beta > 0.0:
            num_meas = data.counts[0] * len(data.out_ops)
            # TODO: decide if can use pinv consistently from one of np or scipy
            Tk = (beta * (np.linalg.pinv(rho) - data.dimension * IdH)
                  + num_meas * (_R(rho, effects, freq) - IdH))

        # compute iterative estimate of rho     
        update_map = (IdH + epsilon * Tk)
        rho = update_map.dot(rho).dot(update_map)
        rho /= np.trace(rho)  # Eq 5 of [DIMLE2].
        if np.linalg.norm(rho - rho_temp, FRO) < tol:
            break
        iteration += 1

    estimate = StateTomographyEstimate(
        state_point_est=rho,
        type=exp_type,
        beta=beta,
        entropy=entropy_penalty,
        dilution=dilution,
        loglike=_LL(rho, effects, freq)
    )

    est_data = TomographyEstimate(
        in_ops=data.in_ops,
        program=data.program,
        out_ops=data.out_ops,
        dimension=data.dimension,
        number_qubits=data.number_qubits,
        expectations=data.expectations,
        variances=data.variances,
        estimate=estimate
    )

    return est_data, status


def _R(state, effects, observed_frequencies):
    r"""
    This is Eqn 5 in [DIMLE1], i.e.

    R(rho) = (1/N) \sum_j (n_j/Pr_j) Pi_j
           = \sum_j (f_j/Pr_j) Pi_j

    N = total number of measurements
    n_j = number of times j'th outcome was observed
    f_j = n_j/N observed frequencies => \sum_j f_j  == 1
    Pi_j = measurement operator or projector
    Pr_j = Tr[Pi_j \rho]

    :param state: The state (given as a density matrix) that we think we have.
    :param effects: The measurements we've performed.
    :param observed_frequencies: The frequencies (normalized histograms of results) we have observed
     associated with effects.
    """
    # this small number ~ 10^-304 is added so that we don't get divide by zero errors
    machine_eps = np.finfo(float).tiny
    # have a zero in the numerator, we can fix this is we look a little more carefully.
    predicted_probs = np.array([np.real(np.trace(state.dot(effect))) for effect in effects])
    update_operator = sum([effect * observed_frequencies[i] / (predicted_probs[i] + machine_eps)
                           for i, effect in enumerate(effects)])
    return update_operator


def _LL(state, effects, observed_frequencies) -> float:
    """
    The log Likelihood function used in the diluted MLE tomography routine.

    :param state: The state (given as a density matrix) that we think we have.
    :param effects: The measurements we've performed.
    :param observed_frequencies: The frequencies (normalized histograms of results) we have observed
     associated with effects.
    :return: The log likelihood that our state is the one we believe it is.
    """
    observed_frequencies = np.array(observed_frequencies)
    predicted_probs = np.array([np.real(np.trace(state.dot(effect))) for effect in effects])
    return sum(np.log10(predicted_probs) * observed_frequencies)


def proj_to_cp(choi_vec):
    """
    Projects the vectorized Choi representation of a process, into the nearest vectorized choi matrix in the space of
    completely positive maps. Equation 9 of [PGD]
    :param choi_vec: vectorized density matrix or Choi representation of a process
    :return: closest vectorized choi matrix in the space of completely positive maps
    """
    matrix = unvec(choi_vec)
    hermitian = (matrix + matrix.conj().T) / 2  # enforce Hermiticity
    d, v = np.linalg.eigh(hermitian)
    d[d < 0] = 0  # enforce completely positive by removing negative eigenvalues
    D = np.diag(d)
    return vec(v @ D @ v.conj().T)


def proj_to_tni(choi_vec):
    """
    Projects the vectorized Choi matrix of a process into the space of trace non-increasing maps. Equation 33 of [PGD]
    :param choi_vec: vectorized Choi representation of a process
    :return: The vectorized Choi representation of the projected TNI process
    """
    dim = int(np.sqrt(np.sqrt(choi_vec.size)))

    # trace out the output Hilbert space
    pt = partial_trace(unvec(choi_vec), dims=[dim, dim], keep=[0])

    hermitian = (pt + pt.conj().T) / 2  # enforce Hermiticity
    d, v = np.linalg.eigh(hermitian)
    d[d > 1] = 1  # enforce trace preserving
    D = np.diag(d)
    projection = v @ D @ v.conj().T

    trace_increasing_part = np.kron((pt - projection) / dim, np.eye(dim))

    return choi_vec - vec(trace_increasing_part)


def proj_to_tp(choi_vec):
    """
    Projects the vectorized Choi representation of a process into the closest processes in the space of trace preserving
    maps. Equation 13 of [PGD]
    :param choi_vec: vectorized Choi representation of a process
    :return: The vectorized Choi representation of the projected TP process
    """
    dim = int(np.sqrt(np.sqrt(choi_vec.size)))
    b = vec(np.eye(dim, dim))
    # construct M, which acts as partial trace over output Hilbert space
    M = np.zeros((dim ** 2, dim ** 4))
    for i in range(dim):
        e = np.zeros((dim, 1))
        e[i] = 1
        B = np.kron(np.eye(dim, dim), e.T)
        M = M + np.kron(B, B)
    return choi_vec + 1 / dim * (M.conj().T @ b - M.conj().T @ M @ choi_vec)


def _constraint_project(choi_mat, trace_preserving=True):
    """
    Projects the given Choi matrix into the subspace of Completetly Positive and either Trace Perserving (TP) or
    Trace-Non-Increasing maps.
    Uses Dykstra's algorithm with the stopping criterion presented in:

    [DYKALG] Dykstra’s algorithm and robust stopping criteria
             Birgin et al.,
             (Springer US, Boston, MA, 2009), pp. 828–833, ISBN 978-0-387-74759-0.
             https://doi.org/10.1007/978-0-387-74759-0_143

    This method is suggested in [PGD]

    :param choi_mat: A density matrix corresponding to the Choi representation estimate of a quantum process.
    :param trace_preserving: Default project the estimate to a trace-preserving process. False for trace non-increasing
    :return: The choi representation of CPTP map that is closest to the given state.
    """
    shape = choi_mat.shape
    old_CP_change = vec(np.zeros(shape))
    old_TP_change = vec(np.zeros(shape))
    last_CP_projection = vec(np.zeros(shape))
    last_state = vec(choi_mat)

    while True:
        # Dykstra's algorithm
        pre_CP = last_state - old_CP_change
        CP_projection = proj_to_cp(pre_CP)
        new_CP_change = CP_projection - pre_CP

        pre_TP = CP_projection - old_TP_change
        if trace_preserving:
            new_state = proj_to_tp(pre_TP)
        else:
            new_state = proj_to_tni(pre_TP)
        new_TP_change = new_state - pre_TP

        CP_change_change = new_CP_change - old_CP_change
        TP_change_change = new_TP_change - old_TP_change
        state_change = new_state - last_state

        # stopping criterion
        if np.linalg.norm(CP_change_change, ord=2) ** 2 + np.linalg.norm(TP_change_change, ord=2) ** 2 \
                + 2 * abs(np.dot(old_TP_change.conj().T, state_change)) \
                + 2 * abs(np.dot(old_CP_change.conj().T, (CP_projection - last_CP_projection))) < 1e-4:
            break

        # store results from this iteration
        old_CP_change = new_CP_change
        old_TP_change = new_TP_change
        last_CP_projection = CP_projection
        last_state = new_state

    return unvec(new_state)


def _extract_from_results(results: List[ExperimentResult], qubits: List[int]):
    """
    Construct the matrix A such that the probabilities p_ij of outcomes n_ij given an estimate E
    can be cast in a vectorized form.

    Specifically::

        p = vec(p_ij) = A x vec(E)

    This yields convenient vectorized calculations of the cost and its gradient, in terms of A, n,
    and E.
    """
    A = []
    n = []
    grand_total_shots = 0

    for result in results:
        in_state_matrix = lifted_state_operator(result.setting.in_state, qubits=qubits)
        operator = lifted_pauli(result.setting.out_operator, qubits=qubits)
        proj_plus = (np.eye(2 ** len(qubits)) + operator) / 2
        proj_minus = (np.eye(2 ** len(qubits)) - operator) / 2

        # Constructing A per eq. (22)
        # TODO: figure out if we can avoid re-splitting into Pi+ and Pi- counts
        A += [
            # vec() turns into a column vector; transpose to a row vector; index into the
            # 1 row to avoid an extra tensor dimension when we call np.asarray(A).
            vec(np.kron(in_state_matrix, proj_plus.T)).T[0],
            vec(np.kron(in_state_matrix, proj_minus.T)).T[0],
        ]

        expected_plus_ones = (1 + result.expectation) / 2
        n += [
            result.total_counts * expected_plus_ones,
            result.total_counts * (1 - expected_plus_ones)
        ]
        grand_total_shots += result.total_counts

    n_qubits = len(qubits)
    dimension = 2 ** n_qubits
    A = np.asarray(A) / dimension ** 2
    n = np.asarray(n)[:, np.newaxis] / grand_total_shots
    return A, n


def pgdb_process_estimate(results: List[ExperimentResult], qubits: List[int],
                          trace_preserving=True) -> np.ndarray:
    """
    Provide an estimate of the process via Projected Gradient Descent with Backtracking.

    [PGD] Maximum-likelihood quantum process tomography via projected gradient descent
          Knee et al.,
          Phys. Rev. A 98, 062336 (2018)
          https://dx.doi.org/10.1103/PhysRevA.98.062336
          https://arxiv.org/abs/1803.10062

    :param results: A tomographically complete list of ExperimentResults
    :param qubits: A list of qubits giving the tensor order of the resulting Choi matrix.
    :param trace_preserving: Whether to project the estimate to a trace-preserving process. If
        set to False, we ensure trace non-increasing.
    :return: an estimate of the process in the Choi matrix representation.
    """
    # construct the matrix A and vector n from the data for vectorized calculations of
    # the cost function and its gradient
    A, n = _extract_from_results(results, qubits[::-1])

    dim = 2 ** len(qubits)
    est = np.eye(dim ** 2, dim ** 2, dtype=complex) / dim  # initial estimate
    old_cost = _cost(A, n, est)  # initial cost, which we want to decrease
    mu = 3 / (2 * dim ** 2)  # inverse learning rate
    gamma = .3  # tolerance of letting the constrained update deviate from true gradient; larger is more demanding
    while True:
        gradient = _grad_cost(A, n, est)
        update = _constraint_project(est - gradient / mu, trace_preserving) - est

        # determine step size factor, alpha
        alpha = 1
        new_cost = _cost(A, n, est + alpha * update)
        change = gamma * alpha * np.dot(vec(update).conj().T, vec(gradient))
        while new_cost > old_cost + change:
            alpha = .5 * alpha
            change = .5 * change  # directly update change, corresponding to update of alpha
            new_cost = _cost(A, n, est + alpha * update)

            # small alpha stopgap
            if alpha < 1e-15:
                break

        # update estimate
        est += alpha * update
        if old_cost - new_cost < 1e-10:
            break
        # store current cost
        old_cost = new_cost


    return est


def _cost(A, n, estimate, eps=1e-6):
    """
    Computes the cost (negative log likelihood) of the estimated process using the vectorized
    version of equation 4 of [PGD].

    See the appendix of [PGD].

    :param A: a matrix constructed from the input states and POVM elements (eq. 22) that aids
        in calculating the model probabilities p.
    :param n: vectorized form of the observed counts n_ij
    :param estimate: the current model Choi representation of an estimated process for which we
        report the cost.
    :return: Cost of the estimate given the data, n
    """
    p = A @ vec(estimate)  # vectorized form of the probabilities of outcomes, p_ij
    # see appendix on "stalling"
    p = np.clip(p, a_min=eps, a_max=None)
    return - n.T @ np.log(p)


def _grad_cost(A, n, estimate, eps=1e-6):
    """
    Computes the gradient of the cost, leveraging the vectorized calculation given in the
    appendix of [PGD]

    :param A: a matrix constructed from the input states and POVM elements (eq. 22) that aids
        in calculating the model probabilities p.
    :param n: vectorized form of the observed counts n_ij
    :param estimate: the current model Choi representation of an estimated process for which we
        compute the gradient.
    :return: Gradient of the cost of the estimate given the data, n
    """
    p = A @ vec(estimate)
    # see appendix on "stalling"
    p = np.clip(p, a_min=eps, a_max=None)
    eta = n / p
    return unvec(-A.conj().T @ eta)


def project_density_matrix(rho) -> np.ndarray:
    """
    Project a possibly unphysical estimated density matrix to the closest (with respect to the
    2-norm) positive semi-definite matrix with trace 1, that is a valid quantum state.

    This is the so called "wizard" method. It is described in the following reference:

    [MLEWIZ] Efficient Method for Computing the Maximum-Likelihood Quantum State from
             Measurements with Additive Gaussian Noise
             Smolin et al.,
             Phys. Rev. Lett. 108, 070502 (2012)
             https://doi.org/10.1103/PhysRevLett.108.070502
             https://arxiv.org/abs/1106.5458

    :param rho: Numpy array containing the density matrix with dimension (N, N)
    :return rho_projected: The closest positive semi-definite trace 1 matrix to rho.
    """

    # Rescale to trace 1 if the matrix is not already
    rho_impure = rho / np.trace(rho)

    dimension = rho_impure.shape[0]  # the dimension of the Hilbert space
    [eigvals, eigvecs] = eigh(rho_impure)

    # If matrix is already trace one PSD, we are done
    if np.min(eigvals) >= 0:
        return rho_impure

    # Otherwise, continue finding closest trace one, PSD matrix
    eigvals = list(eigvals)
    eigvals.reverse()
    eigvals_new = [0.0] * len(eigvals)

    i = dimension
    accumulator = 0.0  # Accumulator
    while eigvals[i - 1] + accumulator / float(i) < 0:
        accumulator += eigvals[i - 1]
        i -= 1
    for j in range(i):
        eigvals_new[j] = eigvals[j] + accumulator / float(i)
    eigvals_new.reverse()

    # Reconstruct the matrix
    rho_projected = functools.reduce(np.dot, (eigvecs,
                                              np.diag(eigvals_new),
                                              np.conj(eigvecs.T)))

    return rho_projected


def _resample_expectations_with_beta(results, prior_counts=1):
    """Resample expectation values by constructing a beta distribution and sampling from it.

    Used by :py:func:`estimate_variance`.

    :param results: A list of ExperimentResults
    :param prior_counts: Number of "counts" to add to alpha and beta for the beta distribution
        from which we sample.
    :return: A new list of ``results`` where each ExperimentResult's ``expectation`` field
        contained a resampled expectation value
    """
    resampled_results = []
    for result in results:
        # reconstruct the raw counts of observations from the pauli observable mean
        num_plus = ((result.expectation + 1) / 2) * result.total_counts
        num_minus = result.total_counts - num_plus

        # We resample this data assuming it was from a beta distribution,
        # with additive smoothing
        alpha = num_plus + prior_counts
        beta = num_minus + prior_counts

        # transform bit bias back to pauli expectation value
        resampled_expect = 2 * np.random.beta(alpha, beta) - 1
        resampled_results += [ExperimentResult(
            setting=result.setting,
            expectation=resampled_expect,
            stddev=result.stddev,
            total_counts=result.total_counts,
        )]
    return resampled_results


def estimate_variance(results: List[ExperimentResult],
                      qubits: List[int],
                      tomo_estimator: Callable,
                      functional: Callable,
                      target_state=None,
                      n_resamples: int = 40,
                      project_to_physical: bool = False) -> Tuple[float, float]:
    """
    Use a simple bootstrap-like method to return an errorbar on some functional of the
    quantum state.

    :param results: Measured results from a state tomography experiment
    :param qubits: Qubits that were tomographized.
    :param tomo_estimator: takes in ``results, qubits`` and returns a corresponding
        estimate of the state rho, e.g. ``linear_inv_state_estimate``
    :param functional: Which functional to find variance, e.g. ``dm.purity``.
    :param target_state: A density matrix of the state with respect to which the distance
        functional is measured. Not applicable if functional is ``dm.purity``.
    :param n_resamples: The number of times to resample.
    :param project_to_physical: Whether to project the estimated state to a physical one
        with :py:func:`project_density_matrix`.
    """
    if functional != dm.purity:
        if target_state is None:
            raise ValueError("You're not using the `purity` functional. "
                             "Please specify a target state.")

    sample_estimate = []
    for _ in range(n_resamples):
        resampled_results = _resample_expectations_with_beta(results)
        estimate = tomo_estimator(resampled_results, qubits)

        # TODO: Shim! over different return values between linear inv. and mle
        if isinstance(estimate, np.ndarray):
            rho = estimate
        else:
            rho = estimate.estimate.state_point_est

        if project_to_physical:
            rho = project_density_matrix(rho)

        # Calculate functional of the state
        if functional == dm.purity:
            sample_estimate.append(np.real(dm.purity(rho, dim_renorm=False)))
        else:
            sample_estimate.append(np.real(functional(target_state, rho)))

    return np.mean(sample_estimate), np.var(sample_estimate)


THREE_COLOR_MAP = ['#48737F', '#FFFFFF', '#D6619E']
rigetti_3_color_cm = LinearSegmentedColormap.from_list("Rigetti", THREE_COLOR_MAP[::-1], N=100)


def plot_pauli_transfer_matrix(ptransfermatrix, ax, title = ''):
    """
    Visualize the Pauli Transfer Matrix of a process.
    :param numpy.ndarray ptransfermatrix: The Pauli Transfer Matrix
    :param ax: The matplotlib axes.
    :param labels: The labels for the operator basis states.
    :param title: The title for the plot
    :return: The modified axis object.
    :rtype: AxesSubplot
    """
    im = ax.imshow(np.real(ptransfermatrix), interpolation="nearest", cmap=rigetti_3_color_cm, vmin=-1,vmax=1)
    dim_squared = ptransfermatrix.shape[0]
    num_qubits = np.int(np.log2(np.sqrt(dim_squared)))
    labels = [''.join(x) for x in itertools.product('IXYZ', repeat=num_qubits)]
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(dim_squared))
    ax.set_xlabel("Input Pauli Operator", fontsize=20)
    ax.set_yticks(range(dim_squared))
    ax.set_ylabel("Output Pauli Operator", fontsize=20)
    ax.set_title(title, fontsize=25)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)
    ax.grid(False)
    return ax
