import functools
import itertools
from operator import mul
from typing import Callable, Tuple, List, Sequence, Iterator
import warnings

import numpy as np
from scipy.linalg import logm, pinv

from pyquil import Program
from pyquil.api import QuantumComputer
from pyquil.unitary_tools import lifted_pauli as pauli2matrix, lifted_state_operator as state2matrix

import forest.benchmarking.distance_measures as dm
from forest.benchmarking.utils import all_traceless_pauli_terms
from forest.benchmarking.operator_tools import vec, unvec, proj_choi_to_physical
from forest.benchmarking.operator_tools.project_state_matrix import project_state_matrix_to_physical
from forest.benchmarking.observable_estimation import ExperimentSetting, ObservablesExperiment, \
    ExperimentResult, SIC0, SIC1, SIC2, SIC3, plusX, minusX, plusY, minusY, plusZ, minusZ, \
    TensorProductState, zeros_state, group_settings
from forest.benchmarking.direct_fidelity_estimation import acquire_dfe_data # TODO: replace

MAXITER = "maxiter"
OPTIMAL = "optimal"
FRO = 'fro'


# ==================================================================================================
# Generate state and process tomography experiments
# ==================================================================================================
def _state_tomo_settings(qubits: Sequence[int]):
    """
    Yield settings over every non-identity observable in the Pauli basis over the qubits.

    Used as a helper function for generate_state_tomography_experiment

    :param qubits: The qubits to tomographize.
    """
    for obs in all_traceless_pauli_terms(qubits):
        yield ExperimentSetting(
            in_state=zeros_state(qubits),
            observable=obs,
        )


def generate_state_tomography_experiment(program: Program, qubits: List[int]) \
        -> ObservablesExperiment:
    """
    Generate an ObservablesExperiment containing the experimental settings required
    to characterize a quantum state.

    To collect data, try::

        from forest.benchmarking.observable_estimation import estimate_observables
        results = list(estimate_observables(qc=qc, experiment, num_shots=100_000))

    :param program: The program to prepare a state to tomographize
    :param qubits: The qubits to tomographize
    """
    return ObservablesExperiment(settings=list(_state_tomo_settings(qubits)), program=program)


def _sic_process_tomo_settings(qubits: Sequence[int]):
    """
    Yield settings over SIC basis across I, X, Y, Z operators

    Used as a helper function for generate_process_tomography_experiment

    :param qubits: The qubits to tomographize.
    """
    for in_sics in itertools.product([SIC0, SIC1, SIC2, SIC3], repeat=len(qubits)):
        i_state = functools.reduce(mul, (state(q) for state, q in zip(in_sics, qubits)),
                                   TensorProductState())
        for obs in all_traceless_pauli_terms(qubits):
            yield ExperimentSetting(
                in_state=i_state,
                observable=obs,
            )


def _pauli_process_tomo_settings(qubits):
    """
    Yield settings over +-XYZ basis across I, X, Y, Z operators

    Used as a helper function for generate_process_tomography_experiment

    :param qubits: The qubits to tomographize.
    """
    for states in itertools.product([plusX, minusX, plusY, minusY, plusZ, minusZ],
                                    repeat=len(qubits)):
        i_state = functools.reduce(mul, (state(q) for state, q in zip(states, qubits)),
                                   TensorProductState())
        for obs in all_traceless_pauli_terms(qubits):
            yield ExperimentSetting(
                in_state=i_state,
                observable=obs,
            )


def generate_process_tomography_experiment(program: Program, qubits: List[int], in_basis='pauli')\
        -> ObservablesExperiment:
    """
    Generate an ObservablesExperiment containing the experiment settings required to
    characterize a quantum process.

    To collect data, try::

        from forest.benchmarking.observable_estimation import estimate_observables
        results = list(estimate_observables(qc=qc, experiment, num_shots=100_000))

    :param program: The program describing the process to tomographize.
    :param qubits: The qubits to tomographize.
    :param in_basis: A string identifying the input basis. Either "sic" or "pauli". SIC requires
        a smaller number of experiment settings to be run.
    """
    if in_basis.upper() == 'SIC':
        func = _sic_process_tomo_settings
    elif in_basis.upper() == 'PAULI':
        func = _pauli_process_tomo_settings
    else:
        raise ValueError(f"Unknown basis {in_basis}")

    return ObservablesExperiment(settings=list(func(qubits)), program=program)


# ==================================================================================================
# STATE tomography: estimation methods and helper functions
# ==================================================================================================

def linear_inv_state_estimate(results: List[ExperimentResult],
                              qubits: List[int]) -> np.ndarray:
    """
    Estimate a quantum state using linear inversion.

    This is the simplest state tomography post processing. To use this function,
    collect state tomography data with :py:func:`generate_state_tomography_experiment`
    and :py:func:`~observable_estimation.estimate_observables`.

    For more details on this post-processing technique,
    see https://en.wikipedia.org/wiki/Quantum_tomography#Linear_inversion or
    see section 3.4 of [WOOD]_

    .. [WOOD] Initialization and characterization of open quantum systems.
           C. Wood.
           PhD thesis from University of Waterloo, (2015).
           http://hdl.handle.net/10012/9557

    :param results: A tomographically complete list of results.
    :param qubits: All qubits that were tomographized. This specifies the order in
        which qubits will be kron'ed together; the first qubit in the list is the left-most
        tensor factor.
    :return: A point estimate of the quantum state rho.
    """
    # state2matrix and pauli2matrix use pyquil tensor factor ordering where the least significant
    # qubit, e.g. qubit 0, is the right-most tensor factor. We stick with the standard convention
    # here that the first qubit in the list is the left-most tensor factor, so we have to reverse
    # the qubits before passing to state2matrix and pauli2matrix
    qs = qubits[::-1]
    measurement_matrix = np.vstack([
        vec(pauli2matrix(result.setting.observable, qubits=qs)).T.conj() for result in results ])
    expectations = np.array([result.expectation for result in results])
    rho = pinv(measurement_matrix) @ expectations
    # add in the traceful identity term
    dim = 2**len(qubits)
    return unvec(rho) + np.eye(dim)/dim


def iterative_mle_state_estimate(results: List[ExperimentResult], qubits: List[int], epsilon=.1,
                                 entropy_penalty=0.0, beta=0.0, tol=1e-9, maxiter=10_000) \
        -> np.ndarray:
    """
    Given tomography data, use one of three iterative algorithms to return an estimate of the
    state.
    
        "... [The iterative] algorithm is characterized by a very high convergence rate and
        features a simple adaptive procedure that ensures likelihood increase in every iteration
        and convergence to the maximum-likelihood state." [DIMLE1]_

    There are three options triggered by appropriately setting input parameters:

        - MLE only:                 ``entropy_penalty=0.0``         and ``beta=0.0``
        - MLE + maximum entropy:    ``entropy_penalty=`` (non-zero)  and ``beta=0.0``
        - MLE + hedging:            ``entropy_penalty=0.0``         and ``beta=`` (non-zero).
    
    The basic algorithm is due to [DIMLE1]_, with improvements from [DIMLE2]_, [HMLE]_,
    and [IHMLE]_.
    
    .. [DIMLE1] Diluted maximum-likelihood algorithm for quantum tomography.
             Řeháček et al.
             PRA 75, 042108 (2007).
             https://doi.org/10.1103/PhysRevA.75.042108
             https://arxiv.org/abs/quant-ph/0611244

    .. [DIMLE2] Quantum-State Reconstruction by Maximizing Likelihood and Entropy.
             Teo et al.
             PRL 107, 020404 (2011).
             https://doi.org/10.1103/PhysRevLett.107.020404
             https://arxiv.org/abs/1102.2662
                
    .. [HMLE]   Hedged Maximum Likelihood Quantum State Estimation.
             Blume-Kohout.
             PRL, 105, 200504 (2010).
             https://doi.org/10.1103/PhysRevLett.105.200504
             https://arxiv.org/abs/1001.2029

    .. [IHMLE]  Iterative Hedged MLE from Yong Siah Teo's PhD thesis. see Eqn. 1.5.13 on page 88:
             Numerical Estimation Schemes for Quantum Tomography.
             Y. S. Teo.
             PhD Thesis, from National University of Singapore, (2013).
             https://arxiv.org/pdf/1302.3399.pdf

    :param results: Measured results from a state tomography experiment
    :param qubits: All qubits that were tomographized. This specifies the order in
        which qubits will be kron'ed together; the first qubit in the list is the left-most
        tensor factor.
    :param epsilon: the dilution parameter used in [DIMLE1]_. In practice epsilon ~ 1/num_shots
    :param entropy_penalty: the entropy penalty parameter from [DIMLE2]_, i.e. lambda
    :param beta: The Hedging parameter from [HMLE]_, i.e. beta
    :param tol: The largest difference in the Frobenious norm between update steps that will cause
         the algorithm to conclude that it has converged.
    :param maxiter: The maximum number of iterations to perform before aborting the procedure.
    :return: A point estimate of the quantum state rho
    """
    # exp_type is vanilla by default, max_ent if entropy_penalty > 0, and hedged if beta > 0
    if (entropy_penalty != 0.0) and (beta != 0.0):
        raise ValueError("One can't sensibly do entropy penalty and hedging. Do one or the other"
                         " but not both.")

    # state2matrix and pauli2matrix use pyquil tensor factor ordering where the least significant
    # qubit, e.g. qubit 0, is the right-most tensor factor. We stick with the standard convention
    # here that the first qubit in the list is the left-most tensor factor, so we have to reverse
    # the qubits before passing to state2matrix and pauli2matrix
    qs = qubits[::-1]

    # Identity prop to the size of Hilbert space
    dim = 2**len(qubits)
    IdH = np.eye(dim, dim) # Identity prop to the size of Hilbert space
    num_meas = sum([res.total_counts for res in results])

    rho = IdH / dim
    iteration = 1
    while True:
        rho_temp = rho
        if iteration >= maxiter:
            warnings.warn('Maximum number of iterations reached before convergence.')
            break
        # Vanilla Iterative MLE
        R = _R(rho, results, qs)
        Tk = R - IdH  # Eq 6 of [DIMLE2] with \lambda = 0.

        # MaxENT Iterative MLE
        if entropy_penalty > 0.0:
            constraint = logm(rho) - IdH * np.trace(rho @ logm(rho))
            Tk -= entropy_penalty * constraint  # Eq 6 of [DIMLE2] with \lambda \neq 0.

        # Hedged Iterative MLE
        if beta > 0.0:
            # delta in equation (1.5.13) of [IHMLE]
            Tk *= num_meas / 2
            Tk += beta * (pinv(rho) - dim * IdH) / 2

        # compute iterative estimate of rho     
        update_map = (IdH + epsilon * Tk)
        rho = update_map @ rho @ update_map
        rho /= np.trace(rho)  # Eq 5 of [DIMLE2].
        if np.linalg.norm(rho - rho_temp, FRO) < tol:
            break
        iteration += 1

    return rho


def _R(state, results, qubits):
    r"""
    This implements Eqn 4 in [DIMLE1]_

    As stated in [DIMLE1]_ eqn 4 reads

    .. math::

        R(\rho) = (1/N) \sum_j (f_j/Pr_j) \Pi_j

    where
    :math:`N` is the total number of measurements,
    :math:`f_j` is the number of times j'th outcome was observed,
    :math:`\Pi_j` is the measurement operator or projector, :math:`\sum_j \Pi_j = Id` and
    :math:`\Pi_j \geq 0`,
    and where :math:`Pr_j = Tr[\Pi_j \rho]` (up to some normalization of the :math:`\Pi_j`)

    We are working with results whose observables are elements of the un-normalized Pauli
    basis. Each Pauli :math:`P_j` can be split into projectors onto the plus and minus eigenspaces

    .. math::

        P_k = \Pi_k^+ - \Pi_k^-
        \Pi_k^+ = (I + P_k) / 2
        \Pi_k^- = (I - P_k) / 2

    where each :math:`Pi \geq 0` as required above. Hence for each :math:`P_k` we associate two
    :math:`\Pi_k`, and subsequently two :math:`f_k`. We can express these in terms of
    :math:`Exp[P_k] := exp_k`

    .. math::

        \rm{plus: }  f_k^+ / N = (1 + exp_k) / 2
        \rm{minus: } f_k^- / N = (1 - exp_k) / 2

    We use these :math:`f_k` and :math:`Pi_k` to arrive at the code below.

    Finally, since our Pauli's are not normalized, i.e. :math:`Pi_k^+ + Pi_k^- = Id`, in order to
    enforce the condition  :math:`\sum_j Pi_j = Id` stated above we need to divide our final answer by
    the number of Paulis.

    :param state: The state (given as a density matrix) that we think we have.
    :param results: Measured results from a state tomography experiment. (assumes Pauli basis)
    :param qubits: Qubits that were tomographized.
    :return: the operator of equation 4 in [DIMLE1]_ which fixes rho by left and right
        multiplication
    """
    # this small number ~ 10^-304 is added so that we don't get divide by zero errors
    machine_eps = np.finfo(float).tiny

    update = np.zeros_like(state, dtype=complex)
    IdH = np.eye(update.shape[0])

    for res in results:
        op_matrix = pauli2matrix(res.setting.observable, qubits)
        meas_exp = res.expectation
        pred_exp = np.trace(op_matrix @ state)

        for sign in [1, -1]:
            f_j_over_n = (1 + sign * meas_exp) / 2
            pr_j = (1 + sign * pred_exp) / 2
            pi_j = (IdH + sign * op_matrix) / 2

            update += f_j_over_n / (pr_j + machine_eps) * pi_j

    return update / len(results)


def state_log_likelihood(state: np.ndarray, results: Iterator[ExperimentResult],
                         qubits: Sequence[int]) -> float:
    """
    The log Likelihood function used in the diluted MLE tomography routine.

    Equation 2 of [DIMLE1]_

    :param state: The state (given as a density matrix) that we think we have.
    :param results: Measured results from a state tomography experiment
    :param qubits: All qubits that were tomographized. This specifies the order in
        which qubits will be kron'ed together; the first qubit in the list is the left-most
        tensor factor. This should agree with the provided state.
    :return: The log likelihood that our state is the one we believe it is.
    """
    # state2matrix and pauli2matrix use pyquil tensor factor ordering where the least significant
    # qubit, e.g. qubit 0, is the right-most tensor factor. We stick with the standard convention
    # here that the first qubit in the list is the left-most tensor factor, so we have to reverse
    # the qubits before passing to state2matrix and pauli2matrix
    qs = qubits[::-1]

    ll = 0
    for res in results:
        n = res.total_counts
        op_matrix = pauli2matrix(res.setting.observable, qs)
        meas_exp = res.expectation
        pred_exp = np.real(np.trace(op_matrix @ state))

        for sign in [1, -1]:
            f_j = n * (1 + sign * meas_exp) / 2
            pr_j = (1 + sign * pred_exp) / 2
            if pr_j <= 0:
                continue
            ll += f_j * np.log10(pr_j)

    return ll


def _resample_expectations_with_beta(results, prior_counts=1):
    """
    Resample expectation values by constructing a beta distribution and sampling from it.

    Used by :py:func:`estimate_variance`.

    :param results: A list of ExperimentResults
    :param prior_counts: Number of "counts" to add to alpha and beta for the beta distribution
        from which we sample.
    :return: A new list of ``results`` where each ExperimentResult's ``expectation`` field
        contained a re-sampled expectation value
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
            std_err=result.std_err,
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
    Use a simple bootstrap-like method to return an error bar on some functional of the
    quantum state.

    :param results: Measured results from a state tomography experiment
    :param qubits: Qubits that were tomographized.
    :param tomo_estimator: takes in ``results, qubits`` and returns a corresponding
        estimate of the state rho, e.g. ``linear_inv_state_estimate``
    :param functional: Which functional to find variance, e.g. ``dm.purity``.
    :param target_state: A density matrix of the state with respect to which the distance
        functional is measured. Not applicable if functional is ``dm.purity``.
    :param n_resamples: The number of times to re-sample.
    :param project_to_physical: Whether to project the estimated state to a physical one
        with :py:func:`project_state_matrix_to_physical`.
    """
    if functional != dm.purity:
        if target_state is None:
            raise ValueError("You're not using the `purity` functional. "
                             "Please specify a target state.")

    sample_estimate = []
    for _ in range(n_resamples):
        resampled_results = _resample_expectations_with_beta(results)
        rho = tomo_estimator(resampled_results, qubits)

        if project_to_physical:
            rho = project_state_matrix_to_physical(rho)

        # Calculate functional of the state
        if functional == dm.purity:
            sample_estimate.append(np.real(dm.purity(rho, dim_renorm=False)))
        else:
            sample_estimate.append(np.real(functional(target_state, rho)))

    return np.mean(sample_estimate), np.var(sample_estimate)


# ==================================================================================================
# PROCESS tomography: estimation methods and helper functions
# ==================================================================================================
def linear_inv_process_estimate(results: List[ExperimentResult], qubits: List[int]) -> np.ndarray:
    """
    Estimate a quantum process using linear inversion.

    This is the simplest process tomography post processing. To use this function,
    collect process tomography data with :py:func:`generate_process_tomography_experiment`
    and :py:func:`~forest.benchmarking.observable_estimation.estimate_observables`.

    For more details on this post-processing technique,
    see https://en.wikipedia.org/wiki/Quantum_tomography#Linear_inversion or
    see section 3.5 of [WOOD]_

    :param results: A tomographically complete list of results.
    :param qubits: All qubits that were tomographized. This specifies the order in
        which qubits will be kron'ed together; the first qubit in the list is the left-most
        tensor factor.
    :return: A point estimate of the quantum process represented by a Choi matrix
    """
    # state2matrix and pauli2matrix use pyquil tensor factor ordering where the least significant
    # qubit, e.g. qubit 0, is the right-most tensor factor. We stick with the standard convention
    # here that the first qubit in the list is the left-most tensor factor, so we have to reverse
    # the qubits before passing to state2matrix and pauli2matrix
    qs = qubits[::-1]
    measurement_matrix = np.vstack([
        vec(np.kron(state2matrix(result.setting.in_state, qs).conj(),
                    pauli2matrix(result.setting.observable, qs))).conj().T
        for result in results
    ])
    expectations = np.array([result.expectation for result in results])
    rho = pinv(measurement_matrix) @ expectations
    # add in identity term
    dim = 2 ** len(qubits)
    return unvec(rho) + np.eye(dim**2) / dim


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
        # 'lift' the result's ExperimentSetting input TensorProductState to the corresponding
        # matrix. This is simply the density matrix of the state that was prepared.
        in_state_matrix = state2matrix(result.setting.in_state, qubits=qubits)
        # 'lift' the result's ExperimentSetting output PauliTerm to the corresponding matrix.
        operator = pauli2matrix(result.setting.observable, qubits=qubits)
        proj_plus = (np.eye(2 ** len(qubits)) + operator) / 2
        proj_minus = (np.eye(2 ** len(qubits)) - operator) / 2

        # Constructing A per eq. (A1) of [PGD]
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
    Provide an estimate of the process via Projected Gradient Descent with Backtracking [PGD]_.

    .. [PGD] Maximum-likelihood quantum process tomography via projected gradient descent.
        Knee et al.
        Phys. Rev. A 98, 062336 (2018).
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
        update = proj_choi_to_physical(est - gradient / mu, trace_preserving) - est

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
    version of equation 3 of [PGD]_.

    See the appendix of [PGD]_.

    :param A: a matrix constructed from the input states and POVM elements (eq. A1) that aids
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
    Computes the gradient of the cost, leveraging the vectorized equation 6 of [PGD]_ given in the
    appendix.

    :param A: a matrix constructed from the input states and POVM elements (eq. A1) that aids
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


def do_tomography(qc: QuantumComputer, program: Program, qubits: List[int], kind: str,
                  num_shots: int = 1_000, active_reset: bool = False,
                  group_tpb_settings: bool = True, symm_type: int = -1,
                  calibrate_observables: bool = True, show_progress_bar: bool = False) \
        -> Tuple[np.ndarray, ObservablesExperiment, List[ExperimentResult]]:
    """
    A wrapper around experiment generation, data acquisition, and estimation that runs a tomography
    experiment and returns the state or process estimate along with the experiment and results.

    :param qc: A quantum computer object on which the experiment will run.
    :param program: A program that either constructs the state or defines the process to be
        estimated, depending on whether ``kind`` is 'state' or 'process' respectively.
    :param qubits: The qubits on which the estimated state or process are supported. This can be a
        superset of the qubits used in ``program``, in which case it is assumed the identity
        acts on these extra qubits. Note that we assume qubits are initialized to the `|0>` state.
    :param kind: A string describing the kind of tomography to do ('state' or 'process')
    :param num_shots: The number of shots to run for each experiment setting.
    :param active_reset: Boolean flag indicating whether experiments should begin with an
        active reset instruction (this can make the collection of experiments run a lot faster).
    :param group_tpb_settings: if true, compatible settings will be formed into groups that can
        be estimated concurrently from the same shot data. This will speed up the data
        acquisition time by reducing the total number of runs, but be aware that grouped settings
        will have non-zero covariance.
    :param symm_type: the type of symmetrization

        * -1 -- exhaustive symmetrization uses every possible combination of flips
        * 0 -- no symmetrization
        * 1 -- symmetrization using an OA with strength 1
        * 2 -- symmetrization using an OA with strength 2
        * 3 -- symmetrization using an OA with strength 3

    :param calibrate_observables: boolean flag indicating whether observable estimates are
        calibrated using the same level of symmetrization as exhaustive_symmetrization.
        Likely, for the best (although slowest) results, symmetrization type should accommodate the
        maximum weight of any observable estimated.
    :param show_progress_bar: displays a progress bar via tqdm if true.
    :return: The estimated state prepared by or process represented by the input ``program``,
        as implemented on the provided ``qc``, along with the experiment and corresponding
        results.
    """
    if kind.lower() == 'state':
        expt = generate_state_tomography_experiment(program, qubits)
    elif kind.lower() == 'process':
        expt = generate_process_tomography_experiment(program, qubits)
    else:
        raise ValueError('Kind must be either \'state\' or \'process\'.')

    if group_tpb_settings:
        expt = group_settings(expt)

    results = list(acquire_dfe_data(qc, expt, num_shots, active_reset=active_reset,
                                    symm_type=symm_type,
                                    calibrate_observables=calibrate_observables,
                                    show_progress_bar=show_progress_bar))

    if kind.lower() == 'state':
        # estimate the state matrix
        est = iterative_mle_state_estimate(results, qubits)
    else:
        # estimate the process represented by a choi matrix
        est = pgdb_process_estimate(results, qubits)

    return est, expt, results
