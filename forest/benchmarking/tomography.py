import functools
import itertools
from operator import mul
from typing import Callable, Tuple, List, Sequence
import warnings

import numpy as np
from scipy.linalg import logm, pinv, eigh

import forest.benchmarking.distance_measures as dm
from forest.benchmarking.superoperator_tools import vec, unvec, proj_choi_to_physical
from forest.benchmarking.utils import n_qubit_pauli_basis
from pyquil import Program
from pyquil.operator_estimation import ExperimentSetting, \
    TomographyExperiment as PyQuilTomographyExperiment, ExperimentResult, SIC0, SIC1, SIC2, SIC3, \
    plusX, minusX, plusY, minusY, plusZ, minusZ, TensorProductState, zeros_state
from pyquil.paulis import sI, sX, sY, sZ, PauliSum, PauliTerm, is_identity
from pyquil.unitary_tools import lifted_pauli as pauli2matrix, lifted_state_operator as state2matrix

MAXITER = "maxiter"
OPTIMAL = "optimal"
FRO = 'fro'


# ==================================================================================================
# Generate state and process tomography experiments
# ==================================================================================================
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
                                      program=program)


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


def generate_process_tomography_experiment(program: Program, qubits: List[int], in_basis='pauli'):
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

    return PyQuilTomographyExperiment(settings=list(func(qubits)), program=program)


# ==================================================================================================
# STATE tomography: estimation methods and helper functions
# ==================================================================================================

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
        vec(pauli2matrix(result.setting.out_operator, qubits=qubits)).T.conj()
        for result in results
    ])
    expectations = np.array([result.expectation for result in results])
    rho = pinv(measurement_matrix) @ expectations
    return unvec(rho)


def iterative_mle_state_estimate(results: List[ExperimentResult], qubits: List[int], epsilon=.1,
                                 entropy_penalty=0.0, beta=0.0, tol=1e-9, maxiter=10_000) \
        -> np.ndarray:
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

    :param results: Measured results from a state tomography experiment
    :param qubits: Qubits that were tomographized.
    :param epsilon: the dilution parameter used in [DIMLE1]. In practice epsilon ~ 1/num_shots
    :param entropy_penalty: the entropy penalty parameter from [DIMLE2], i.e. lambda
    :param beta: The Hedging parameter from [HMLE], i.e. beta
    :param tol: The largest difference in the frobenious norm between update steps that will cause
         the algorithm to conclude that it has converged.
    :param maxiter: The maximum number of iterations to perform before aborting the procedure.
    :return: A point estimate of the quantum state rho
    """
    # exp_type is vanilla by default, max_ent if entropy_penalty > 0, and hedged if beta > 0
    if (entropy_penalty != 0.0) and (beta != 0.0):
        raise ValueError("One can't sensibly do entropy penalty and hedging. Do one or the other"
                         " but not both.")

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
        R = _R(rho, results, qubits)
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
    This implements Eqn 4 in [DIMLE1]

    As stated in [DIMLE1] eqn 4 reads

    R(rho) = (1/N) \sum_j (f_j/Pr_j) Pi_j

    N = total number of measurements
    f_j = number of times j'th outcome was observed
    Pi_j = measurement operator or projector, with \sum_j Pi_j = Id and Pi_j \geq 0
    Pr_j = Tr[Pi_j \rho]  (up to some normalization of the Pi_j)

    We are working with results whose out_operators are elements of the un-normalized Pauli
    basis. Each Pauli P_j can be split into projectors onto the plus and minus eigenspaces
        P_k = Pi_k^+ - Pi_k^-   ;   Pi_k^+ = (I + P_k) / 2   ;   Pi_k^- = (I - P_k) / 2
    where each Pi \geq 0 as required above. Hence for each P_k we associate two Pi_k,
    and subsequently two f_k. We can express these in terms of Exp[P_k] := exp_k
        plus: f_k^+ / N = (1 + exp_k) / 2
        minus: f_k^- / N = (1 - exp_k) / 2

    We use these f_k and Pi_k to arrive at the code below.

    Finally, since our Pauli's are not normalized, i.e. Pi_k^+ + Pi_k^- = Id, in order to enforce
    the condition  \sum_j Pi_j = Id stated above we need to divide our final answer by the number
    of Paulis.

    :param state: The state (given as a density matrix) that we think we have.
    :param results: Measured results from a state tomography experiment. (assumes Pauli basis)
    :param qubits: Qubits that were tomographized.
    :return: the operator of equation 4 in [DIMLE1] which fixes rho by left and right multiplication
    """
    # this small number ~ 10^-304 is added so that we don't get divide by zero errors
    machine_eps = np.finfo(float).tiny

    update = np.zeros_like(state, dtype=complex)
    IdH = np.eye(update.shape[0])

    for res in results:
        op_matrix = pauli2matrix(res.setting.out_operator, qubits)
        meas_exp = res.expectation
        pred_exp = np.trace(op_matrix @ state)

        for sign in [1, -1]:
            f_j_over_n = (1 + sign * meas_exp) / 2
            pr_j = (1 + sign * pred_exp) / 2
            pi_j = (IdH + sign * op_matrix) / 2

            update += f_j_over_n / (pr_j + machine_eps) * pi_j

    return update / len(results)


def state_log_likelihood(state, results, qubits) -> float:
    """
    The log Likelihood function used in the diluted MLE tomography routine.

    Equation 2 of [DIMLE1]

    :param state: The state (given as a density matrix) that we think we have.
    :param results: Measured results from a state tomography experiment
    :param qubits: Qubits that were tomographized.
    :return: The log likelihood that our state is the one we believe it is.
    """
    ll = 0
    for res in results:
        n = res.total_counts
        op_matrix = pauli2matrix(res.setting.out_operator, qubits)
        meas_exp = res.expectation
        pred_exp = np.real(np.trace(op_matrix @ state))

        for sign in [1, -1]:
            f_j = n * (1 + sign * meas_exp) / 2
            pr_j = (1 + sign * pred_exp) / 2
            if pr_j <= 0:
                continue
            ll += f_j * np.log10(pr_j)

    return ll


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
        rho = tomo_estimator(resampled_results, qubits)

        if project_to_physical:
            rho = project_density_matrix(rho)

        # Calculate functional of the state
        if functional == dm.purity:
            sample_estimate.append(np.real(dm.purity(rho, dim_renorm=False)))
        else:
            sample_estimate.append(np.real(functional(target_state, rho)))

    return np.mean(sample_estimate), np.var(sample_estimate)


# ==================================================================================================
# PROCESS tomography: estimation methods and helper functions
# ==================================================================================================

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
        operator = pauli2matrix(result.setting.out_operator, qubits=qubits)
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
    version of equation 3 of [PGD].

    See the appendix of [PGD].

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
    Computes the gradient of the cost, leveraging the vectorized equation 6 of [PGD] given in the
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
