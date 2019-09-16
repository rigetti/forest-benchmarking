import functools
import itertools
from operator import mul
from typing import List, Sequence, Iterable, Tuple

import numpy as np

from pyquil import Program
from pyquil.api import BenchmarkConnection, QuantumComputer
from forest.benchmarking.observable_estimation import ExperimentResult, ExperimentSetting, \
    ObservablesExperiment, TensorProductState, estimate_observables, plusX, minusX, plusY, minusY,\
    plusZ, minusZ, calibrate_observable_estimates, group_settings
from pyquil.paulis import PauliTerm, sI, sX, sY, sZ


def _state_to_pauli(state: TensorProductState) -> PauliTerm:
    """
    Converts a TensorProductState that is the eigenstate of some Pauli operator into its
    associated PauliTerm.

    The overall sign of the returned Pauli is the sign of the eigenvalue for the state.

    .. note:: exactly the qubits explicitly referenced in the `state` will be considered,
        so be careful distinguishing between an explicit ``|0>`` (+Z) state and an implicit ``|0>``
        state which is ignored.

    :param state: the state to convert
    :return: the associated Pauli represented as a :class:`pyquil.paulis.PauliTerm`.
    """
    term = sI()
    for oneq_st in state.states:
        if oneq_st.label == 'X':
            term *= sX(oneq_st.qubit)
        elif oneq_st.label == 'Y':
            term *= sY(oneq_st.qubit)
        elif oneq_st.label == 'Z':
            term *= sZ(oneq_st.qubit)
        else:
            raise ValueError(f"Can't convert state {state} to a PauliTerm")

        if oneq_st.index == 1:
            term *= -1
    return term


def _exhaustive_dfe(benchmarker: BenchmarkConnection, program: Program, qubits: Sequence[int],
                    in_states) -> Iterable[ExperimentSetting]:
    """
    Yield experiments over itertools.product(in_states).

    Used as a helper function for generate_exhaustive_xxx_dfe_experiment routines.

    :param benchmarker: object returned from pyquil.api.get_benchmarker() used to conjugate each 
        Pauli by the Clifford program
    :param program: A program comprised of Clifford gates
    :param qubits: The qubits to perform DFE on. This can be a superset of the qubits
        used in ``program``, in which case it is assumed the identity acts on these qubits. 
        Note that we assume qubits are initialized to the ``|0>`` state.
    :param in_states: Use these single-qubit Pauli operators in every itertools.product()
        to generate an exhaustive list of DFE experiments.
    :return: experiment setting iterator for exhaustive dfe settings
    """
    n_qubits = len(qubits)
    for i_states in itertools.product(in_states, repeat=n_qubits):
        # distinguish between a Z eigenstate and None
        i_st = functools.reduce(mul, (op(q) for op, q in zip(i_states, qubits) if op is not None),
                                TensorProductState())

        # explicitly initialize the in_state with None set to zero, i.e. plus Z eigenstate.
        in_state_with_zeros =  functools.reduce(mul,
                                                (plusZ(q) if op is None else op(q)
                                                 for op, q in zip(i_states, qubits)),
                                TensorProductState())

        if len(i_st) == 0:
            continue

        yield ExperimentSetting(
            in_state=in_state_with_zeros,
            observable=benchmarker.apply_clifford_to_pauli(program, _state_to_pauli(i_st)),
        )


def generate_exhaustive_process_dfe_experiment(benchmarker: BenchmarkConnection, program: Program,
                                               qubits: list) -> ObservablesExperiment:
    """
    Estimate process fidelity by exhaustive direct fidelity estimation (DFE).

    This leads to a quadratic reduction in overhead w.r.t. process tomography for
    fidelity estimation.

    The algorithm is due to:

    .. [DFE1]  Practical Characterization of Quantum Devices without Tomography.
            Silva et al.
            PRL 107, 210404 (2011).
            https://doi.org/10.1103/PhysRevLett.107.210404
            https://arxiv.org/abs/1104.3835

    .. [DFE2]  Direct Fidelity Estimation from Few Pauli Measurements.
            Flammia and Liu.
            PRL 106, 230501 (2011).
            https://doi.org/10.1103/PhysRevLett.106.230501
            https://arxiv.org/abs/1104.4695

    :param benchmarker: object returned from pyquil.api.get_benchmarker() used to conjugate each 
        Pauli by the Clifford program
    :param program: A program comprised of Clifford group gates that defines the process for
        which we estimate the fidelity.
    :param qubits: The qubits to perform DFE on. This can be a superset of the qubits
        used in ``program``, in which case it is assumed the identity acts on these qubits. 
        Note that we assume qubits are initialized to the ``|0>`` state.
    :return: an ObservablesExperiment that constitutes a process DFE experiment.
    """
    expt = ObservablesExperiment(
        list(_exhaustive_dfe(benchmarker=benchmarker, program=program, qubits=qubits,
                             in_states=[None, plusX, minusX, plusY, minusY, plusZ, minusZ])),
        program=program)
    return expt


def generate_exhaustive_state_dfe_experiment(benchmarker: BenchmarkConnection, program: Program,
                                             qubits: list) -> ObservablesExperiment:
    """
    Estimate state fidelity by exhaustive direct fidelity estimation.

    This leads to a quadratic reduction in overhead w.r.t. state tomography for
    fidelity estimation.

    The algorithm is due to [DFE1]_ and [DFE2]_.

    :param benchmarker: object returned from pyquil.api.get_benchmarker() used to conjugate each 
        Pauli by the Clifford program
    :param program: A program comprised of Clifford group gates that constructs a state
        for which we estimate the fidelity.
    :param qubits: The qubits to perform DFE on. This can be a superset of the qubits
        used in ``program``, in which case it is assumed the identity acts on these qubits. 
        Note that we assume qubits are initialized to the ``|0>`` state.
    :return: an ObservablesExperiment that constitutes a state DFE experiment.
    """
    expt = ObservablesExperiment(
        list(_exhaustive_dfe(benchmarker=benchmarker, program=program, qubits=qubits,
                             in_states=[None, plusZ])),
        program=program)
    return expt


def _monte_carlo_dfe(benchmarker: BenchmarkConnection, program: Program, qubits: Sequence[int],
                     in_states: list, n_terms: int) -> Iterable[ExperimentSetting]:
    """
    Yield experiments over itertools.product(in_paulis).

    Used as a helper function for generate_monte_carlo_xxx_dfe_experiment routines.

    :param benchmarker: object returned from pyquil.api.get_benchmarker() used to conjugate each 
        Pauli by the Clifford program
    :param program: A program comprised of clifford gates
    :param qubits: The qubits to perform DFE on. This can be a superset of the qubits
        used in ``program``, in which case it is assumed the identity acts on these qubits. 
        Note that we assume qubits are initialized to the ``|0>`` state.
    :param in_states: Use these single-qubit Pauli operators in every itertools.product()
        to generate an exhaustive list of DFE experiments.
    :param n_terms: Number of preparation and measurement settings to be chosen at random
    :return: experiment setting iterator
    """
    all_st_inds = np.random.randint(len(in_states), size=(n_terms, len(qubits)))
    for st_inds in all_st_inds:
        # begin loop in case the state ends up being trivial (all chosen states are None)
        while True:
            i_st = functools.reduce(mul, (in_states[si](qubits[i])
                                          for i, si in enumerate(st_inds)
                                          if in_states[si] is not None), TensorProductState())
            if len(i_st) > 0:
                # this choice is not trivial so continue
                break

            # pick new state indices and try again
            st_inds = np.random.randint(len(in_states), size=len(qubits))

        # explicitly initialize the in_state with None set to zero, i.e. plus Z eigenstate.
        in_state_with_zeros =  functools.reduce(mul, (plusZ(qubits[i]) if in_states[si] is None
                                                      else in_states[si](qubits[i])
                                                      for i, si in enumerate(st_inds)),
                                TensorProductState())

        yield ExperimentSetting(
            in_state=in_state_with_zeros,
            observable=benchmarker.apply_clifford_to_pauli(program, _state_to_pauli(i_st)),
        )


def generate_monte_carlo_state_dfe_experiment(benchmarker: BenchmarkConnection, program: Program,
                                              qubits: List[int], n_terms=200) \
        -> ObservablesExperiment:
    """
    Estimate state fidelity by sampled direct fidelity estimation.

    This leads to constant overhead (w.r.t. number of qubits) fidelity estimation.

    The algorithm is due to [DFE1]_ and [DFE2]_.

    :param program: A program comprised of clifford gates that constructs a state
        for which we estimate the fidelity.
    :param qubits: The qubits to perform DFE on. This can be a superset of the qubits
        used in ``program``, in which case it is assumed the identity acts on these qubits. 
        Note that we assume qubits are initialized to the ``|0>`` state.
    :param benchmarker: The `BenchmarkConnection` object used to design experiments
    :param n_terms: Number of randomly chosen observables to measure. This number should be 
        a constant less than ``2**len(qubits)``, otherwise ``exhaustive_state_dfe`` is more efficient.
    :return: an ObservablesExperiment that constitutes a state DFE experiment.
    """
    expt = ObservablesExperiment(
        list(_monte_carlo_dfe(benchmarker=benchmarker, program=program, qubits=qubits,
                              in_states=[None, plusZ], n_terms=n_terms)),
        program=program)
    return expt


def generate_monte_carlo_process_dfe_experiment(benchmarker: BenchmarkConnection, program: Program,
                                                qubits: List[int], n_terms: int = 200) \
        -> ObservablesExperiment:
    """
    Estimate process fidelity by randomly sampled direct fidelity estimation.

    This leads to constant overhead (w.r.t. number of qubits) fidelity estimation.

    The algorithm is due to [DFE1]_ and [DFE2]_.

    :param program: A program comprised of Clifford group gates that constructs a state
        for which we estimate the fidelity.
    :param qubits: The qubits to perform DFE on. This can be a superset of the qubits
        used in ``program``, in which case it is assumed the identity acts on these qubits. 
        Note that we assume qubits are initialized to the ``|0>`` state.
    :param benchmarker: The `BenchmarkConnection` object used to design experiments
    :param n_terms: Number of randomly chosen observables to measure. This number should be 
        a constant less than ``2**len(qubits)``, otherwise ``exhaustive_process_dfe`` is more efficient.
    :return: an ObservablesExperiment that constitutes a process DFE experiment.
    """
    expt = ObservablesExperiment(
        list(_monte_carlo_dfe(benchmarker=benchmarker, program=program, qubits=qubits,
                              in_states=[None, plusX, minusX, plusY, minusY, plusZ, minusZ],
                              n_terms=n_terms)),
        program=program)
    return expt


def acquire_dfe_data(qc: QuantumComputer, expt: ObservablesExperiment, num_shots: int = 10_000,
                     active_reset: bool = False, symm_type: int = -1,
                     calibrate_observables: bool = True,
                     show_progress_bar: bool = False) -> List[ExperimentResult]:
    """
    Acquire data necessary for direct fidelity estimate (DFE).

    :param qc: A quantum computer object where the experiment will run.
    :param expt: An ObservablesExperiment object describing the experiments to be run.
    :param num_shots: The number of shots to be taken in each experiment. If
        calibrate_observables is set to True then this number of shots may be increased.
    :param active_reset: Boolean flag indicating whether experiments should begin with an
        active reset instruction (this can make the collection of experiments run a lot faster).
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
    :return: results from running the given DFE experiment. These can be passed to estimate_dfe
    """
    res = list(estimate_observables(qc, expt, num_shots=num_shots,
                                    symm_type=symm_type,
                                    active_reset=active_reset,
                                    show_progress_bar=show_progress_bar))
    if calibrate_observables:
        res = list(calibrate_observable_estimates(qc, res, num_shots=num_shots,
                                                  symm_type=symm_type, active_reset=active_reset))

    return res


def estimate_dfe(results: List[ExperimentResult], kind: str) -> Tuple[float, float]:
    """
    Analyse data from experiments to obtain a direct fidelity estimate (DFE).

    State fidelity between the experimental state σ and the ideal (pure) state ρ
    (both states represented by density matrices) is defined as F(σ,ρ) = tr σ⋅ρ [Joz]_.

    The direct fidelity estimate for a state is given by the average expected value of the Pauli
    operators in the stabilizer group of the ideal pure state (i.e., Eqn. 1 of [DFE1]_).

    The average gate fidelity between the experimental process ℰ and the ideal (unitary)
    process 𝒰 is defined as F(ℰ,𝒰) = (tr ℰ 𝒰⁺ + d)/(d^2+d) where the processes are represented
    by linear superoperators acting of vectorized density matrices, and d is the dimension of the
    Hilbert space ℰ and 𝒰 act on (and where ⁺ represents the Hermitian conjugate). See [Nie]_ for
    details.

    The average gate fidelity can be re-written a F(ℰ,𝒰)=(d^2 tr J(ℰ)⋅J(𝒰) + d)/(d^2+d) where
    J() is the Choi-Jamiolkoski representation of the superoperator in the argument. Since the
    Choi-Jamiolkowski representation is given by a density operator, the connection to the
    calculation of state fidelity becomes apparent:
    F(J(ℰ),J(𝒰)) = tr J(ℰ)⋅J(𝒰) is the state fidelity between Choi-Jamiolkoski states.

    Noting that the Choi-Jamiolkoski state is prepared by acting on half of a maximally entangled
    state with the process in question, the direct fidelity estimate of the Choi-Jamiolkoski
    state is given by the average expected value of a Pauli operator resulting from applying the
    ideal unitary 𝒰 to a Pauli operator Pᵢ, for the state resulting from applying the ideal
    unitary to a stabilizer state that has Pᵢ in its stabilizer group (one must be careful to
    prepare states that have both +1 and -1 eigenstates of the operator in question, to emulate
    the random state preparation corresponding to measuring half of a maximally entangled state).

    .. [Joz]  Fidelity for Mixed Quantum States.
           Jozsa. Journal of Modern Optics. 41:12, 2315-2323 (1994).
           DOI: 10.1080/09500349414552171
           https://doi.org/10.1080/09500349414552171

    .. [Nie]  A simple formula for the average gate fidelity of a quantum dynamical operation.
           Nielsen. Phys. Lett. A 303 (4): 249-252 (2002).
           https://doi.org/10.1016/S0375-9601(02)01272-0
           https://arxiv.org/abs/quant-ph/0205035

    :param results: A list of ExperimentResults from running a DFE experiment
    :param kind: A string describing the kind of DFE data being analysed ('state' or 'process')
    :return: the estimate of the mean fidelity along with the associated standard err
    """
    # identify the qubits being measured
    qubits = list(functools.reduce(lambda x, y: set(x) | set(y),
                                   [res.setting.observable.get_qubits() for res in results]))

    d = 2**len(qubits)

    # The subtlety in estimating the fidelity from a set of expectations of Pauli operators is that it is essential
    # to include the expectation of the identity in the calculation -- without it the fidelity estimate will be biased
    # downwards.

    # However, there is no need to estimate the expectation of the identity: it is an experiment that always
    # yields 1 as a result, so its expectation is 1. This quantity must be included in the calculation with the proper
    # weight, however. For state fidelity estimation, we should choose the identity to be measured one out of every
    # d times. For process fidelity estimation, we should choose to prepare and measure the identity one out of every
    # d**2 times.

    # The mean expected value for the (non-trivial) Pauli operators that are measured must be scaled
    # as well -- each non-trivial Pauli should be selected 1 in every d or d**2 times (depending on whether we do
    # states or processes), but if we choose Pauli ops uniformly from the d-1 or d**2-1 non-trivial Paulis, we
    # again introduce a bias. So the mean expected value of non-trivial Paulis that are sampled must be weighted
    # by (d-1)/d (for states) or (d**2-1)/d**2 (for processes). Similarly, variance estimates must
    # be scaled appropriately.

    expectations = [res.expectation for res in results]
    std_errs = np.asarray([res.std_err for res in results])

    if kind.lower() == 'state':
        # introduce bias due to measuring the identity
        mean_est = (d-1)/d * np.mean(expectations) + 1.0/d
        var_est = (d-1)**2/d**2 * np.sum(std_errs**2) / len(expectations) ** 2
    elif kind.lower() == 'process':
        # introduce bias due to measuring the identity
        p_mean = (d**2-1)/d**2 * np.mean(expectations) + 1.0/d**2
        mean_est = (d**2 * p_mean + d)/(d**2+d)
        var_est = d**2/(d+1)**2 * (d**2-1)**2/d**4 * np.sum(std_errs**2) / len(expectations) ** 2
    else:
        raise ValueError('Kind can only be \'state\' or \'process\'.')

    return mean_est, np.sqrt(var_est)


def do_dfe(qc: QuantumComputer, benchmarker: BenchmarkConnection, program: Program,
           qubits: List[int], kind: str, mc_n_terms: int = None, num_shots: int = 1_000,
           active_reset: bool = False, group_tpb_settings: bool = False,
           symm_type: int = -1, calibrate_observables: bool = True,
           show_progress_bar: bool =  False) \
        -> Tuple[Tuple[float, float], ObservablesExperiment, List[ExperimentResult]]:
    """
    A wrapper around experiment generation, data acquisition, and estimation that runs a DFE 
    experiment and returns the (fidelity, std_err) pair along with the experiment and results.

    :param qc: A quantum computer object on which the experiment will run.
    :param benchmarker: object returned from pyquil.api.get_benchmarker() used to conjugate each
        Pauli by the Clifford program
    :param program: A program comprised of Clifford group gates that either constructs the
        state or defines the process for which we estimate the fidelity, depending on whether
        ``kind`` is 'state' or 'process' respectively.
    :param qubits: The qubits to perform DFE on. This can be a superset of the qubits
        used in ``program``, in which case it is assumed the identity acts on these qubits. 
        Note that we assume qubits are initialized to the ``|0>`` state.
    :param kind: A string describing the kind of DFE to do ('state' or 'process')
    :param mc_n_terms: Number of randomly chosen observables to measure for Monte Carlo DFE.
        By default, when this is None, we do exhaustive DFE. The number should be a constant less
        than ``2**len(qubits)``, otherwise exhaustive DFE is more efficient.
    :param num_shots: The number of shots to be taken for each experiment setting.
    :param active_reset: Boolean flag indicating whether experiments should begin with an
        active reset instruction (this can make the collection of experiments run a lot faster).
    :param group_tpb_settings: if true, compatible settings will be formed into groups that can
        be estimated concurrently from the same shot data. This will speed up the data
        acquisition time by reducing the total number of runs, but be aware that grouped settings
        will have non-zero covariance. TODO: set default True after handling covariance.
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
    :return: The estimated fidelity of the state prepared by or process represented by the input
        ``program``, as implemented on the provided ``qc``, along with the standard error of the
        estimate. The experiment and corresponding results are also returned.
    """
    if kind.lower() not in ['state', 'process']:
        raise ValueError('Kind must be either \'state\' or \'process\'.')

    if mc_n_terms is None:
        if kind.lower() == 'state':
            expt = generate_exhaustive_state_dfe_experiment(benchmarker, program, qubits)
        else:
            expt = generate_exhaustive_process_dfe_experiment(benchmarker, program, qubits)
    else:
        if kind.lower() == 'state':
            expt = generate_monte_carlo_state_dfe_experiment(benchmarker, program, qubits,
                                                             mc_n_terms)
        else:
            expt = generate_monte_carlo_process_dfe_experiment(benchmarker, program, qubits,
                                                               mc_n_terms)
    if group_tpb_settings:
        expt = group_settings(expt)

    results = list(acquire_dfe_data(qc, expt, num_shots, active_reset=active_reset,
                                    symm_type=symm_type,
                                    calibrate_observables=calibrate_observables,
                                    show_progress_bar=show_progress_bar))

    fid, std_err = estimate_dfe(results, kind)

    return (fid, std_err), expt, results
