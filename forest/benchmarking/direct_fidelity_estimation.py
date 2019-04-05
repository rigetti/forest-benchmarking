import functools
import itertools
from operator import mul
from typing import List, Sequence

import numpy as np
from dataclasses import dataclass

from pyquil import Program
from pyquil.api import BenchmarkConnection, QuantumComputer
from pyquil.operator_estimation import ExperimentResult, ExperimentSetting, TomographyExperiment, \
    TensorProductState, measure_observables, plusX, minusX, plusY, minusY, plusZ, minusZ
from pyquil.paulis import PauliTerm, sI, sX, sY, sZ


@dataclass
class DFEData:
    """Experimental data from a DFE experiment"""

    results: List[ExperimentResult]
    """The experimental results"""

    in_states: List[str]
    """The input tensor product states being acted on by the `program`"""

    program: Program
    """The pyquil Program DFE data refers to"""

    out_pauli: List[str]
    """The expected output Pauli operators after the program acts on the corresponding `in_pauli`"""

    pauli_point_est: np.ndarray
    """Point estimate of Pauli operators"""

    pauli_std_err: np.ndarray
    """Estimate of std error in the point estimate"""

    cal_point_est: np.ndarray
    """Point estimate of readout calibration for Pauli operators"""

    cal_std_err: np.ndarray
    """Estimate of std error in the point estimate of readout calibration for Pauli operators"""

    dimension: int
    """Dimension of the Hilbert space"""

    qubits: List[int]
    """qubits involved in the experiment"""


@dataclass
class DFEEstimate:
    """State/Process estimates from DFE experiments"""

    dimension: int
    """Dimension of the Hilbert space"""

    qubits: List[int]
    """qubits involved in the experiment"""

    fid_point_est: float
    """Point estimate of fidelity between ideal gate or state and measured, rescaled by the calibration."""

    fid_std_err: float
    """Standard error of the fidelity point estimate, including the calibration."""


def _state_to_pauli(state: TensorProductState) -> PauliTerm:
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


def _exhaustive_dfe(program: Program, qubits: Sequence[int], in_states,
                    benchmarker: BenchmarkConnection) -> ExperimentSetting:
    """Yield experiments over itertools.product(in_paulis).

    Used as a helper function for generate_exhaustive_xxx_dfe_experiment routines.

    :param program: A program comprised of clifford gates
    :param qubits: The qubits to perform DFE on. This can be a superset of the qubits
        used in ``program``.
    :param in_states: Use these single-qubit Pauli operators in every itertools.product()
        to generate an exhaustive list of DFE experiments.
    :return: experiment setting iterator
    :rtype: ``ExperimentSetting``
    """
    n_qubits = len(qubits)
    for i_states in itertools.product(in_states, repeat=n_qubits):
        i_st = functools.reduce(mul, (op(q) for op, q in zip(i_states, qubits) if op is not None),
                                TensorProductState())

        if len(i_st) == 0:
            continue

        yield ExperimentSetting(
            in_state=i_st,
            out_operator=benchmarker.apply_clifford_to_pauli(program, _state_to_pauli(i_st)),
        )


def generate_exhaustive_process_dfe_experiment(program: Program, qubits: list,
                                               benchmarker: BenchmarkConnection) -> TomographyExperiment:
    """
    Estimate process fidelity by exhaustive direct fidelity estimation (DFE).

    This leads to a quadratic reduction in overhead w.r.t. process tomography for
    fidelity estimation.

    The algorithm is due to:

    [DFE1]  Practical Characterization of Quantum Devices without Tomography
            Silva et al.,
            PRL 107, 210404 (2011)
            https://doi.org/10.1103/PhysRevLett.107.210404
            https://arxiv.org/abs/1104.3835

    [DFE2]  Direct Fidelity Estimation from Few Pauli Measurements
            Flammia et al.,
            PRL 106, 230501 (2011)
            https://doi.org/10.1103/PhysRevLett.106.230501
            https://arxiv.org/abs/1104.4695

    :param program: A program comprised of Clifford group gates that defines the process for
        which we estimate the fidelity.
    :param qubits: The qubits to perform DFE on. This can be a superset of the qubits
        used in ``program``.
    :param benchmarker: A ``BecnhmarkConnection`` object to be used in experiment design
    :return: a set of experiments
    :rtype: ``TomographyExperiment`
    """
    expr = TomographyExperiment(list(
        _exhaustive_dfe(program=program,
                        qubits=qubits,
                        in_states=[None, plusX, minusX, plusY, minusY, plusZ, minusZ],
                        benchmarker=benchmarker)),
        program=program, qubits=qubits)
    return expr


def generate_exhaustive_state_dfe_experiment(program: Program, qubits: list,
                                             benchmarker: BenchmarkConnection) -> TomographyExperiment:
    """
    Estimate state fidelity by exhaustive direct fidelity estimation.

    This leads to a quadratic reduction in overhead w.r.t. state tomography for
    fidelity estimation.

    The algorithm is due to:

    [DFE1]  Practical Characterization of Quantum Devices without Tomography
            Silva et al.,
            PRL 107, 210404 (2011)
            https://doi.org/10.1103/PhysRevLett.107.210404
            https://arxiv.org/abs/1104.3835

    [DFE2]  Direct Fidelity Estimation from Few Pauli Measurements
            Flammia et al.,
            PRL 106, 230501 (2011)
            https://doi.org/10.1103/PhysRevLett.106.230501
            https://arxiv.org/abs/1104.4695

    :param program: A program comprised of Clifford group gates that constructs a state
        for which we estimate the fidelity.
    :param qubits: The qubits to perform DFE on. This can be a superset of the qubits
        used in ``program``.
    :param benchmarker: A ``BecnhmarkConnection`` object to be used in experiment design
    :return: a set of experiments
    :rtype: ``TomographyExperiment`
    """
    expr = TomographyExperiment(list(
        _exhaustive_dfe(program=program,
                        qubits=qubits,
                        in_states=[None, plusZ],
                        benchmarker=benchmarker)),
        program=program, qubits=qubits)
    return expr


def _monte_carlo_dfe(program: Program, qubits: Sequence[int], in_states: list, n_terms: int,
                     benchmarker: BenchmarkConnection) -> ExperimentSetting:
    """Yield experiments over itertools.product(in_paulis).

    Used as a helper function for generate_monte_carlo_xxx_dfe_experiment routines.

    :param program: A program comprised of clifford gates
    :param qubits: The qubits to perform DFE on. This can be a superset of the qubits
        used in ``program``.
    :param in_states: Use these single-qubit Pauli operators in every itertools.product()
        to generate an exhaustive list of DFE experiments.
    :param n_terms: Number of preparation and measurement settings to be chosen at random
    :return: experiment setting iterator
    :rtype: ``ExperimentSetting``
    """
    all_st_inds = np.random.randint(len(in_states), size=(n_terms, len(qubits)))
    for st_inds in all_st_inds:
        i_st = functools.reduce(mul, (in_states[si](qubits[i])
                                      for i, si in enumerate(st_inds)
                                      if in_states[si] is not None), TensorProductState())

        while len(i_st) == 0:
            # pick a new one
            second_try_st_inds = np.random.randint(len(in_states), size=len(qubits))
            i_st = functools.reduce(mul, (in_states[si](qubits[i])
                                          for i, si in enumerate(second_try_st_inds)
                                          if in_states[si] is not None), TensorProductState())

        yield ExperimentSetting(
            in_state=i_st,
            out_operator=benchmarker.apply_clifford_to_pauli(program, _state_to_pauli(i_st)),
        )


def generate_monte_carlo_state_dfe_experiment(program: Program, qubits: List[int], benchmarker: BenchmarkConnection,
                                              n_terms=200) -> TomographyExperiment:
    """
    Estimate state fidelity by sampled direct fidelity estimation.

    This leads to constant overhead (w.r.t. number of qubits) fidelity estimation.

    The algorithm is due to:

    [DFE1]  Practical Characterization of Quantum Devices without Tomography
            Silva et al., PRL 107, 210404 (2011)
            https://doi.org/10.1103/PhysRevLett.107.210404

    [DFE2]  Direct Fidelity Estimation from Few Pauli Measurements
            Flammia and Liu, PRL 106, 230501 (2011)
            https://doi.org/10.1103/PhysRevLett.106.230501

    :param program: A program comprised of clifford gates that constructs a state
        for which we estimate the fidelity.
    :param qubits: The qubits to perform DFE on. This can be a superset of the qubits
        used in ``program``.
    :param benchmarker: The `BenchmarkConnection` object used to design experiments
    :param n_terms: Number of randomly chosen observables to measure. This number should be 
        a constant less than ``2**len(qubits)``, otherwise ``exhaustive_state_dfe`` is more efficient.
    :return: a set of experiments
    :rtype: ``TomographyExperiment`
    """
    expr = TomographyExperiment(list(
        _monte_carlo_dfe(program=program, qubits=qubits,
                         in_states=[None, plusZ],
                         n_terms=n_terms, benchmarker=benchmarker)),
        program=program, qubits=qubits)
    return expr


def generate_monte_carlo_process_dfe_experiment(program: Program, qubits: List[int], benchmarker: BenchmarkConnection,
                                                n_terms: int = 200) -> TomographyExperiment:
    """
    Estimate process fidelity by randomly sampled direct fidelity estimation.

    This leads to constant overhead (w.r.t. number of qubits) fidelity estimation.

    The algorithm is due to:

    [DFE1]  Practical Characterization of Quantum Devices without Tomography
            Silva et al., PRL 107, 210404 (2011)
            https://doi.org/10.1103/PhysRevLett.107.210404

    [DFE2]  Direct Fidelity Estimation from Few Pauli Measurements
            Flammia and Liu, PRL 106, 230501 (2011)
            https://doi.org/10.1103/PhysRevLett.106.230501

    :param program: A program comprised of Clifford group gates that constructs a state
        for which we estimate the fidelity.
    :param qubits: The qubits to perform DFE on. This can be a superset of the qubits
        used in ``program``.
    :param benchmarker: The `BenchmarkConnection` object used to design experiments
    :param n_terms: Number of randomly chosen observables to measure. This number should be 
        a constant less than ``2**len(qubits)``, otherwise ``exhaustive_process_dfe`` is more efficient.
    :return: a DFE experiment object
    :rtype: ``DFEExperiment`
    """
    expr = TomographyExperiment(list(
        _monte_carlo_dfe(program=program, qubits=qubits,
                         in_states=[None, plusX, minusX, plusY, minusY, plusZ, minusZ],
                         n_terms=n_terms, benchmarker=benchmarker)),
        program=program, qubits=qubits)
    return expr


def acquire_dfe_data(qc: QuantumComputer, expr: TomographyExperiment, n_shots=10_000, active_reset=False,
                     mitigate_readout_errors=True) -> DFEData:
    """
    Acquire data necessary for direct fidelity estimate (DFE).

    :param qc: A quantum computer object where the experiment will run.
    :param expr: A partial tomography(``TomographyExperiment``) object describing the experiments to be run.
    :param n_shots: The minimum number of shots to be taken in each experiment (including calibration).
    :param active_reset: Boolean flag indicating whether experiments should terminate with an active reset instruction
        (this can make experiments a lot faster).
    :param mitigate_readout_errors: Boolean flag indicating whether bias due to imperfect readout should be corrected
        for
    :return: a DFE data object
    :rtype: ``DFEData`
    """
    if mitigate_readout_errors:
        res = list(measure_observables(qc, expr, n_shots=n_shots, active_reset=active_reset))
    else:
        res = list(measure_observables(qc, expr, n_shots=n_shots, active_reset=active_reset, readout_symmetrize=None,
                                       calibrate_readout=None))

    return DFEData(results=res,
                   in_states=[str(e[0].in_state) for e in expr],
                   program=expr.program,
                   out_pauli=[str(e[0].out_operator) for e in expr],
                   pauli_point_est=np.array([r.expectation for r in res]),
                   pauli_std_err=np.array([r.stddev for r in res]),
                   cal_point_est=np.array([r.calibration_expectation for r in res]),
                   cal_std_err=np.array([r.calibration_stddev for r in res]),
                   dimension=2**len(expr.qubits),
                   qubits=expr.qubits)


def estimate_dfe(data: DFEData, kind: str) -> DFEEstimate:
    """
    Analyse data from experiments to obtain a direct fidelity estimate (DFE).

    State fidelity between the experimental state σ and the ideal (pure) state ρ (both states represented by density
    matrices) is defined as F(σ,ρ) = tr σ⋅ρ [Joz].

    The direct fidelity estimate for a state is given by the average expected value of the Pauli operators in the
    stabilizer group of the ideal pure state (i.e., Eqn. 1 of [DFE1]).

    The average gate fidelity between the experimental process ℰ and the ideal (unitary) process 𝒰 is defined as
    F(ℰ,𝒰) = (tr ℰ 𝒰⁺ + d)/(d^2+d) where the processes are represented by linear superoperators acting of vectorized
    density matrices, and d is the dimension of the Hilbert space ℰ and 𝒰 act on (where ⁺ represents the Hermitian
    conjugate). See [Nie] for details.

    The average gate fidelity can be re-written a F(ℰ,𝒰)=(d^2 tr J(ℰ)⋅J(𝒰) + d)/(d^2+d) where J() is the
    Choi-Jamiolkoski representation of the superoperator in the argument. Since the Choi-Jamiolkowski representation
    is given by a density operator, the connection to the calculation of state fidelity becomes apparent:
    F(J(ℰ),J(𝒰)) = tr J(ℰ)⋅J(𝒰) is the state fidelity between Choi-Jamiolkoski states.

    Noting that the Choi-Jamiolkoski state is prepared by acting on half of a maximally entangled state with the
    process in question, the direct fidelity estimate of the Choi-Jamiolkoski state is given by the average expected
    value of a Pauli operator resulting from applying the ideal unitary 𝒰 to a Pauli operator Pᵢ, for the state
    resulting from applying the ideal unitary to a stabilizer state that has Pᵢ in its stabilizer group (one must be
    careful to prepare states that have both +1 and -1 eigenstates of the operator in question, to emulate the random
    state preparation corresponding to measuring half of a maximally entangled state).

    [Joz]  Fidelity for Mixed Quantum States
           Jozsa, Journal of Modern Optics, 41:12, 2315-2323 (1994)
           DOI: 10.1080/09500349414552171
           https://doi.org/10.1080/09500349414552171

    [DFE1]  Practical Characterization of Quantum Devices without Tomography
            Silva et al., PRL 107, 210404 (2011)
            https://doi.org/10.1103/PhysRevLett.107.210404
            https://arxiv.org/abs/1104.3835

    [Nie]  A simple formula for the average gate fidelity of a quantum dynamical operation
           Nielsen, Phys. Lett. A 303 (4): 249-252 (2002)
           DOI: 10.1016/S0375-9601(02)01272-0
           https://doi.org/10.1016/S0375-9601(02)01272-0
           https://arxiv.org/abs/quant-ph/0205035

    :param data: A ``DFEData`` object containing raw experimental results.
    :param kind: A string describing the kind of DFE data being analysed ('state' or 'process')
    :return: a DFE estimate object
    :rtype: ``DFEEstimate`

    """
    d = data.dimension

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
    # by d-1/d (for states) or d**2-1/d**2 (for processes). Similarly, variance estimates must be scaled appropriately.

    if kind == 'state':
        # introduce bias due to measuring the identity
        mean_est = (d-1)/d * np.mean(data.pauli_point_est) + 1.0/d
        var_est = (d-1)**2/d**2 * np.sum(data.pauli_std_err**2) / len(data.pauli_point_est) ** 2
    elif kind == 'process':
        # introduce bias due to measuring the identity
        p_mean = (d**2-1)/d**2 * np.mean(data.pauli_point_est) + 1.0/d**2
        mean_est = (d**2 * p_mean + d)/(d**2+d)
        var_est = d**2/(d+1)**2 * (d**2-1)**2/d**4 * np.sum(data.pauli_std_err**2) / len(data.pauli_point_est) ** 2
    else:
        raise ValueError('DFEdata can only be of kind \'state\' or \'process\'.')

    return DFEEstimate(dimension=data.dimension,
                       qubits=data.qubits,
                       fid_point_est=mean_est,
                       fid_std_err=np.sqrt(var_est))
