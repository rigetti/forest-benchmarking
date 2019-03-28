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

    pauli_point_est: List[float]
    """Point estimate of Pauli operators"""

    pauli_std_err: List[float]
    """Estimate of std error in the point estimate"""

    cal_point_est: List[float]
    """Point estimate of readout calibration for Pauli operators"""

    cal_std_err: List[float]
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


def acquire_dfe_data(qc: QuantumComputer, expr: TomographyExperiment, n_shots=10_000, active_reset=False) -> DFEData:
    """
    Acquire data necessary for direct fidelity estimate (DFE).

    :param qc: A quantum computer object where the experiment will run.
    :param expr: A partial tomography(``TomographyExperiment``) object describing the experiments to be run.
    :param n_shots: The minimum number of shots to be taken in each experiment (including calibration).
    :param active_reset: Boolean flag indicating whether experiments should terminate with an active reset instruction
        (this can make experiments a lot faster).
    :return: a DFE data object
    :rtype: ``DFEData`
    """
    res = list(measure_observables(qc, expr, n_shots=n_shots, active_reset=active_reset))
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

    :param data: A ``DFEData`` object containing raw experimental results.
    :param kind: A string describing the kind of DFE data being analysed ('state' or 'process')
    :return: a DFE estimate object
    :rtype: ``DFEEstimate`

    """
    p_mean = np.mean(data.pauli_point_est)
    p_variance = np.sum(data.pauli_std_err**2)
    d = data.dimension

    if kind == 'state':
        mean_est = p_mean
        var_est = p_variance / len(data.results) ** 2
    elif kind == 'process':
        # TODO: double check this
        mean_est = (d**2 * p_mean + d)/(d**2+d)
        v = p_variance / len(data.results) ** 2
        var_est = d**2/(d+1)**2 * v
    else:
        raise ValueError('DFEdata can only be of kind \'state\' or \'process\'.')

    return DFEEstimate(dimension=data.dimension,
                       qubits=data.qubits,
                       fid_point_est=mean_est,
                       fid_std_err=np.sqrt(var_est))
