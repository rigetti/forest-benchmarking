import copy
import functools
import itertools
from operator import mul
from typing import List, Sequence
from typing import Tuple

import numpy as np
from dataclasses import dataclass

import forest.benchmarking.operator_estimation as est
from forest.benchmarking.utils import prepare_prod_pauli_eigenstate
from pyquil import Program
from pyquil.api import BenchmarkConnection, QuantumComputer
from pyquil.operator_estimation import ExperimentSetting, TomographyExperiment, \
    TensorProductState, plusX, minusX, plusY, minusY, plusZ, minusZ
from pyquil.paulis import PauliTerm, PauliSum, sI, sX, sY, sZ


def _exhaustive_dfe(program: Program, qubits: Sequence[int], in_states,
                    benchmarker: BenchmarkConnection) -> ExperimentSetting:
    """Yield experiments over itertools.product(in_paulis).

    Used as a helper function for exhaustive_xxx_dfe routines.

    :param program: A program comprised of clifford gates
    :param qubits: The qubits to perform DFE on. This can be a superset of the qubits
        used in ``program``.
    :param in_paulis: Use these single-qubit Pauli operators in every itertools.product()
        to generate an exhaustive list of DFE experiments.
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


def exhaustive_process_dfe(program: Program, qubits: list,
                           benchmarker: BenchmarkConnection) -> TomographyExperiment:
    """
    Estimate process fidelity by exhaustive direct fidelity estimation.

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
    """
    exp = TomographyExperiment(list(
        _exhaustive_dfe(program=program,
                        qubits=qubits,
                        in_states=[None, plusX, minusX, plusY, minusY, plusZ, minusZ],
                        benchmarker=benchmarker)),
        program=program, qubits=qubits)
    return DFEexperiment(exp, 'process')


def exhaustive_state_dfe(program: Program, qubits: list,
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
    """
    exp = TomographyExperiment(list(
        _exhaustive_dfe(program=program,
                        qubits=qubits,
                        in_states=[None, plusZ],
                        benchmarker=benchmarker)),
        program=program, qubits=qubits)
    return DFEexperiment(exp, 'state')


def _monte_carlo_dfe(program: Program, qubits: Sequence[int], in_states: list, n_terms: int,
                     benchmarker: BenchmarkConnection) -> ExperimentSetting:
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

def monte_carlo_state_dfe(program: Program, qubits: List[int], benchmarker: BenchmarkConnection,
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
    """
    exp = TomographyExperiment(list(
        _monte_carlo_dfe(program=program, qubits=qubits,
                         in_states=[None, plusZ],
                         n_terms=n_terms, benchmarker=benchmarker)),
        program=program, qubits=qubits)
    return DFEexperiment(exp,'state')


def monte_carlo_process_dfe(program: Program, qubits: List[int], benchmarker: BenchmarkConnection,
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
    """
    exp = TomographyExperiment(list(
        _monte_carlo_dfe(program=program, qubits=qubits,
                         in_states=[None, plusX, minusX, plusY, minusY, plusZ, minusZ],
                         n_terms=n_terms, benchmarker=benchmarker)),
        program=program, qubits=qubits)
    return DFEexperiment(exp,'process')


def acquire_dfe_data(qc: QuantumComputer, dfe: DFEexperiment, active_reset=False):
    res = list(measure_observables(qc, dfe.exp, active_reset=active_reset))
    return DFEdata(res, 
                   in_states = [exp[0].in_state for exp in dfe.exp],
                   program = dfe.exp.program,
                   out_pauli = [str(exp[0].out_operator) for exp in bell_state], 
                   pauli_point_est = np.array([r.expectation for r in res ]), 
                   pauli_var_est = np.array([r.stddev**2 for r in res]), 
                   dimesion = 2**len(dfe.exp.qubits), 
                   qubits = dfe.exp.qubits,
                   type = dfe.type)

def analyse_dfe_data(res: DFEdata):
    mean = np.sum([r.expectation for r in res])
    variance = np.sum([r.stddev**2 for r in res])

    if res.type == 'state':
        mean_est = (1+mean)/res.dimension
        var_est = variance/res.dimension**2
    elif res.type == 'process':
        mean_est = (1+mean+res.dimension)/(res.dimension+1)/(res.dimension)
        var_est = variance/(res.dimension+1)**2/(res.dimension)**2
    else:
        raise ValueError('DFEdata can only be of type \'state\' or \'process\'.')

    return DFEestimate( dimension= res.dimension,
                        qubits = res.qubits,
                        fid_point_est = mean_est, 
                        fid_var_est = var_est, 
                        fid_std_err_est = sqrt(var_est))

@dataclass 
class DFEexperiment:
    """Experiments to obtain an estimate of fidelity via DFE"""

    exp: TomographyExperiment
    """The experiment to be performed"""

    type: str
    """Type of fidelity to be estimate (state or process)"""


@dataclass
class DFEdata:
    """Experimental data from a DFE experiment"""

    res: List[ExperimentResults]
    """The experimental results"""

    in_states: List[TensorProductState]
    """The input tensor product states being acted on by the `program`"""

    program: Program
    """The pyquil Program to perform DFE on"""

    out_pauli: List[str]
    """The expected output Pauli operators after the program acts on the corresponding `in_pauli`"""

    pauli_point_est: List[float]
    """Point estimate of Pauli operators"""

    pauli_var_est: List[float]
    """Estimate of variance in the point estimate"""

    dimension: int
    """Dimension of the Hilbert space"""

    qubits: List[int]
    """qubits involved in the experiment"""

    type: str
    """Type of fidelity to be estimate (state or process)"""


@dataclass
class DFEestimate:
    """State/Process estimates from DFE experiments"""

    dimension: int
    """Dimension of the Hilbert space"""

    qubits: List[int]
    """qubits involved in the experiment"""

    fid_point_est: float
    """Point estimate of fidelity between ideal gate or state and measured, rescaled by the calibration."""

    fid_var_est: float
    """Error variance of the fidelity point estimate, after considering the calibration."""

    fid_std_err_est: float
    """Standard error of the fidelity point estimate, after considering the calibration."""
