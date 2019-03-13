import copy
import functools
import itertools
from operator import mul
from typing import List, Sequence
from typing import Tuple

import numpy as np
from dataclasses import dataclass

import pyrethrum.operator_estimation as est
from forest.benchmarking.utils import prepare_prod_pauli_eigenstate
from pyquil import Program
from pyquil.api import QuantumComputer, get_benchmarker
from pyquil.operator_estimation import ExperimentSetting, TomographyExperiment, \
    TensorProductState, plusX, minusX, plusY, minusY, plusZ, minusZ
from pyquil.paulis import PauliTerm, PauliSum, sI, sX, sY, sZ

bm = get_benchmarker()


def calibrate_readout_imperfections(pauli: PauliTerm, quantum_machine: QuantumComputer,
                                    var: float = 0.01):
    """
    Compute the expectation value and variance of measuring a pauli without any pre or post circuit.
    This can be used to attribute lack of of visibility to readout imperfections.

    TODO:
      + look into whether we can calibrate multiple PauliTerm with one call
      + test how well this calibration works with correlated readout errors

    :param pauli: The Pauli expectation to calibrate.
    :param quantum_machine: The QVM or QPU to use for calibration.
    :param var: The desired variance of the result.
    :return: The expectation, variance, and number of shots of the expectation value of a +1
     eigenstate of pauli.
    """
    this_prog = prepare_prod_pauli_eigenstate(pauli)
    expectation, variance, count = est.estimate_locally_commuting_operator(
        this_prog, PauliSum([pauli]), var, quantum_machine)
    return expectation[0], variance[0, 0].real, count


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


def _exhaustive_dfe(program: Program, qubits: Sequence[int], in_states):
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
            out_operator=bm.apply_clifford_to_pauli(program, _state_to_pauli(i_st)),
        )


def exhaustive_process_dfe(program, qubits):
    """
    Estimate process fidelity by exhaustive direct fidelity estimation.

    This leads to a quadratic reduction in overhead w.r.t. process tomography for
    fidelity estimation.

    The algorithm is due to:

    [DFE1]  Practical Characterization of Quantum Devices without Tomography
            Silva et al., PRL 107, 210404 (2011)
            https://doi.org/10.1103/PhysRevLett.107.210404

    [DFE2]  Direct Fidelity Estimation from Few Pauli Measurements
            Flammia and Liu, PRL 106, 230501 (2011)
            https://doi.org/10.1103/PhysRevLett.106.230501

    :param program: A program comprised of clifford gates that defines the process for
        which we estimate the fidelity.
    :param qubits: The qubits to perform DFE on. This can be a superset of the qubits
        used in ``program``.
    """
    return TomographyExperiment(list(
        _exhaustive_dfe(program=program, qubits=qubits,
                        in_states=[None, plusX, minusX, plusY, minusY, plusZ, minusZ])),
        program=program, qubits=qubits)


def exhaustive_state_dfe(program, qubits):
    """
    Estimate state fidelity by exhaustive direct fidelity estimation.

    This leads to a quadratic reduction in overhead w.r.t. state tomography for
    fidelity estimation.

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
    """
    return TomographyExperiment(list(
        _exhaustive_dfe(program=program, qubits=qubits, in_states=[None, plusZ, minusZ])),
        program=program, qubits=qubits)


def _monte_carlo_dfe(program: Program, qubits: Sequence[int], in_states, n_terms: int):
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
            out_operator=bm.apply_clifford_to_pauli(program, _state_to_pauli(i_st)),
        )


def monte_carlo_process_dfe(program: Program, qubits: List[int], n_terms=200):
    return TomographyExperiment(list(
        _monte_carlo_dfe(program=program, qubits=qubits,
                         in_states=[None, plusX, minusX, plusY, minusY, plusZ, minusZ],
                         n_terms=n_terms)),
        program=program, qubits=qubits)


@dataclass
class DFEdata:
    """Experimental data from a DFE experiment"""
    in_pauli: List[str]
    """The input Pauli operators being acted on by the `program`"""

    program: Program
    """The pyquil Program to perform DFE on"""

    out_pauli: List[str]
    """The expected output Pauli operators after the program acts on the corresponding `in_pauli`"""

    dimension: int
    """Dimension of the Hilbert space"""

    number_qubits: int
    """number of qubits"""

    expectation: List[float]
    """expectation values as reported from the QPU"""

    variance: List[float]
    """variances associated with the `expectation`"""

    count: List[int]
    """number of shots used to calculate the `expectation`"""

def acquire_dfe_data(experiment, quantum_machine: QuantumComputer, var: float = 0.01) -> Tuple[DFEdata, DFEdata]:
    """
    Estimate state/process fidelity by exhaustive direct fidelity estimation.

    This leads to a quadratic reduction in overhead wrt state tomography for fidelity estimation.

    The algorithm is due to:

    [DFE1]  Practical Characterization of Quantum Devices without Tomography
            Silva et al., PRL 107, 210404 (2011)
            https://doi.org/10.1103/PhysRevLett.107.210404

    [DFE2]  Direct Fidelity Estimation from Few Pauli Measurements
            Flammia and Liu, PRL 106, 230501 (2011)
            https://doi.org/10.1103/PhysRevLett.106.230501


    :param experiment: namedtuple with fields 'in_pauli', 'program', and 'out_pauli'.
    :param quantum_machine: QPUConnection or QVMConnection object to be used
    :param var: maximum tolerable variance per observable
    :return: the experiment and calibration data
    """
    # get qubit information
    qubits = experiment.program.get_qubits()
    n_qubits = len(qubits)
    dimension = 2 ** len(qubits)

    expectations = []
    variances = []
    counts = []
    cal_exps = []
    cal_vars = []
    cal_counts = []

    for (ip, op) in zip(experiment.in_pauli, experiment.out_pauli):
        # at the moment estimate_locally_commuting_operator mutates prog so deepcopy is needed
        this_prog = copy.deepcopy(experiment.program)

        # Create preparation program for the corresponding input Pauli and then append the
        # circuit we want to characterize
        tot_prog = prepare_prod_pauli_eigenstate(ip) + this_prog

        # measure the output Pauli operator in question i.e. data aqcuisition
        expectation, variance, count = \
            est.estimate_locally_commuting_operator(tot_prog, PauliSum([op]), var, quantum_machine)
        expectations += [expectation[0]]
        variances += [variance[0, 0].real]
        counts += [count]

        # calibration
        cal_exp, cal_var, cal_count = calibrate_readout_imperfections(op, quantum_machine, var)
        cal_exps += [cal_exp]
        cal_vars += [cal_var]
        cal_counts += [cal_count]

    exp_data = DFEdata(
        in_pauli=[ip.pauli_string(qubits) for ip in experiment.in_pauli],
        program=experiment.program,
        out_pauli=[op.pauli_string(qubits) for op in experiment.out_pauli],
        dimension=dimension,
        number_qubits=n_qubits,
        expectation=expectations,
        variance=variances,
        count=counts
    )
    cal_data = DFEdata(
        in_pauli=[ip.pauli_string(qubits) for ip in experiment.in_pauli],
        program=experiment.program,
        out_pauli=[op.pauli_string(qubits) for op in experiment.out_pauli],
        dimension=dimension,
        number_qubits=n_qubits,
        expectation=cal_exps,
        variance=cal_vars,
        count=cal_counts
    )
    return exp_data, cal_data


@dataclass
class DFEestimate:
    """State/Process estimates from DFE experiments"""
    in_pauli: List[str]
    """The input Pauli operators being acted on by the `program`"""

    program: Program
    """The pyquil Program to perform DFE on"""

    out_pauli: List[str]
    """The expected output Pauli operators after the program acts on the corresponding `in_pauli`"""

    dimension: int
    """Dimension of the Hilbert space"""

    number_qubits: int
    """number of qubits"""

    pauli_point_est: List[float]
    """Point estimate of Pauli operators"""

    pauli_var_est: List[float]
    """Estimate of varaince in the point estimate"""

    fid_point_est: float
    """Point estimate of fidelity between ideal gate or state and measured, rescaled by the calibration."""

    fid_var_est: float
    """Variance of the fidelity point estimate, after considering the calibration."""

    fid_std_err_est: float
    """Standard deviation of the fidelity point estimate, after considering the calibration."""

def direct_fidelity_estimate(data: DFEdata, cal: DFEdata, type: str) -> DFEestimate:
    """
    Estimate state or process fidelity by exhaustive direct fidelity estimation.

    :param data: data from DFE experiment
    :param cal: calibration data from DFE experiment
    :param type: 'state' or 'process', 'process' is default.
    """
    pauli_est = data.expectation / np.abs(cal.expectation)
    var_est = ratio_variance(data.expectation, data.variance, cal.expectation, cal.variance)

    temp_data = { 'dimension': data.dimension, 'expectation': pauli_est, 'variance': var_est }

    if type == 'state':
        fid = aggregate_state_dfe(temp_data)
    elif type == 'process':
        fid = aggregate_process_dfe(temp_data)
    else:
        raise ValueError('Error: must specify state or process DFE.')

    return DFEestimate(
        in_pauli=data.in_pauli,
        program=data.program,
        out_pauli=data.out_pauli,
        dimension=data.dimension,
        number_qubits=data.number_qubits,
        pauli_point_est=pauli_est ,
        pauli_var_est=var_est,
        fid_point_est=fid['expectation'],
        fid_var_est=fid['variance'],
        fid_std_err_est=np.sqrt(fid['variance'])
    )

def aggregate_process_dfe(data: dict):
    if isinstance(data['dimension'], int):
        dim = data['dimension']
    else:
        dim = list(set(data['dimension']))
        assert len(dim) == 0
        dim = dim[0]
    return {
        'expectation': (1.0 + np.sum(data['expectation']) + dim) / (dim * (dim + 1)),
        'variance': np.sum(data['variance']) / (dim * (dim + 1)) ** 2,
    }


def aggregate_state_dfe(data: dict):
    return {
        'expectation': (1.0 + np.sum(data['expectation'])) / data['dimension'],
        'variance': np.sum(data['variance'])
    }


def ratio_variance(a: np.ndarray, var_a: np.ndarray, b: np.ndarray, var_b: np.ndarray) -> np.ndarray:
    """
    Given random variables 'A' and 'B', compute the variance on the ratio Y = A/B. Denote the
    mean of the random variables as a = E[A] and b = E[B] while the variances are var_a = Var[A]
    and var_b = Var[B] and the covariance as Cov[A,B]. The following expression approximates the
    variance of Y

    Var[Y] \approx (a/b) ^2 * ( var_a /a^2 + var_b / b^2 - 2 * Cov[A,B]/(a*b) )

    Below we assume the covariance of a and b is negligible.

    See the following for more details:
      - https://doi.org/10.1002/(SICI)1097-0320(20000401)39:4<300::AID-CYTO8>3.0.CO;2-O
      - http://www.stat.cmu.edu/~hseltman/files/ratio.pdf
      - https://en.wikipedia.org/wiki/Taylor_expansions_for_the_moments_of_functions_of_random_variables

    :param a: Iterable of means, to be used as the numerator in a ratio.
    :param var_a: Iterable of variances for each mean in a.
    :param b: Iterable of means, to be used as the denominator in a ratio.
    :param var_b: Iterable of variances for each mean in b.

    TODO: add in covariance correction?

    """
    return (np.asarray(a)/np.asarray(b))**2 * (var_a/np.asarray(a)**2 + var_b/np.asarray(b)**2)
