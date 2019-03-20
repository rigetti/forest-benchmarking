from collections import OrderedDict
from math import pi
from typing import Iterable, List, Tuple
from itertools import chain

import numpy as np
from lmfit import Model
from numpy import bincount
from numpy import pi
from pandas import DataFrame, Series
from pyquil.operator_estimation import measure_observables
from scipy.stats import beta

from pyquil.api import BenchmarkConnection, QuantumComputer
from pyquil.gates import CZ, RX, RZ
from pyquil.quilbase import Gate
from pyquil.quil import address_qubits, merge_programs
from pyquil import Program
from pyquil.quilatom import QubitPlaceholder
from forest.benchmarking.tomography import generate_state_tomography_experiment, acquire_tomography_data

RB_TYPES = ["std-1q", "std-2q", "sim-1q", "sim-2q"]


def rb_dataframe(rb_type: str, subgraph: List[Tuple], depths: List[int],
                 num_sequences: int) -> DataFrame:
    """
    Generate and return a DataFrame to characterize an RB or unitarity measurement.

    For standard RB see
    [RB] Scalable and Robust Randomized Benchmarking of Quantum Processes
         Magesan et al.,
         Phys. Rev. Lett. 106, 180504 (2011)
         https://dx.doi.org/10.1103/PhysRevLett.106.180504
         https://arxiv.org/abs/1009.3639

    Unitarity algorithm is due to
    [ECN]  Estimating the Coherence of Noise
           Wallman et al.,
           New Journal of Physics 17, 113020 (2015)
           https://dx.doi.org/10.1088/1367-2630/17/11/113020
           https://arxiv.org/abs/1503.07865

    :param rb_type: Label saying which type of RB measurement we're running
    :param subgraph: List of tuples, where each tuple specifies a single qubit or pair of qubits on
         which to run RB. If the length of this list is >1 then we are running simultaneous RB.
    :param depths: List of RB sequence depths (numbers of Cliffords) to include in this measurement
    :param num_sequences: Number of different random sequences to run at each depth
    :return: DataFrame with one row for every sequence in the measurement
    """
    if not rb_type in RB_TYPES:
        raise ValueError(f"Invalid RB type {rb_type} specified. Valid choices are {RB_TYPES}.")
    if len(subgraph) > 1 and not "sim" in rb_type:
        raise ValueError("Found multiple entries in subgraph but rb_type does not specify a "
                         "simultaneous experiment.")

    def df_dict():
        for d in depths:
            for _ in range(num_sequences):
                yield OrderedDict({"RB Type": rb_type,
                                   "Subgraph": subgraph,
                                   "Depth": d})

    # TODO: Put dtypes on this DataFrame in the right way
    return DataFrame(df_dict())


def add_sequences_to_dataframe(df: DataFrame, bm: BenchmarkConnection, random_seed: int = None, interleaved_gate: Program = None):
    """
    Generates a new random sequence for each row in the measurement DataFrame and adds these to a
    copy of the DataFrame. Returns the new DataFrame.

    :param df: An rb dataframe populated with subgraph and depth columns whose copy will be populated with sequences.
    :param bm: A benchmark connection that will do the grunt work of generating the sequences
    :param random_seed: Base random seed used to seed compiler for sequence generation for each subgraph element
    :param interleaved_gate: Gate to interleave in between Cliffords; used for interleaved RB experiment
    :return: New DataFrame with the desired rb sequences stored in "Sequence" column
    """
    new_df = df.copy()
    if random_seed is None:
        new_df["Sequence"] = Series([generate_simultaneous_rb_sequence(bm, s, d, interleaved_gate=interleaved_gate) for (s, d)
                                     in zip(new_df["Subgraph"].values, new_df["Depth"].values)])
    else:
        # use random_seed as the seed for the first call, and increment for subsequent calls
        new_df["Sequence"] = Series(
            [generate_simultaneous_rb_sequence(bm, s, d, random_seed + j * len(s), interleaved_gate)
             for j, (s, d) in enumerate(zip(new_df["Subgraph"].values, new_df["Depth"].values))])
    return new_df


def rb_seq_to_program(rb_seq: List[Program], subgraph: List[Tuple]) -> Program:
    """
    Combines an RB sequence into a single program that includes appends measurements and returns
    this Program.

    :rtype: Program
    """
    qubits = list(chain.from_iterable(subgraph))
    program = merge_programs(rb_seq)
    ro = program.declare('ro', 'BIT', len(qubits))
    return program.measure_all(*zip(qubits,ro))


def run_rb_measurement(df: DataFrame, qc: QuantumComputer, num_trials: int):
    """
    Executes trials on all sequences and adds the results to a copy of the DataFrame. Returns
    the new DataFrame.
    """
    new_df = df.copy()

    def run(qc: QuantumComputer, seq: List[Program], sg: List[Tuple], num_trials: int) -> np.ndarray:
        prog = rb_seq_to_program(seq, sg).wrap_in_numshots_loop(num_trials)
        executable = qc.compiler.native_quil_to_executable(prog)
        return qc.run(executable)
    seqs = new_df["Sequence"].values
    sgs = new_df["Subgraph"].values
    new_df["Results"] = Series([run(qc, seq, sg, num_trials) for seq, sg in zip(seqs, sgs)])
    return new_df


def oneq_rb_gateset(qubit: QubitPlaceholder) -> Gate:
    """
    Yield the gateset for 1-qubit randomized benchmarking.

    :param qubit: The qubit to effect the gates on. Might I suggest you provide
        a :py:class:`QubitPlaceholder`?
    """
    for angle in [-pi, -pi / 2, pi / 2, pi]:
        for gate in [RX, RZ]:
            yield gate(angle, qubit)


def twoq_rb_gateset(q1: QubitPlaceholder, q2: QubitPlaceholder) -> Iterable[Gate]:
    """
    Yield the gateset for 2-qubit randomized benchmarking.

    This is two 1-q gatesets and ``CZ``.

    :param q1: The first qubit. Might I suggest you provide a :py:class:`QubitPlaceholder`?
    :param q2: The second qubit. Might I suggest you provide a :py:class:`QubitPlaceholder`?
    """
    yield from oneq_rb_gateset(q1)
    yield from oneq_rb_gateset(q2)
    yield CZ(q1, q2)


def get_rb_gateset(rb_type: str) -> Tuple[List[Gate], Tuple[QubitPlaceholder]]:
    """
    A wrapper around the gateset generation functions.

    :param rb_type: "1q" or "2q".
    :returns: list of gates, tuple of qubits
    """
    if rb_type == '1q':
        q = QubitPlaceholder()
        return list(oneq_rb_gateset(q)), (q,)

    if rb_type == '2q':
        q1, q2 = QubitPlaceholder.register(n=2)
        return list(twoq_rb_gateset(q1, q2)), (q1, q2)

    raise ValueError(f"No RB gateset for {rb_type}")


def generate_rb_sequence(compiler: BenchmarkConnection, rb_type: str,
                         depth: int,  random_seed: int = None) -> (List[Program], List[QubitPlaceholder]):
    """
    Generate a complete randomized benchmarking sequence.

    :param compiler: A compiler connection that will do the grunt work of generating the sequences
    :param rb_type: "1q" or "2q".
    :param depth: The total number of Cliffords in the sequence (including inverse)
    :param random_seed: Random seed passed to compiler to seed sequence generation.
    :return: A dictionary with keys "program", "qubits", and "bits".
    """
    if depth < 2:
        raise ValueError("Sequence depth must be at least 2 for rb sequences, or at least 1 for unitarity sequences.")
    gateset, q_placeholders = get_rb_gateset(rb_type=rb_type)
    programs = compiler.generate_rb_sequence(depth=depth, gateset=gateset, seed=random_seed)
    return programs, q_placeholders


def generate_simultaneous_rb_sequence(bm: BenchmarkConnection, subgraph: list,
                                      depth: int,  random_seed: int = None, interleaved_gate: Program = None) -> list:
    """
    Generates a Simultaneous RB Sequence -- a list of Programs where each Program performs a
    simultaneous Clifford on the given subgraph (single qubit or pair of qubits), and where the
    execution of all the Programs composes to the Identity on all edges.

    :param bm: A benchmark connection that will do the grunt work of generating the sequences
    :param subgraph: Iterable of tuples of integers specifying qubit singletons or pairs
    :param depth: The total number of Cliffords to perform on all edges (including inverse)
    :param random_seed: Base random seed used to seed compiler for sequence generation for each subgraph element
    :param interleaved_gate: Gate to interleave in between Cliffords; used for interleaved RB experiment
    :return: RB Sequence as a list of Programs
    """
    if depth < 2:
        raise ValueError("Sequence depth must be at least 2 for rb sequences, or at least 1 for unitarity sequences.")
    size = len(subgraph[0])
    assert all([len(x) == size for x in subgraph])
    if size == 1:
        q_placeholders = QubitPlaceholder().register(n=1)
        gateset = list(oneq_rb_gateset(*q_placeholders))
    elif size == 2:
        q_placeholders = QubitPlaceholder.register(n=2)
        gateset = list(twoq_rb_gateset(*q_placeholders))
    else:
        raise ValueError("Subgraph elements must have length 1 or 2.")
    sequences = []
    for j, qubits in enumerate(subgraph):
        if random_seed is not None:
            sequence = bm.generate_rb_sequence(depth=depth, gateset=gateset, seed=random_seed+j,
                                                     interleaver=interleaved_gate)
        else:
            sequence = bm.generate_rb_sequence(depth=depth, gateset=gateset, interleaver=interleaved_gate)
        qubit_map = {qp: qid for (qp, qid) in zip(q_placeholders, qubits)}
        sequences.append([address_qubits(prog, qubit_map) for prog in sequence])
    return merge_sequences(sequences)


def merge_sequences(sequences: list) -> list:
    """
    Takes a list of equal-length "sequences" (lists of Programs) and merges them element-wise,
    returning the merged outcome.

    :param sequences: List of equal-length Lists of Programs
    :return: A single List of Programs
    """
    depth = len(sequences[0])
    assert all([len(s) == depth for s in sequences])
    return [merge_programs([seq[idx] for seq in sequences]) for idx in range(depth)]


########
# Unitarity
########


def strip_inverse_from_sequences(df: DataFrame):
    """
    Removes the inverse (the last gate) from each of the RB sequences in a copy of the
    DataFrame and returns the copy.

    :param df: Dataframe with "Sequence" series.
    :return new_df: A copy of the input df with each entry in the "Sequence" series
               lacking the last inverse gate
    """
    new_df = df.copy()
    new_df["Sequence"] = Series([seq[:-1] for seq in new_df["Sequence"].values])
    return new_df


def add_unitarity_sequences_to_dataframe(df: DataFrame, bm: BenchmarkConnection, random_seed: int = None):
    """
    Generates a new random unitarity sequence for each row in the measurement DataFrame and adds
    these to a copy of the DataFrame.

    A unitarity sequence of depth D is a standard RB sequence
    of depth D+1 with the last (inversion) gate stripped. Returns the new DataFrame.
    """
    new_df = df.copy()
    if random_seed is not None:
        new_df["Sequence"] = Series([generate_simultaneous_rb_sequence(bm, s, d, random_seed + len(s) * j) for j, (s, d)
                                     in enumerate(zip(new_df["Subgraph"].values, new_df["Depth"].values + 1))])
                            #TODO: check consistency with depth for RB
    else:
        new_df["Sequence"] = Series([generate_simultaneous_rb_sequence(bm, s, d) for (s, d)
                                     in zip(new_df["Subgraph"].values, new_df["Depth"].values + 1)])
                            #TODO: check consistency with depth for RB
    stripped_seq_df = strip_inverse_from_sequences(new_df)
    return stripped_seq_df


def run_unitarity_measurement(df: DataFrame, qc: QuantumComputer, num_trials: int):
    """
    Execute trials on all sequences and add the results to a copy of the DataFrame. Returns the
    new DataFrame.
    """
    new_df = df.copy()
    def run(qc: QuantumComputer, seq: List[Program], subgraph: List[List[int]], num_trials: int) -> np.ndarray:
        prog = merge_programs(seq)
        # TODO: parallelize
        results = []
        for qubits in subgraph:
            state_prep = prog
            tomo_exp = generate_state_tomography_experiment(state_prep, qubits=qubits)
            _rs = list(measure_observables(qc, tomo_exp, num_trials))
            # Inelegant shim from state tomo refactor. To clean up!
            expectations=[r.expectation for r in _rs[1:]]
            variances=[r.stddev ** 2 for r in _rs[1:]]
            results.append((expectations, variances))
        return results

    new_df["Results"] = Series(
        [run(qc, seq, sg, num_trials) for seq, sg in zip(new_df["Sequence"].values, new_df["Subgraph"].values)])
    return new_df


def unitarity_to_RB_decay(unitarity, dimension):
    """
    This allows comparison of measured unitarity and RB decays.

    This function provides an upper bound on the
    RB decay given the input unitarity, where the upperbound is saturated when no unitary errors are present,
    e.g. in the case of depolarizing noise. For more, see Proposition 8. in [ECN]
        unitarity >= (1-dr/(d-1))^2
    where r is the average gate infidelity and d is the dimension

    :param unitarity: The measured decay parameter in a unitarity measurement
    :param dimension: The dimension of the Hilbert space, 2^num_qubits
    :return: The upperbound on RB decay, saturated if no unitary errors are present Proposition 8 [ECN]
    """
    r = (np.sqrt(unitarity) - 1)*(1-dimension)/dimension
    return average_gate_infidelity_to_RB_decay(r, dimension)


#########
# Analysis stuff
#########


def survival_statistics(bitstrings):
    """
    Calculate the mean and variance of the estimated probability of the ground state given shot
    data on one or more bits.

    For binary classified data with N counts of 1 and M counts of 0, these
    can be estimated using the mean and variance of the beta distribution beta(N+1, M+1) where the
    +1 is used to incorporate an unbiased Bayes prior.

    :param ndarray bitstrings: A 2D numpy array of repetitions x bit-arrays.
    :return: (survival mean, sqrt(survival variance))
    """
    survived = np.sum(bitstrings, axis=1) == 0

    # count obmurrences of 000...0 and anything besides 000...0
    n_died, n_survived = bincount(survived, minlength=2)

    # mean and variance given by beta distribution with a uniform prior
    survival_mean = beta.mean(n_survived + 1, n_died + 1)
    survival_var = beta.var(n_survived + 1, n_died + 1)
    return survival_mean, np.sqrt(survival_var)


def survivals_from_results(subgraph: List[Tuple],
                           results: np.ndarray) -> (List[float], List[float]):
    """
    Group shot data by subgraph element and turn the shot data into survival statistics.

    :return: Survival means and errors, in lists whose indices correspond to subgraph elements.
    """
    l = len(subgraph[0])  # 1 for 1Q, 2 for 2Q
    means, errs = [], []
    for idx, s in enumerate(subgraph):
        # Group shot data by (qubit, ) or (qubit, pair)
        shot_data = np.asarray(results)[:, l * idx:l * (idx + 1)]
        mean, err = survival_statistics(shot_data)
        means.append(mean)
        errs.append(err)
    return means, errs


def add_survivals(df: DataFrame):
    """
    Calculate survival statistics on the measurement and add these to a copy of the DataFrame.
    Returns the new DataFrame.
    """
    new_df = df.copy()
    new_df["Survival Means"], new_df["Survival Errors"] = zip(
        *new_df.apply(lambda row: survivals_from_results(row["Subgraph"], row["Results"]), axis=1))
    return new_df


def survivals_by_qubits(df: DataFrame, qubits: Tuple) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a DataFrame that is already populated with survival data matching each subgraph entry,
    find and return all depths, survival means, and survival errors for a given set of qubits.
    """
    depths, means, errs = [], [], []
    # TODO: Make this smarter/faster/more scalable than iterrow'ing
    for _, row in df.iterrows():
        qubits_idx = row["Subgraph"].index(qubits)
        depths.append(row["Depth"])
        means.append(row["Survival Means"][qubits_idx])
        errs.append(row["Survival Errors"][qubits_idx])
    return np.asarray(depths), np.asarray(means), np.asarray(errs)


def standard_rb(x, baseline, amplitude, decay):
    """
    Fitting function for randomized benchmarking.

    :param numpy.ndarray x: Independent variable
    :param float baseline: Offset value
    :param float amplitude: Amplitude of exponential decay
    :param float decay: Decay parameter
    :return: Fit function
    """
    return baseline + amplitude * decay ** x


def standard_rb_guess(model: Model, y):
    """
    Guess the parameters for a fit.

    :param model: an lmfit model to make guess parameters for. This should probably be an
        instance of ``Model(standard_rb)``.
    :param y: Dependent variable
    :return: Lmfit parameters object appropriate for passing to ``Model.fit()``.
    """
    b_guess = y[-1]
    a_guess = y[0] - y[-1]
    d_guess = 0.95
    return model.make_params(baseline=b_guess, amplitude=a_guess, decay=d_guess)


def _check_data(x, y, weights):
    if not len(x) == len(y):
        raise ValueError("Lengths of x and y arrays must be equal.")
    if weights is not None and not len(x) == len(weights):
        raise ValueError("Lengths of x and weights arrays must be equal is weights is not None.")


def fit_standard_rb(depths, survivals, weights=None):
    """
    Construct and fit an RB curve with appropriate guesses

    :param depths: The clifford circuit depths (independent variable)
    :param survivals: The survival probabilities (dependent variable)
    :param weights: Optional weightings of each point to use when fitting.
    :return: an lmfit Model
    """
    _check_data(depths, survivals, weights)
    rb_model = Model(standard_rb)
    params = standard_rb_guess(model=rb_model, y=survivals)
    return rb_model.fit(survivals, x=depths, params=params, weights=weights)


########
# Unitarity Analysis
########

def estimate_purity(D: int, op_expect: np.ndarray, renorm: bool=True):
    """
    The renormalized, or 'shifted', purity is given in equation (10) of [ECN]
    where d is the dimension of the Hilbert space, 2**num_qubits

    :param D: dimension of the hilbert space
    :param op_expect: array of estimated expectations of each operator being measured
    :param renorm: flag that renormalizes result to be between 0 and 1
    :return: purity given the operator expectations
    """
    purity = (1 / D) * (1 + np.sum(op_expect * op_expect))
    if renorm:
        purity = (D / (D - 1.0)) * (purity - 1.0 / D)
    return purity


def estimate_purity_err(D: int, op_expect: np.ndarray, op_expect_var: np.ndarray, renorm=True):
    """
    Propagate the observed variance in operator expectation to an error estimate on the purity.
    This assumes that each operator expectation is independent.

    :param D: dimension of the Hilbert space
    :param op_expect: array of estimated expectations of each operator being measured
    :param op_expect_var: array of estimated variance for each operator expectation
    :param renorm: flag that provides error for the renormalized purity
    :return: purity given the operator expectations
    """
    #TODO: check validitiy of approximation |op_expect| >> 0, and functional form below (squared?)
    var_of_square_op_expect = (2 * np.abs(op_expect)) ** 2 * op_expect_var
    #TODO: check if this adequately handles |op_expect| >\> 0
    need_second_order = np.isclose([0.]*len(var_of_square_op_expect), var_of_square_op_expect, atol=1e-6)
    var_of_square_op_expect[need_second_order] = op_expect_var[need_second_order]**2

    purity_var = (1 / D) ** 2 * (np.sum(var_of_square_op_expect))


    if renorm:
        purity_var = (D / (D - 1.0)) ** 2 * purity_var

    return np.sqrt(purity_var)


def shifted_purities_from_results(subgraph: List[Tuple],
                          results: np.ndarray) -> (List[float], List[float]):
    """
    Group results by subgraph element and calculate the purity statistics.

    :return: Shifted purities and corresponding errors, in lists whose indices correspond to subgraph elements.
    """
    purities, errs = [], []
    for component, component_results in zip(subgraph, results):
        expectations, variances = component_results
        dimension = 2 ** len(component)
        shifted_purity = estimate_purity(dimension, np.asarray(expectations))
        err = estimate_purity_err(dimension,  np.asarray(expectations),  np.asarray(variances))
        purities.append(shifted_purity)
        errs.append(err)
    return purities, errs


def add_shifted_purities(df: DataFrame):
    """
    Calculate purities given a DataFrame containing "Results" and add these to a copy of the
    DataFrame. Returns the new DataFrame.
    """
    new_df = df.copy()
    new_df["Shifted Purities"], new_df["Purity Errors"] = zip(
        *new_df.apply(lambda row: shifted_purities_from_results(row["Subgraph"], row["Results"]), axis=1))
    return new_df


def shifted_purities_by_qubits(df: DataFrame, qubits: Tuple) -> (np.array, np.array, np.array):
    """
    Given a DataFrame that is already populated with purity data matching each subgraph entry,
    find and return all depths, shifted purities, and purity errors for a given set of qubits.
    """
    depths, shifted_purities, errs = [], [], []
    # TODO: Make this smarter/faster/more scalable than iterrow'ing
    for _, row in df.iterrows():
        qubits_idx = row["Subgraph"].index(qubits)
        depths.append(row["Depth"])
        shifted_purities.append(row["Shifted Purities"][qubits_idx])
        errs.append(row["Purity Errors"][qubits_idx])
    return np.asarray(depths), np.asarray(shifted_purities), np.asarray(errs)

def unitarity_fn(x, baseline, amplitude, unitarity):
    """
    Fitting function for unitarity randomized benchmarking, equation (8) of [ECN]

    :param numpy.ndarray x: Independent variable
    :param float baseline: Offset value
    :param float amplitude: Amplitude of exponential decay
    :param float decay: Decay parameter
    :return: Fit function
    """
    return baseline + amplitude * unitarity ** (x-1)

#TODO: confirm validity or update guesses
def unitarity_guess(model: Model, y):
    """
    Guess the parameters for a fit.

    :param model: an lmfit model to make guess parameters for. This should probably be an
        instance of ``Model(unitarity)``.
    :param y: Dependent variable
    :return: Lmfit parameters object appropriate for passing to ``Model.fit()``.
    """
    b_guess = 0.
    a_guess = y[0]
    d_guess = 0.95
    return model.make_params(baseline=b_guess, amplitude=a_guess, unitarity=d_guess)


def fit_unitarity(depths, shifted_purities, weights=None):
    """Construct and fit an RB curve with appropriate guesses

    :param depths: The clifford circuit depths (independent variable)
    :param shifted_purities: The shifted purities (dependent variable)
    :param weights: Optional weightings of each point to use when fitting.
    :return: an lmfit Model
    """
    _check_data(depths, shifted_purities, weights)
    unitarity_model = Model(unitarity_fn)
    params = unitarity_guess(model=unitarity_model, y=shifted_purities)
    return unitarity_model.fit(shifted_purities, x=depths, params=params, weights=weights)


########
# Interleaved RB Analysis
########


def coherence_angle(rb_decay, unitarity):
    """
    Equation 29 of [U+IRB]

    :param rb_decay: Observed decay parameter in standard rb experiment
    :param unitarity: Observed decay parameter in unitarity experiment
    :return: coherence angle
    """
    return np.arccos(rb_decay / np.sqrt(unitarity))


def gamma(irb_decay, unitarity):
    """
    Corollary 5 of [U+IRB], second line

    :param irb_decay: Observed decay parameter in irb experiment with desired gate interleaved between Cliffords
    :param unitarity: Observed decay parameter in unitarity experiment
    :return: gamma
    """
    return irb_decay/np.sqrt(unitarity)


def interleaved_gate_fidelity_bounds(irb_decay, rb_decay, dim, unitarity = None):
    """
    Use observed rb_decay to place a bound on fidelity of a particular gate with given interleaved rb decay.
    Optionally, use unitarity measurement result to provide improved bounds on the interleaved gate's fidelity.

    Bounds due to
    [IRB] Efficient measurement of quantum gate error by interleaved randomized benchmarking
          Magesan et al.,
          Phys. Rev. Lett. 109, 080505 (2012)
          https://dx.doi.org/10.1103/PhysRevLett.109.080505
          https://arxiv.org/abs/1203.4550

    Improved bounds using unitarity due to
    [U+IRB]  Efficiently characterizing the total error in quantum circuits
             Dugas et al.,
             arXiv:1610.05296 (2016)
             https://arxiv.org/abs/1610.05296

    :param irb_decay: Observed decay parameter in irb experiment with desired gate interleaved between Cliffords
    :param rb_decay: Observed decay parameter in standard rb experiment
    :param dim: Dimension of the Hilbert space, 2**num_qubits
    :param unitarity: Observed decay parameter in unitarity experiment; improves bounds if provided.
    :return: The pair of lower and upper bounds on the fidelity of the interleaved gate.
    """
    if unitarity is not None:
        # Corollary 5 of [U+IRB]. Here, the channel X corresponds to the interleaved gate
        # whereas Y corresponds to the averaged-Clifford channel of standard rb.

        pm = [-1, 1]
        theta = coherence_angle(rb_decay, unitarity)
        g =  gamma(irb_decay, unitarity)
        # calculate bounds on the equivalent gate-only decay parameter
        decay_bounds = [sign * (sign * g * np.cos(theta) + np.sin(theta) * np.sqrt(1-g**2) ) for sign in pm]
        # convert decay bounds to bounds on fidelity of the gate
        fidelity_bounds = [RB_decay_to_gate_fidelity(decay, dim) for decay in decay_bounds]

    else:
        # Equation 5 of [IRB]

        E1 = (abs(rb_decay - irb_decay/rb_decay) + (1-rb_decay)) * (dim-1)/dim
        E2 = 2*(dim**2 - 1)*(1-rb_decay)/(rb_decay*dim**2) + 4*np.sqrt(1-rb_decay)*np.sqrt(dim**2-1)/rb_decay

        E = min(E1,E2)
        infidelity = irb_decay_to_gate_infidelity(irb_decay, rb_decay, dim)

        fidelity_bounds = [1-infidelity-E, 1-infidelity+E]

    return fidelity_bounds


def gate_infidelity_to_irb_decay(irb_infidelity, rb_decay, dim):
    """
    For convenience, inversion of Eq. 4 of [IRB]. See irb_decay_to_infidelity

    :param irb_infidelity: Infidelity of the interleaved gate.
    :param rb_decay: Observed decay parameter in standard rb experiment.
    :param dim: Dimension of the Hilbert space, 2**num_qubits
    :return: Decay parameter in irb experiment with relevant gate interleaved between Cliffords
    """
    return (1 - irb_infidelity * (dim/(dim-1)) ) * rb_decay


def irb_decay_to_gate_infidelity(irb_decay, rb_decay, dim):
    """
    Eq. 4 of [IRB], which provides an estimate of the infidelity of the interleaved gate,
    given both the observed interleaved and standard decay parameters.

    :param irb_decay: Observed decay parameter in irb experiment with desired gate interleaved between Cliffords
    :param rb_decay: Observed decay parameter in standard rb experiment.
    :param dim: Dimension of the Hilbert space, 2**num_qubits
    :return: Estimated gate infidelity (1 - fidelity) of the interleaved gate.
    """
    return ((dim - 1) / dim) * (1 - irb_decay / rb_decay)


def average_gate_infidelity_to_RB_decay(gate_infidelity, dimension):
    """
    Inversion of eq. 5 of [RB] arxiv paper.

    :param gate_infidelity: The average gate infidelity.
    :param dimension: Dimension of the Hilbert space, 2^num_qubits
    :return: The RB decay corresponding to the gate_infidelity
    """
    return (gate_infidelity - 1 + 1/dimension)/(1/dimension -1)


def RB_decay_to_gate_fidelity(rb_decay, dimension):
    """
    Derived from eq. 5 of [RB] arxiv paper. Note that 'gate' here typically means an element of the Clifford group,
    which comprise standard rb sequences.

    :param rb_decay: Observed decay parameter in standard rb experiment.
    :param dimension: Dimension of the Hilbert space, 2**num_qubits
    :return: The gate fidelity corresponding to the input decay.
    """
    return 1/dimension - rb_decay*(1/dimension -1)
