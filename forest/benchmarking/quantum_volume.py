from typing import List, Sequence, Tuple, Callable, Dict
import warnings
from tqdm import tqdm
import numpy as np
from statistics import median
from collections import OrderedDict
from pandas import DataFrame, Series
import time
from copy import copy

from pyquil.api import QuantumComputer
from pyquil.numpy_simulator import NumpyWavefunctionSimulator
from pyquil.quil import DefGate, Program
from pyquil.gates import RESET
from rpcq.messages import TargetDevice
from rpcq._utils import RPCErrorError

from forest.benchmarking.random_operators import haar_rand_unitary
from forest.benchmarking.utils import bit_array_to_int
import logging
log = logging.getLogger(__name__)


def _naive_program_generator(qc: QuantumComputer, qubits: Sequence[int], permutations: np.ndarray,
                             gates: np.ndarray) -> Program:
    """
    Naively generates a native quil program to implement the circuit which is comprised of the given
    permutations and gates.

    :param qc: the quantum resource that will implement the PyQuil program for each model circuit
    :param qubits: the qubits available for the implementation of the circuit. This naive
        implementation simply takes the first depth-many available qubits.
    :param permutations: array of depth-many arrays of size n_qubits indicating a qubit permutation
    :param gates: a depth by depth//2 array of matrices representing the 2q gates at each layer.
        The first row of matrices is the earliest-time layer of 2q gates applied.
    :return: a PyQuil program in native_quil instructions that implements the circuit represented by
        the input permutations and gates. Note that the qubits are measured in the proper order
        such that the results may be directly compared to the simulated heavy hitters from
        collect_heavy_outputs.
    """
    # artificially restrict the entire computation to num_measure_qubits
    num_measure_qubits = len(permutations[0])
    # if these measure_qubits do not have a topology that supports the program, the compiler may
    # act on a different (potentially larger) subset of the input sequence of qubits.
    measure_qubits = qubits[:num_measure_qubits]

    # create a simple program that uses the compiler to directly generate 2q gates from the matrices
    prog = Program()
    for layer_idx, (perm, layer) in enumerate(zip(permutations, gates)):
        for gate_idx, gate in enumerate(layer):
            # get the Quil definition for the new gate
            g_definition = DefGate("LYR" + str(layer_idx) + "_RAND" + str(gate_idx), gate)
            # get the gate constructor
            G = g_definition.get_constructor()
            # add definition to program
            prog += g_definition
            # add gate to program, acting on properly permuted qubits
            prog += G(int(measure_qubits[perm[gate_idx]]), int(measure_qubits[perm[gate_idx+1]]))

    ro = prog.declare("ro", "BIT", num_measure_qubits)
    for idx, qubit in enumerate(measure_qubits):
        prog.measure(qubit, ro[idx])

    # restrict compilation to chosen qubits
    isa_dict = qc.device.get_isa().to_dict()
    single_qs = isa_dict['1Q']
    two_qs = isa_dict['2Q']

    new_1q = {}
    for key, val in single_qs.items():
        if int(key) in qubits:
            new_1q[key] = val
    new_2q = {}
    for key, val in two_qs.items():
        q1, q2 = key.split('-')
        if int(q1) in qubits and int(q2) in qubits:
            new_2q[key] = val

    new_isa = {'1Q': new_1q, '2Q': new_2q}

    new_compiler = copy(qc.compiler)
    new_compiler.target_device = TargetDevice(isa=new_isa, specs=qc.device.get_specs().to_dict())
    # try to compile with the restricted qubit topology
    try:
        native_quil = new_compiler.quil_to_native_quil(prog)
    except RPCErrorError as e:
        if "Multiqubit instruction requested between disconnected components of the QPU graph:" \
                in str(e):
            raise ValueError("naive_program_generator could not generate a program using only the "
                             "qubits supplied; expand the set of allowed qubits or supply "
                             "a custom program_generator.")
        raise

    return native_quil


def collect_heavy_outputs(wfn_sim: NumpyWavefunctionSimulator, permutations: np.ndarray,
                          gates: np.ndarray) -> List[int]:
    """
    Collects and returns those 'heavy' bitstrings which are output with greater than median
    probability among all possible bitstrings on the given qubits.

    The method uses the provided wfn_sim to calculate the probability of measuring each bitstring
    from the output of the circuit comprised of the given permutations and gates.

    :param wfn_sim: a NumpyWavefunctionSimulator that can simulate the provided program
    :param permutations: array of depth-many arrays of size n_qubits indicating a qubit permutation
    :param gates: depth by num_gates_per_layer many matrix representations of 2q gates.
            The first row of matrices is the earliest-time layer of 2q gates applied.
    :return: a list of the heavy outputs of the circuit, represented as ints
    """
    wfn_sim.reset()

    for layer_idx, (perm, layer) in enumerate(zip(permutations, gates)):
        for gate_idx, gate in enumerate(layer):
            wfn_sim.do_gate_matrix(gate, (perm[gate_idx], perm[gate_idx+1]))

    # Note that probabilities are ordered lexicographically with qubit 0 leftmost.
    probabilities = np.abs(wfn_sim.wf.reshape(-1)) ** 2

    median_prob = median(probabilities)

    # store the integer indices, which implicitly represent the bitstring outcome.
    heavy_outputs = [idx for idx, prob in enumerate(probabilities) if prob > median_prob]

    return heavy_outputs


def generate_abstract_qv_circuit(depth: int) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Produces an abstract description of the square model circuit of given depth=width used in a
    quantum volume measurement.

    The description remains abstract as it does not directly reference qubits in a circuit; rather,
    the circuit is specified as a list of depth many permutations and depth many layers of two
    qubit gates specified as a depth by depth//2 numpy array whose entries are each a haar random
    four by four matrix (a single 2 qubit gate). Each permutation is simply a list of the numbers
    0 through depth-1, where the number x at index i indicates the qubit in position i should be
    moved to position x. The 4 by 4 matrix at gates[i, j] is the gate acting on the qubits at
    positions 2j, 2j+1 after the i^th permutation has occurred.

    :param depth: the depth, and also width, of the model circuit
    :return: the random depth-many permutations and depth by depth//2 many 2q-gates which comprise
        the model quantum circuit of [QVol] for a given depth.
    """
    # generate a simple list representation for each permutation of the depth many qubits
    permutations = [np.random.permutation(range(depth)) for _ in range(depth)]

    # generate a matrix representation of each 2q gate in the circuit
    num_gates_per_layer = depth // 2  # if odd number of qubits, don't do anything to last qubit
    gates = np.asarray([[haar_rand_unitary(4) for _ in range(num_gates_per_layer)]
                        for _ in range(depth)])

    return permutations, gates


def sample_rand_circuits_for_heavy_out(qc: QuantumComputer,
                                       qubits: Sequence[int], depth: int,
                                       program_generator: Callable[[QuantumComputer, Sequence[int],
                                                                    np.ndarray, np.ndarray],
                                                                   Program],
                                       num_circuits: int = 100, num_shots: int = 1000,
                                       show_progress_bar: bool = False) -> int:
    """
    This method performs the bulk of the work in the quantum volume measurement.

    For the given depth, num_circuits many random model circuits are generated, the heavy outputs
    are determined from the ideal output distribution of each circuit, and a native quil
    implementation of the model circuit output by the program generator is run on the qc. The total
    number of sampled heavy outputs is returned.

    :param qc: the quantum resource that will implement the PyQuil program for each model circuit
    :param qubits: the qubits available in the qc for the program_generator to use.
    :param depth: the depth (and width in num of qubits) of the model circuits
    :param program_generator: a method which takes an abstract description of a model circuit and
        returns a native quil program that implements that circuit. See measure_quantum_volume
        docstring for specifics.
    :param num_circuits: the number of random model circuits to sample at this depth; should be >100
    :param num_shots: the number of shots to sample from each model circuit
    :param show_progress_bar: displays a progress bar via tqdm if true.
    :return: the number of heavy outputs sampled among all circuits generated for this depth
    """
    wfn_sim = NumpyWavefunctionSimulator(depth)

    num_heavy = 0
    # display progress bar using tqdm
    for _ in tqdm(range(num_circuits), disable=not show_progress_bar):

        permutations, gates = generate_abstract_qv_circuit(depth)

        # generate a PyQuil program in native quil that implements the model circuit
        # The program should measure the output qubits in the order that is consistent with the
        # comparison of the bitstring results to the heavy outputs given by collect_heavy_outputs
        program = program_generator(qc, qubits, permutations, gates)

        # run the program num_shots many times
        program.wrap_in_numshots_loop(num_shots)
        executable = qc.compiler.native_quil_to_executable(program)
        results = qc.run(executable)

        # classically simulate model circuit represented by the perms and gates for heavy outputs
        heavy_outputs = collect_heavy_outputs(wfn_sim, permutations, gates)

        # determine if each result bitstring is a heavy output, as determined from simulation
        for result in results:
            # convert result to int for comparison with heavy outputs.
            output = bit_array_to_int(result)
            if output in heavy_outputs:
                num_heavy += 1

    return num_heavy


def calculate_prob_est_and_err(num_heavy: int, num_circuits: int, num_shots: int) \
        -> Tuple[float, float]:
    """
    Helper to calculate the estimate for the probability of sampling a heavy output at a
    particular depth as well as the 2 sigma one-sided confidence interval on this estimate.

    :param num_heavy: total number of heavy outputs sampled at particular depth across all circuits
    :param num_circuits: the total number of depth=width model circuits whose output was sampled
    :param num_shots: the total number of shots taken for each circuit
    :return: estimate for the probability of sampling a heavy output at a particular depth as
        well as the 2 sigma one-sided confidence interval on this estimate.
    """
    total_sampled_outputs = num_circuits * num_shots
    prob_sample_heavy = num_heavy / total_sampled_outputs

    # Eq. (C3) of [QVol]. Assume that num_heavy/num_shots is worst-case binomial with param
    # num_circuits and take gaussian approximation. Get 2 sigma one-sided confidence interval.
    one_sided_confidence_interval = prob_sample_heavy - \
        2 * np.sqrt(num_heavy * (num_shots - num_heavy / num_circuits)) / total_sampled_outputs

    return prob_sample_heavy, one_sided_confidence_interval


def measure_quantum_volume(qc: QuantumComputer, qubits: Sequence[int] = None,
                           program_generator: Callable[[QuantumComputer, Sequence[int],
                                                        np.ndarray, np.ndarray], Program] =
                           _naive_program_generator, num_circuits: int = 100, num_shots: int = 1000,
                           depths: np.ndarray = None, achievable_threshold: float = 2/3,
                           stop_when_fail: bool = True, show_progress_bar: bool = False) \
        -> Dict[int, Tuple[float, float]]:
    """
    Measures the quantum volume of a quantum resource, as described in [QVol].

    By default this method scans increasing depths from 2 to len(qubits) and tests whether the qc
    can adequately implement random model circuits on depth-many qubits such that the given
    depth is 'achieved'. A model circuit depth is achieved if the sample distribution for a
    sample of num_circuits many randomly generated model circuits of the given depth sufficiently
    matches the ideal distribution of that circuit (See Eq. 6  of [QVol]). The frequency of
    sampling 'heavy-outputs' is used as a measure of closeness of the circuit distributions. This
    estimated frequency (across all sampled circuits) is reported for each depth along with a
    bool which indicates whether that depth was achieved. The logarithm of the quantum volume is by
    definition the largest achievable depth of the circuit; see
    extract_quantum_volume_from_results for obtaining the quantum volume from the results
    returned by this method.

    [QVol] Validating quantum computers using randomized model circuits
           Cross et al.,
           arXiv:1811.12926v1  (2018)
           https://arxiv.org/abs/1811.12926

    :param qc: the quantum resource whose volume you wish to measure
    :param qubits: available qubits on which to act during measurement. Default all qubits in qc.
    :param program_generator: a method which
        1) takes in a quantum computer, the qubits on that
            computer available for use, an array of sequences representing the qubit permutations
            in a model circuit, an array of matrices representing the 2q gates in the model circuit
        2) outputs a native quil program that implements the circuit and measures the appropriate
            qubits in the order implicitly dictated by the model circuit representation created in
            sample_rand_circuits_for_heavy_out.
        The default option simply picks the smallest qubit labels and lets the compiler do the rest.
    :param num_circuits: number of unique random circuits that will be sampled.
    :param num_shots: number of shots for each circuit sampled.
    :param depths: the circuit depths to scan over. Defaults to all depths from 2 to len(qubits)
    :param achievable_threshold: threshold at which a depth is considered 'achieved'. Eq. 6 of
        [QVol] defines this to be the default of 2/3. To be considered achievable, the estimated
        probability of sampling a heavy output at the given depth must be large enough such that
        the one-sided confidence interval of this estimate is greater than the given threshold.
    :param stop_when_fail: if true, the measurement will stop after the first un-achievable depth
    :param show_progress_bar: displays a progress bar for each depth if true.
    :return: dict with key depth: (prob_sample_heavy, ons_sided_conf_interval) gives both the
        estimated probability of sampling a heavy output at each depth and the 2-sigma lower
        bound on this estimate; a depth qualifies as being achievable only if this lower bound
        exceeds the threshold, defined in [QVol] to be 2/3
    """
    if num_circuits < 100:
        warnings.warn("The number of random circuits ran ought to be greater than 100 for results "
                      "to be valid.")
    if qubits is None:
        qubits = qc.qubits()

    if depths is None:
        depths = np.arange(2, len(qubits) + 1)

    results = {}
    for depth in depths:
        log.info("Starting depth {}".format(depth))

        # Use the program generator to implement random model circuits for this depth and compare
        # the outputs to the ideal simulations; get the count of the total number of heavy outputs
        num_heavy = sample_rand_circuits_for_heavy_out(qc, qubits, depth, program_generator,
                                                       num_circuits, num_shots, show_progress_bar)

        prob_sample_heavy, one_sided_conf_intrvl = calculate_prob_est_and_err(num_heavy,
                                                                              num_circuits,
                                                                              num_shots)

        # prob of sampling heavy output must be large enough such that the one-sided confidence
        # interval is larger than the threshold
        is_achievable = one_sided_conf_intrvl > achievable_threshold

        results[depth] = (prob_sample_heavy, one_sided_conf_intrvl)

        if stop_when_fail and not is_achievable:
            break

    return results


def generate_quantum_volume_experiments(depths: Sequence[int], num_circuits: int) -> DataFrame:
    """
    Generate a dataframe with (depth * num_circuits) many rows each populated with an abstract
    description of a model circuit of given depth=width necessary to measure quantum volume.

    See generate_abstract_qv_circuit and the reference [QVol] for more on the structure of each
    circuit and the representation used here.

    :param num_circuits: The number of circuits to run for each depth. Should be > 100
    :param depths: The depths to measure. In order to properly lower bound the quantum volume of
        a circuit, the depths should start at 2 and increase in increments of 1. Depths greater
        than 4 will take several minutes for data collection. Further, the acquire_heavy_hitters
        step involves a classical simulation that scales exponentially with depth.
    :return: a dataframe with columns "Depth" and "Abstract Ckt" populated with the depth and an
        abstract representation of a model circuit with that depth and width.
    """
    def df_dict():
        for d in depths:
            for _ in range(num_circuits):
                yield OrderedDict({"Depth": d,
                                   "Abstract Ckt": generate_abstract_qv_circuit(d)})
    return DataFrame(df_dict())


def add_programs_to_dataframe(df: DataFrame, qc: QuantumComputer,
                              qubits_at_depth: Dict[int, Sequence[int]] = None,
                              program_generator: Callable[[QuantumComputer, Sequence[int],
                                                           np.ndarray, np.ndarray], Program] =
                              _naive_program_generator) -> DataFrame:
    """
    Passes the abstract circuit description in each row of the dataframe df along to the supplied
    program_generator which yields a program that can be run on the available
    qubits_at_depth[depth] on the given qc resource.

    :param df: a dataframe populated with abstract descriptions of model circuits, i.e. a df
        returned by a call to generate_quantum_volume_experiments.
    :param qc: the quantum resource on which each output program will be run.
    :param qubits_at_depth: the qubits of the qc available for use at each depth, default all
        qubits in the qc for each depth. Any subset of these may actually be used by the program.
    :param program_generator: a method which uses the given qc, its available qubits, and an
        abstract description of the model circuit to produce a PyQuil program implementing the
        circuit using only native gates and the given qubits. This program must respect the
        topology of the qc induced by the given qubits. The default _naive_program_generator uses
        the qc's compiler to achieve this result.
    :return: a copy of df with a new "Program" column populated with native PyQuil programs that
        implement the circuit in "Abstract Ckt" on the qc using a subset of the qubits specified
        as available for the given depth. The used qubits are also recorded in a "Qubits" column.
        Note that although the abstract circuit has depth=width, for the program width >= depth.
    """
    new_df = df.copy()

    depths = new_df["Depth"].values
    circuits = new_df["Abstract Ckt"].values

    if qubits_at_depth is None:
        all_qubits = qc.qubits()  # by default the program can act on any qubit in the computer
        qubits = [all_qubits for _ in circuits]
    else:
        qubits = [qubits_at_depth[depth] for depth in depths]

    programs = [program_generator(qc, qbits, *ckt) for qbits, ckt in zip(qubits, circuits)]
    new_df["Program"] = Series(programs)

    # these are the qubits actually used in the program, a subset of qubits_at_depth[depth]
    new_df["Qubits"] = Series([program.get_qubits() for program in programs])

    return new_df


def acquire_quantum_volume_data(df: DataFrame, qc: QuantumComputer, num_shots: int = 1000,
                                use_active_reset: bool = False) -> DataFrame:
    """
    Runs each program in the dataframe df on the given qc and outputs a copy of df with results.

    :param df: a dataframe populated with PyQuil programs that can be run natively on the given qc,
        i.e. a df returned by a call to add_programs_to_dataframe(df, qc, etc.) with identical qc.
    :param qc: the quantum resource on which to run each program.
    :param num_shots: the number of times to sample the output of each program.
    :param use_active_reset: if true, speeds up the overall computation (only on a real qpu) by
        actively resetting at the start of each program.
    :return: a copy of df with a new "Results" column populated with num_shots many depth-bit arrays
        that can be compared to the Heavy Hitters with a call to bit_array_to_int. There is also
        a column "Run Time" which records the time taken to acquire the data for each program.
    """
    new_df = df.copy()

    def run(q_comp, program, n_shots):
        start = time.time()

        if use_active_reset:
            reset_measure_program = Program(RESET())
            program = reset_measure_program + program

        # run the program num_shots many times
        program.wrap_in_numshots_loop(n_shots)
        executable = q_comp.compiler.native_quil_to_executable(program)

        res = q_comp.run(executable)

        end = time.time()
        return res, end - start

    programs = new_df["Program"].values
    data = [run(qc, program, num_shots) for program in programs]

    results = [datum[0] for datum in data]
    times = [datum[1] for datum in data]

    new_df["Results"] = Series(results)
    new_df["Run Time"] = Series(times)

    # supply the count of heavy hitters sampled if heavy hitters are known.
    if "Heavy Hitters" in new_df.columns.values:
        new_df = count_heavy_hitters_sampled(new_df)

    return new_df


def acquire_heavy_hitters(df: DataFrame) -> DataFrame:
    """
    Runs a classical simulation of each circuit in the dataframe df and records which outputs
    qualify as heavy hitters in a copied df with newly populated "Heavy Hitters" column.

    An output is a heavy hitter if the ideal probability of measuring that output from the
    circuit is greater than the median probability among all possible bitstrings of the same size.

    :param df: a dataframe populated with abstract descriptions of model circuits, i.e. a df
        returned by a call to generate_quantum_volume_experiments.
    :return: a copy of df with a new "Heavy Hitters" column. There is also a column "Sim Time"
        which records the time taken to simulate and collect the heavy hitters for each circuit.
    """
    new_df = df.copy()

    def run(depth, circuit):
        wfn_sim = NumpyWavefunctionSimulator(depth)

        start = time.time()
        heavy_outputs = collect_heavy_outputs(wfn_sim, *circuit)
        end = time.time()

        return heavy_outputs, end - start

    circuits = new_df["Abstract Ckt"].values
    depths = new_df["Depth"].values

    data = [run(d, ckt) for d, ckt in zip(depths, circuits)]

    heavy_hitters = [datum[0] for datum in data]
    times = [datum[1] for datum in data]

    new_df["Heavy Hitters"] = Series(heavy_hitters)
    new_df["Sim Time"] = Series(times)

    # supply the count of heavy hitters sampled if sampling results are known.
    if "Results" in new_df.columns.values:
        new_df = count_heavy_hitters_sampled(new_df)

    return new_df


def count_heavy_hitters_sampled(df: DataFrame) -> DataFrame:
    """
    Given a df populated with both sampled results and the actual heavy hitters, copies the df
    and populates a new column with the number of samples which are heavy hitters.

    :param df: a dataframe populated with sampled results and heavy hitters.
    :return: a copy of df with a new "Num HH Sampled" column.
    """
    new_df = df.copy()

    def count(hh, res):
        num_heavy = 0
        # determine if each result bitstring is a heavy output, as determined from simulation
        for result in res:
            # convert result to int for comparison with heavy outputs.
            output = bit_array_to_int(result)
            if output in hh:
                num_heavy += 1
        return num_heavy

    exp_results = new_df["Results"].values
    heavy_hitters = new_df["Heavy Hitters"].values

    new_df["Num HH Sampled"] = Series([count(hh, exp_res) for hh, exp_res in zip(heavy_hitters,
                                                                                 exp_results)])

    return new_df


def get_results_by_depth(df: DataFrame) -> Dict[int, Tuple[float, float]]:
    """
    Analyzes a dataframe df to determine an estimate of the probability of outputting a heavy
    hitter at each depth in the df, a lower bound on this estimate, and whether that depth was
    achieved.

    The output of this method can be fed directly into extract_quantum_volume_from_results to
    obtain the quantum volume measured.

    :param df: a dataframe populated with results, num hh sampled, and circuits for some number
        of depths.
    :return: for each depth key, provides a tuple of (estimate of probability of outputting hh for
        that depth=width, 2-sigma confidence interval (lower bound) on that estimate). The lower
        bound on the estimate is used to judge whether a depth is considered "achieved" in the
        context of the quantum volume.
    """
    depths = df["Depth"].values

    results = {}
    for depth in depths:
        single_depth = df.loc[df["Depth"] == depth]
        num_shots = len(single_depth["Results"].values[0])
        num_heavy = sum(single_depth["Num HH Sampled"].values)
        num_circuits = len(single_depth["Abstract Ckt"].values)

        prob_est, conf_intrvl = calculate_prob_est_and_err(num_heavy, num_circuits, num_shots)

        results[depth] = (prob_est, conf_intrvl)

    return results


def extract_quantum_volume_from_results(results: Dict[int, Tuple[float, float]]) -> int:
    """
    Provides convenient extraction of quantum volume from the results returned by a default run of
    measure_quantum_volume above

    :param results: results of measure_quantum_volume with sequential depths and their achievability
    :return: the quantum volume, eq. 7 of [QVol]
    """
    depths = sorted(results.keys())

    max_depth = 1
    for depth in depths:
        (_, lower_bound) = results[depth]
        if lower_bound <= 2/3:
            break
        max_depth = depth

    quantum_volume = 2**max_depth
    return quantum_volume
