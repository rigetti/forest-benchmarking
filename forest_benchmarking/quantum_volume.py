from typing import List, Sequence, Tuple, Callable
import warnings
import logging
log = logging.getLogger(__name__)
from tqdm import tqdm
import numpy as np
from statistics import median
from collections import OrderedDict
from pandas import DataFrame, Series
import time
from pyquil.api import QuantumComputer
from pyquil.numpy_simulator import NumpyWavefunctionSimulator
from pyquil.quil import DefGate, Program
from pyquil.gates import RESET

from forest_benchmarking.random_operators import haar_rand_unitary
from forest_benchmarking.utils import bit_array_to_int


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
    num_measure_qubits = len(permutations[0])
    # at present, naively select the minimum number of qubits to run on
    qubits = qubits[:num_measure_qubits]

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
            prog += G(int(qubits[perm[gate_idx]]), int(qubits[perm[gate_idx+1]]))

    ro = prog.declare("ro", "BIT", len(qubits))
    for idx, qubit in enumerate(qubits):
        prog.measure(qubit, ro[idx])

    native_quil = qc.compiler.quil_to_native_quil(prog)

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
        # generate a simple list representation for each permutation of the depth many qubits
        permutations = [np.random.permutation(range(depth)) for _ in range(depth)]

        # generate a matrix representation of each 2q gate in the circuit
        num_gates_per_layer = depth // 2
        gates = np.asarray([[haar_rand_unitary(4) for _ in range(num_gates_per_layer)]
                            for _ in range(depth)])

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


def quantum_volume_dataframe(qubits, num_circuits, depths):

    def df_dict():
        for d in depths:
            for _ in range(num_circuits):
                yield OrderedDict({"Qubits": qubits,
                                   "Depth": d})

    return DataFrame(df_dict())


def generate_qv_circuit(depth):
    # generate a simple list representation for each permutation of the depth many qubits
    permutations = [np.random.permutation(range(depth)) for _ in range(depth)]

    # generate a matrix representation of each 2q gate in the circuit
    num_gates_per_layer = depth // 2
    gates = np.asarray([[haar_rand_unitary(4) for _ in range(num_gates_per_layer)]
                        for _ in range(depth)])

    return permutations, gates


def add_circuits_to_dataframe(df):
    new_df = df.copy()
    new_df["Circuit"] = Series([generate_qv_circuit(d) for d in new_df["Depth"].values])
    return new_df


def acquire_qv_data(df, qc, num_shots, use_active_reset = False):
    new_df = df.copy()

    total_start = time.time()

    def run(qc, qbits, circuit):
        start = time.time()

        program = _naive_program_generator(qc, qbits, *circuit)
        actual_qubits = program.get_qubits()

        if use_active_reset:
            reset_measure_program = Program(RESET())
            program = reset_measure_program + program

        # run the program num_shots many times
        program.wrap_in_numshots_loop(num_shots)
        executable = qc.compiler.native_quil_to_executable(program)

        res = qc.run(executable)

        end = time.time()
        return res, end - start, actual_qubits

    qubits = new_df["Qubits"].values
    circuits = new_df["Circuit"].values
    data = [run(qc, qbits, circuit) for qbits, circuit in zip(qubits, circuits)]

    results = [datum[0] for datum in data]
    times = [datum[1] for datum in data]
    act_qubits = [datum[2] for datum in data]

    new_df["Results"] = Series(results)
    new_df["RunTime"] = Series(times)
    new_df["Qubits"] = Series(act_qubits)

    total_end = time.time()

    return new_df, total_end-total_start


def acquire_heavy_hitters(df):
    new_df = df.copy()

    total_start = time.time()

    def run(depth, circuit):
        wfn_sim = NumpyWavefunctionSimulator(depth)

        start = time.time()
        heavy_outputs = collect_heavy_outputs(wfn_sim, *circuit)
        end = time.time()
        return heavy_outputs, end - start

    circuits = new_df["Circuit"].values
    depths = new_df["Depth"].values

    data = [run(d, ckt) for d, ckt in zip(depths, circuits)]

    heavy_hitters = [datum[0] for datum in data]
    times = [datum[1] for datum in data]

    new_df["HeavyHitters"] = Series(heavy_hitters)
    new_df["SimTime"] = Series(times)

    def count_hh_samples(hh, res):
        num_heavy = 0
        # determine if each result bitstring is a heavy output, as determined from simulation
        for result in res:
            # convert result to int for comparison with heavy outputs.
            output = bit_array_to_int(result)
            if output in hh:
                num_heavy += 1
        return num_heavy

    exp_results = new_df["Results"].values

    new_df["NumHHSampled"] = Series([count_hh_samples(hh, exp_res) for hh, exp_res in zip(
        heavy_hitters, exp_results)])

    total_end = time.time()

    return new_df, total_end - total_start


def get_results_by_depth(df):
    depths = df["Depth"].values

    results = {}
    for depth in depths:
        single_depth = df.loc[df["Depth"] == depth]
        num_shots = len(single_depth["Results"].values[0])
        num_heavy = sum(single_depth["NumHHSampled"].values)
        num_circuits = len(single_depth["Circuit"].values)

        total_sampled_outputs = num_circuits * num_shots
        prob_sample_heavy = num_heavy / total_sampled_outputs

        # Eq. (C3) of [QVol]. Assume that num_heavy/num_shots is worst-case binomial with param
        # num_circuits and take gaussian approximation. Get 2 sigma one-sided confidence interval.
        one_sided_confidence_interval = prob_sample_heavy - \
                                        2 * np.sqrt(
            num_heavy * (num_shots - num_heavy / num_circuits)) / total_sampled_outputs

        results[depth] = (prob_sample_heavy, one_sided_confidence_interval)

    return results


def measure_quantum_volume(qc: QuantumComputer, qubits: Sequence[int] = None,
                           program_generator: Callable[[QuantumComputer, Sequence[int],
                                                        np.ndarray, np.ndarray], Program] =
                           _naive_program_generator,
                           num_circuits: int = 100, num_shots: int = 1000,
                           depths: np.ndarray = None, achievable_threshold: float = 2/3,
                           stop_when_fail: bool = True, show_progress_bar: bool = False) \
                            -> List[Tuple[int, float, float, bool]]:
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
            Cross et al., arXiv:1811.12926v1, Nov 2018
            https://arxiv.org/pdf/1811.12926.pdf

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
    :return: a list of all tuples (depth, prob_sample_heavy, is_achievable) indicating the
        estimated probability of sampling a heavy output at each depth and whether that depth
        qualifies as being achievable.
    """

    if num_circuits < 100:
        warnings.warn("The number of random circuits ran ought to be greater than 100 for results "
                      "to be valid.")
    if qubits is None:
        qubits = qc.qubits()

    if depths is None:
        depths = np.arange(2, len(qubits) + 1)

    results = []
    for depth in depths:
        log.info("Starting depth {}".format(depth))

        # Use the program generator to implement random model circuits for this depth and compare
        # the outputs to the ideal simulations; get the count of the total number of heavy outputs
        num_heavy = sample_rand_circuits_for_heavy_out(qc, qubits, depth, program_generator,
                                                       num_circuits, num_shots, show_progress_bar)

        total_sampled_outputs = num_circuits * num_shots
        prob_sample_heavy = num_heavy/total_sampled_outputs

        # Eq. (C3) of [QVol]. Assume that num_heavy/num_shots is worst-case binomial with param
        # num_circuits and take gaussian approximation. Get 2 sigma one-sided confidence interval.
        one_sided_confidence_interval = prob_sample_heavy - \
            2 * np.sqrt(num_heavy * (num_shots - num_heavy / num_circuits)) / total_sampled_outputs

        # prob of sampling heavy output must be large enough such that the one-sided confidence
        # interval is larger than the threshold
        is_achievable = one_sided_confidence_interval > achievable_threshold

        results.append((depth, prob_sample_heavy, one_sided_confidence_interval, is_achievable))

        if stop_when_fail and not is_achievable:
            break

    return results


def extract_quantum_volume_from_results(results: List[Tuple[int, float, bool]]) -> int:
    """
    Provides convenient extraction of quantum volume from the results returned by a default run of
    measure_quantum_volume above

    :param results: results of measure_quantum_volume with sequential depths and their achievability
    :return: the quantum volume, eq. 7 of [QVol]
    """
    max_depth = 1
    for (d, prob, is_ach) in results:
        if not is_ach:
            break
        max_depth = d

    quantum_volume = 2**max_depth
    return quantum_volume
