from typing import List, Sequence, Tuple, Callable, Dict, Iterator
import warnings
from tqdm import tqdm
import numpy as np
from statistics import median
from copy import copy

from pyquil.api import QuantumComputer
from pyquil.numpy_simulator import NumpyWavefunctionSimulator
from pyquil.quil import DefGate, Program, Pragma
from rpcq.messages import TargetDevice
from rpcq._utils import RPCErrorError

from forest.benchmarking.operator_tools.random_operators import haar_rand_unitary
from forest.benchmarking.utils import bit_array_to_int
import logging
log = logging.getLogger(__name__)


def _naive_program_generator(qc: QuantumComputer, qubits: Sequence[int],
                             permutations: Sequence[np.ndarray], gates: np.ndarray) -> Program:
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
    prog = Program(Pragma('INITIAL_REWIRING', ['"PARTIAL"']))
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
        the model quantum circuit of [QVol]_ for a given depth.
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
                                                                    Sequence[np.ndarray],
                                                                    np.ndarray], Program],
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
                                                        Sequence[np.ndarray], np.ndarray],
                                                       Program] = _naive_program_generator,
                           num_circuits: int = 100, num_shots: int = 1000,
                           depths: np.ndarray = None, achievable_threshold: float = 2/3,
                           stop_when_fail: bool = True, show_progress_bar: bool = False) \
        -> Dict[int, Tuple[float, float]]:
    """
    Measures the quantum volume of a quantum resource, as described in [QVol]_.

    By default this method scans increasing depths from 2 to len(qubits) and tests whether the qc
    can adequately implement random model circuits on depth-many qubits such that the given
    depth is 'achieved'. A model circuit depth is achieved if the sample distribution for a
    sample of num_circuits many randomly generated model circuits of the given depth sufficiently
    matches the ideal distribution of that circuit (See Eq. 6  of [QVol]_). The frequency of
    sampling 'heavy-outputs' is used as a measure of closeness of the circuit distributions. This
    estimated frequency (across all sampled circuits) is reported for each depth along with a
    bool which indicates whether that depth was achieved. The logarithm of the quantum volume is by
    definition the largest achievable depth of the circuit; see
    :func:`extract_quantum_volume_from_results` for obtaining the quantum volume from the results
    returned by this method.

    .. [QVol] Validating quantum computers using randomized model circuits.
           Cross et al.
           arXiv:1811.12926v1  (2018).
           https://arxiv.org/abs/1811.12926

    :param qc: the quantum resource whose volume you wish to measure
    :param qubits: available qubits on which to act during measurement. Default all qubits in qc.
    :param program_generator: a method which

        1) takes in a quantum computer, the qubits on that
            computer available for use, a series of sequences representing the qubit permutations
            in a model circuit, an array of matrices representing the 2q gates in the model circuit
        2) outputs a native quil program that implements the circuit and measures the appropriate
            qubits in the order implicitly dictated by the model circuit representation created in
            sample_rand_circuits_for_heavy_out.

        The default option simply picks the smallest qubit labels and lets the compiler do the rest.
    :param num_circuits: number of unique random circuits that will be sampled.
    :param num_shots: number of shots for each circuit sampled.
    :param depths: the circuit depths to scan over. Defaults to all depths from 2 to len(qubits)
    :param achievable_threshold: threshold at which a depth is considered 'achieved'. Eq. 6 of
        [QVol]_ defines this to be the default of 2/3. To be considered achievable, the estimated
        probability of sampling a heavy output at the given depth must be large enough such that
        the one-sided confidence interval of this estimate is greater than the given threshold.
    :param stop_when_fail: if true, the measurement will stop after the first un-achievable depth
    :param show_progress_bar: displays a progress bar for each depth if true.
    :return: dict with key depth: (prob_sample_heavy, ons_sided_conf_interval) gives both the
        estimated probability of sampling a heavy output at each depth and the 2-sigma lower
        bound on this estimate; a depth qualifies as being achievable only if this lower bound
        exceeds the threshold, defined in [QVol]_ to be 2/3
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


def count_heavy_hitters_sampled(qc_results: Iterator[np.ndarray],
                                heavy_hitters: Iterator[List[int]]) -> Iterator[int]:
    """
    Simple helper to count the number of heavy hitters sampled given the sampled results for a
    number of circuits along with the the actual heavy hitters for each circuit.

    :param qc_results: results from running each circuit on a quantum computer.
    :param heavy_hitters: the heavy hitters for each circuit (presumably calculated through
        simulating the circuit classically)
    :return: the number of samples which were heavy for each circuit.
    """
    for results, hh_list in zip(qc_results, heavy_hitters):
        num_heavy = 0
        # determine if each result bitstring is a heavy output, as determined from simulation
        for result in results:
            # convert result to int for comparison with heavy outputs.
            output = bit_array_to_int(result)
            if output in hh_list:
                num_heavy += 1
        yield num_heavy


def get_prob_sample_heavy_by_depth(depths: Iterator[int], num_hh_sampled: Iterator[int],
                                   num_shots: Iterator[int]) -> Dict[int, Tuple[float, float]]:
    """
    Analyzes the given information for each circuit to determine [an estimate of the probability of
    outputting a heavy hitter at each depth, a lower bound on this estimate, and whether that
    depth was achieved]

    The output of this method can be fed directly into extract_quantum_volume_from_results to
    obtain the quantum volume measured.

    :param depths: the depth of each circuit
    :param num_hh_sampled: the number of heavy hitters sampled from each circuit
    :param num_shots: the number of shots / total number of samples from each circuit
    :return: for each depth key, provides a tuple of (estimate of probability of outputting hh for
        that depth=width, 2-sigma confidence interval (lower bound) on that estimate). The lower
        bound on the estimate is used to judge whether a depth is considered "achieved" in the
        context of the quantum volume.
    """
    nheavy_by_depth = {}
    for depth, num_heavy, n_shots in zip(depths, num_hh_sampled, num_shots):
        if depth not in nheavy_by_depth.keys():
            nheavy_by_depth[depth] = ([num_heavy], n_shots)
        else:
            nheavy_by_depth[depth][0].append(num_heavy)
            assert n_shots == nheavy_by_depth[depth][1], 'The number of shots should be the same ' \
                                                         'for each circuit of a given depth.'

    results_by_depth = {}
    for depth, (n_heavy, n_shots) in nheavy_by_depth.items():
        prob_est, conf_intrvl = calculate_prob_est_and_err(sum(n_heavy), len(n_heavy), n_shots)
        results_by_depth[depth] = (prob_est, conf_intrvl)

    return results_by_depth


def extract_quantum_volume_from_results(results: Dict[int, Tuple[float, float]]) -> int:
    """
    Provides convenient extraction of quantum volume from the results returned by a default run of
    measure_quantum_volume above

    :param results: results of measure_quantum_volume with sequential depths and their achievability
    :return: the quantum volume, eq. 7 of [QVol]_
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
