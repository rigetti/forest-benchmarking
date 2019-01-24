from typing import List, Sequence, Tuple
import warnings
import logging
from tqdm import tqdm

import numpy as np
from pyquil.api import QuantumComputer
from pyquil.numpy_simulator import NumpyWavefunctionSimulator
from pyquil.quil import DefGate, Program

from forest_qcvv.random_operators import haar_rand_unitary


def _bit_array_to_int(bit_array: Sequence[int]) -> int:
    """
    Converts a bit array into an integer where the right-most bit is least significant.

    :param bit_array: an array of bits with right-most bit considered least significant.
    :return: the integer corresponding to the bitstring.
    """
    output = 0
    for bit in bit_array:
        output = (output << 1) | bit
    return output


def _naive_program_generator(permutations: np.ndarray, gates: np.ndarray, qubits: Sequence[int]) \
        -> Program:
    """
    Generates a PyQuil program to implement the circuit which is comprised of the given
    permutations and gates.

    :param permutations: an array of qubit labels
    :param gates: a depth by depth//2 array of matrices representing the 2q gates at each layer.
    :return: a PyQuil program the implements the circuit represented by the input permutations
        and gates
    """
    p = Program()  # include all qubits in program
    for layer_idx, (perm, layer) in enumerate(zip(permutations, gates)):
        for gate_idx, gate in enumerate(layer):
            # get the Quil definition for the new gate
            g_definition = DefGate("LYR" + str(layer_idx) + "_RAND" + str(gate_idx), gate)
            # get the gate constructor
            G = g_definition.get_constructor()
            # add definition to program
            p += g_definition
            # add gate to program, acting on properly permuted qubits
            p += G(int(qubits[perm[gate_idx]]), int(qubits[perm[gate_idx+1]]))
    return p


def collect_heavy_outputs(wfn_sim: NumpyWavefunctionSimulator, permutations: np.ndarray,
                          gates: np.ndarray) -> List[int]:
    """
    Uses the provided wfn_sim to calculate the probability of measuring each bitstring from the
    output of the circuit comprised of the given permutations and gates; those 'heavy' bitstrings
    which are output with greater than median probability among all possible bitstrings on the
    given qubits are collected and returned.

    :param wfn_sim: a NumpyWavefunctionSimulator that can simulate the provided program
    :param permutations: array of depth-many arrays of size n_qubits indicating a qubit permutation
    :param gates: depth by num_gates_per_layer many matrix representations of 2q gates
    :return: a list of the heavy outputs of the circuit, represented as ints
    """
    wfn_sim.reset()

    num_qubits = len(permutations[0])
    for layer_idx, (perm, layer) in enumerate(zip(permutations, gates)):
        for gate_idx, gate in enumerate(layer):
            wfn_sim.do_gate_matrix(gate, (perm[gate_idx], perm[gate_idx+1]))

    probabilities = np.abs(wfn_sim.wf.reshape(-1)) ** 2

    # get the indices of the sorted probabilities and store the first half, i.e. those which have
    # greater than median probability. Qubit 0 is on the left in numpy simulator
    sorted_bitstring_indices = np.argsort(probabilities)
    heavy_outputs = sorted_bitstring_indices[2 ** (num_qubits - 1):]

    return heavy_outputs


def sample_rand_circuits_for_heavy_out(qc: QuantumComputer,
                                       depth: int, qubits: Sequence[int] = None,
                                       num_circuits: int = 100, num_shots: int = 1000,
                                       show_progress_bar: bool = False) -> int:
    """
    This method performs the bulk of the work in the quantum volume measurement; for the given
    depth, num_circuits many random model circuits are generated, the heavy outputs are
    determined from the ideal output distribution of each circuit, and a PyQuil implementation of
    the model circuit is run on the qc. The total number of sampled heavy outputs is returned.

    :param qc: the quantum resource that will implement the PyQuil program for each model circuit
    :param depth: the depth (and width in num of qubits) of the model circuits
    :param qubits: the qubits in the qc that will be measured for heavy output sampling. Qubit
        labels should be listed in proper order so results can be compared to the wfn_sim results.
    :param num_circuits: the number of random model circuits to sample at this depth; should be >100
    :param num_shots: the number of shots to sample from each model circuit
    :param show_progress_bar: displays a progress bar via tqdm if true.
    :return: the number of heavy outputs sampled among all circuits generated for this depth
    """
    wfn_sim = NumpyWavefunctionSimulator(depth)

    num_heavy = 0
    # display progress bar using tqdm
    for _ in tqdm(range(num_circuits), disable=not show_progress_bar):
        # at present, naively select the first depth many available qubits
        qubits = qubits[:depth]

        # generate a simple list representation for each permutation of the depth many qubits
        permutations = [np.random.permutation(range(depth)) for _ in range(depth)]

        # generate a matrix representation of each 2q gate in the circuit
        num_gates_per_layer = depth // 2
        gate_list = np.array([haar_rand_unitary(4) for _ in range(depth * num_gates_per_layer)])
        gates = np.reshape(gate_list, (depth, num_gates_per_layer, 4, 4))

        # generate a PyQuil program for the model circuit
        program = _naive_program_generator(permutations, gates, qubits)

        # classically simulate model circuit represented by the perms and gates for heavy outputs
        heavy_outputs = collect_heavy_outputs(wfn_sim, permutations, gates)

        ro = program.declare("ro", "BIT", len(qubits))
        for idx, qubit in enumerate(qubits):
            program.measure(int(qubit), ro[idx])
        program.wrap_in_numshots_loop(num_shots)
        executable = qc.compile(program)
        results = qc.run(executable)

        for result in results:
            # convert result to int for comparison with heavy outputs.
            output = _bit_array_to_int(result)
            if output in heavy_outputs:
                num_heavy += 1

    return num_heavy


def measure_quantum_volume(qc: QuantumComputer, qubits: Sequence[int] = None,
                           num_circuits: int = 100, num_shots: int = 1000,
                           depths: np.ndarray = None, achievable_threshold: float = 2/3,
                           stop_when_fail: bool = True, show_progress_bar: bool = False) \
                            -> List[Tuple[int, float, bool]]:
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
    :param qubits: available qubits on which to measure quantum volume. Default all qubits in qc.
    :param num_circuits: number of unique random circuits that will be sampled.
    :param num_shots: number of shots for each circuit sampled.
    :param depths: the circuit depths to scan over. Defaults to all depths from 2 to len(qubits)
    :param achievable_threshold: threshold at which a depth is considered to be achieved.
        Eq. 6 of [QVol] defines this to be the default of 2/3
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
    qubits = sorted(qubits)

    if depths is None:
        depths = np.arange(2, len(qubits) + 1)

    results = []
    for depth in depths:
        logging.info("Starting depth {}".format(depth))
        num_heavy = sample_rand_circuits_for_heavy_out(qc, depth, qubits, num_circuits,
                                                       num_shots, show_progress_bar)

        total_sampled_outputs = num_circuits * num_shots
        prob_sample_heavy = num_heavy/total_sampled_outputs

        # Eq. (C3) of [QVol]. Assume that num_heavy/num_shots is worst-case binomial with param
        # num_circuits and take gaussian approximation. Get 2 sigma one-sided confidence interval.
        one_sided_confidence_interval = prob_sample_heavy - \
            2 * np.sqrt(num_heavy * (num_shots - num_heavy / num_circuits)) / total_sampled_outputs

        is_achievable = one_sided_confidence_interval > achievable_threshold

        results.append((depth, prob_sample_heavy, is_achievable))

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
