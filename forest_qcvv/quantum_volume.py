from typing import Union, List, Sequence, Tuple
import warnings

import numpy as np
from pyquil.api import QuantumComputer, WavefunctionSimulator
from pyquil.gates import I
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


def _naive_program_generator(permutations: np.ndarray, gates: np.ndarray) -> Program:
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
            p += G(int(perm[gate_idx]), int(perm[gate_idx + 1]))
    return p


def collect_heavy_outputs(wfn_sim: WavefunctionSimulator, program: Program, qubits: Sequence[int])\
                                                                                    -> List[int]:
    """
    Uses the provided wfn_sim to calculate the probability of measuring each bitstring from the
    output of the program; those 'heavy' bitstrings which are output with greater than median
    probability among all possible bitstrings on the given qubits are collected and returned.

    :param wfn_sim: a WavefunctionSimulator that can simulate the provided program
    :param program: a PyQuil program whose heavy outputs are returned
    :param qubits: the qubits measured in the program. It is assumed that program get_qubits() is a
        subset of the input sequence of qubits.
    :return: a list of the heavy outputs, represented as ints
    """
    # ensure non-active qubits are reported by adding identities to program
    all_qubit_program = program.inst([I(int(q)) for q in qubits])
    wfn = wfn_sim.wavefunction(all_qubit_program)

    # get the indices of the sorted probabilities and store the first half, i.e. those which have
    # greater than median probability. wfn implicitly lists bitstring indices with qubit 0 rightmost
    sorted_bitstring_indices = np.argsort(wfn.probabilities())
    heavy_outputs = sorted_bitstring_indices[2 ** (len(qubits) - 1):]

    return heavy_outputs


def sample_rand_circuits_for_heavy_out(qc: QuantumComputer, wfn_sim: WavefunctionSimulator,
                                       depth: int, qubits: Sequence[int] = None,
                                       num_circuits: int = 100, num_shots: int = 1000) -> int:
    """
    This method performs the bulk of the work in the quantum volume measurement; for the given
    depth, num_circuits many random model circuits are generated, the heavy outputs are
    determined from the ideal output distribution of each circuit, and a PyQuil implementation of
    the model circuit is run on the qc. The total number of sampled heavy outputs is returned.

    :param qc: the quantum resource that will implement the PyQuil program for each model circuit
    :param wfn_sim: used to simulate the ideal output probabilities for each model circuit
    :param depth: the depth (and width in num of qubits) of the model circuits
    :param qubits: the qubits in the qc that will be measured for heavy output sampling. The
        qubit labels should be listed in increasing order so that they are properly compared to the
        wfn_simu results.
    :param num_circuits: the number of random model circuits to sample at this depth; should be >100
    :param num_shots: the number of shots to sample from each model circuit
    :return: the number of heavy outputs sampled among all circuits generated for this depth
    """
    num_heavy = 0
    for _ in range(num_circuits):
        # at present, naively select the first depth many available qubits
        qubits = qubits[:depth]

        num_gates_per_layer = depth // 2
        permutations = [np.random.permutation(qubits) for _ in range(depth)]
        gate_list = np.array([haar_rand_unitary(4) for _ in range(depth * num_gates_per_layer)])
        gates = np.reshape(gate_list, (depth, num_gates_per_layer, 4, 4))

        # generate a PyQuil program for the model circuit.
        program = _naive_program_generator(permutations, gates)

        # simulate the PyQuil program on a wfn_sim. At present, this requires that the program only
        # act on as many qubits as are measured for the heavy outputs.
        heavy_outputs = collect_heavy_outputs(wfn_sim, program, qubits)

        ro = program.declare("ro", "BIT", len(qubits))
        for idx, qubit in enumerate(qubits):
            program.measure(int(qubit), ro[idx])
        program.wrap_in_numshots_loop(num_shots)
        executable = qc.compile(program)
        results = qc.run(executable)

        for result in results:
            # convert result to int for comparison with heavy outputs. Note heavy_outputs
            # follow wfn indexing with smallest qubit in right-most position, so reverse result.
            output = _bit_array_to_int(result[::-1])
            if output in heavy_outputs:
                num_heavy += 1

    return num_heavy


def measure_quantum_volume(qc: QuantumComputer, qubits: Sequence[int] = None,
                           num_circuits: int = 100, num_shots: int = 1000,
                           depths: np.ndarray = None, achievable_threshold: float = 2/3,
                           stop_when_fail: bool = True) \
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

    wfn_sim = WavefunctionSimulator()

    results = []
    for depth in depths:
        num_heavy = sample_rand_circuits_for_heavy_out(qc, wfn_sim, depth, qubits, num_circuits,
                                                       num_shots)

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
