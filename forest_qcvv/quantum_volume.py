from typing import Union, List, Sequence, Tuple
import warnings

import numpy as np
from pyquil.api import QuantumComputer, WavefunctionSimulator
from pyquil.gates import I
from pyquil.quil import DefGate, Program

from forest_qcvv.random_operators import haar_rand_unitary


def _bit_array_to_int(bit_array: Union[Sequence[int], str]) -> int:
    """
    Converts a bit array into an integer where the right-most bit is least significant.

    :param bit_array: an array of bits with right-most bit considered least significant.
    :return: the integer corresponding to the bitstring.
    """
    if type(bit_array) == str:
        bit_array = [int(bit) for bit in bit_array]
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
    prob_dict = wfn.get_outcome_probs()

    sorted_by_prob = sorted([(bitstring, prob_dict[bitstring]) for bitstring in prob_dict],
                            key=lambda pair: pair[1])
    # heavy outputs are those bit strings with measurement probability greater than median
    heavy_outputs = [pair[0] for pair in sorted_by_prob][2 ** (len(qubits) - 1):]
    assert (len(heavy_outputs) == 2 ** (len(qubits) - 1))  # half of possible bitstrings are heavy

    # convert heavy outputs to int; wfn lists smallest qubit number as leftmost
    heavy_outputs = [_bit_array_to_int(output[::-1]) for output in heavy_outputs]
    return heavy_outputs


def measure_quantum_volume(qc: QuantumComputer, qubits: Sequence[int] = None,
                           num_circuits: int = 100, num_shots: int = 1000,
                           depths: np.ndarray = None, achievable_threshold: float = 2/3,
                           stop_when_fail: bool = True) \
                            -> Tuple[int, List[Tuple[int, float, bool]]]:
    """
    A naive implementation of the quantum volume measurement described here:
        [QVol] Validating quantum computers using randomized model circuits
            Cross et al., arXiv:1811.12926v1, Nov 2018
            https://arxiv.org/pdf/1811.12926.pdf
    Currently picks the smallest qubit labels in the QC and
    :param qc: the quantum resource whose volume you wish to measure
    :param qubits: available qubits on which to measure quantum volume. Default all qubits in qc.
    :param num_circuits: number of unique random circuits that will be sampled.
    :param num_shots: number of shots for each circuit sampled.
    :param depths: the circuit depths to scan over. Defaults to all depths from 2 to len(qubits)
    :param achievable_threshold: threshold at which a depth is considered to be achieved.
        Eq. 6 of [QVol] defines this to be the default of 2/3
    :param stop_when_fail: if true, the measurement will stop after the first un-achievable depth
    :return: the quantum volume, equation [7] of [QVol], as well as a list of all tuples
        (depth, prob_sample_heavy, is_achievable)
        indicating the estimated probability of sampling a heavy output at each depth and whether
        that depth qualifies as being achievable.
    """

    if num_circuits < 100:
        warnings.warn("The number of random circuits ran ought to be greater than 100 for results "
                      "to be valid.")
    if qubits is None:
        qubits = qc.qubits()

    if depths is None:
        depths = np.arange(2, len(qubits) + 1)

    wfn_sim = WavefunctionSimulator()

    probs_sample_heavy = []
    for depth in depths:
        print("Starting depth ", depth)
        num_heavy = 0
        for _ in range(num_circuits):
            qubits = np.array(np.arange(depth))
            num_gates_per_layer = depth // 2
            permutations = [np.random.permutation(qubits) for layer in range(depth)]
            gates = np.array([haar_rand_unitary(4)
                              for layer in range(depth)
                              for gate in range(num_gates_per_layer)]).reshape(
                (depth, num_gates_per_layer, 4, 4))

            program = _naive_program_generator(permutations, gates)
            heavy_outputs = collect_heavy_outputs(wfn_sim, program, qubits)

            ro = program.declare("ro", "BIT", len(qubits))
            for idx, qubit in enumerate(qubits):
                program.measure(int(qubit), ro[idx])
            program.wrap_in_numshots_loop(num_shots)
            executable = qc.compile(program)
            results = qc.run(executable)

            for result in results:
                # convert result to int for comparison with heavy outputs
                output = _bit_array_to_int(result)
                if output in heavy_outputs:
                    num_heavy += 1

        total_sampled_outputs = num_circuits * num_shots
        prob_sample_heavy = num_heavy/total_sampled_outputs

        # Eq. (C3) of [QVol]. Assume that num_heavy/num_shots is worst-case binomial with param
        # num_circuits and take gaussian approximation. Get 2 sigma one-sided confidence interval.
        one_sided_confidence_interval = prob_sample_heavy - \
            2 * np.sqrt(num_heavy * (num_shots - num_heavy / num_circuits)) / total_sampled_outputs

        is_achievable = one_sided_confidence_interval > achievable_threshold

        probs_sample_heavy.append((depth, prob_sample_heavy, is_achievable))

        if stop_when_fail and not is_achievable:
            break

    max_depth = 1
    for (d, prob, is_ach) in probs_sample_heavy:
        if not is_ach:
            break
        max_depth = d

    quantum_volume = 2**max_depth
    return quantum_volume, probs_sample_heavy
