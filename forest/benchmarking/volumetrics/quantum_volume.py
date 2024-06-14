from typing import Tuple, Dict, List, Optional
import numpy as np
from statistics import median

from pyquil.quil import Program, address_qubits, merge_programs
from pyquil.gates import *
from pyquil.numpy_simulator import NumpyWavefunctionSimulator

from forest.benchmarking.utils import bit_array_to_int
from forest.benchmarking.volumetrics._templates import get_quantum_volume_template


def collect_heavy_outputs(wfn_sim: NumpyWavefunctionSimulator,
                          program_array: Dict[int, Dict[int, List[Program]]],
                          measure_qubits: Optional[Dict[int, Dict[int, List[int]]]] = None) \
        -> Dict[int, Dict[int, List[List[int]]]]:
    """
    Collects and returns those 'heavy' bitstrings which are output with greater than median
    probability among all possible bitstrings on the given qubits.

    The method uses the provided wfn_sim to calculate the probability of measuring each bitstring
    from the output of the circuit comprised of the given permutations and gates.

    :param wfn_sim: a NumpyWavefunctionSimulator that can simulate the provided program
    :param program_array: a collection of PyQuil Programs sampled from the circuit family for
        each (width, depth) pair.
    :param measure_qubits: optional list of qubits to measure for each Program in
        `program_array`. By default all qubits in the Program are measured
    :return: a list of the heavy outputs of the circuit, represented as ints
    """
    heavy_output_array = {w: {d: [] for d in d_arr.keys()} for w, d_arr in program_array.items()}

    for w, d_progs in program_array.items():
        for d, ckts in d_progs.items():
            for idx, ckt in enumerate(ckts):
                wfn_sim.reset()
                for gate in ckt:
                    wfn_sim.do_gate(gate)

                if measure_qubits is not None:
                    qubits = measure_qubits[w][d][idx]
                else:
                    qubits = sorted(list(ckt.get_qubits()))

                # Note that probabilities are ordered lexicographically with qubit 0 leftmost.
                # we need to restrict attention to the subset `qubits`
                probs = abs(wfn_sim.wf) ** 2
                probs = probs.reshape([2] * wfn_sim.n_qubits)
                marginal = probs
                for q in reversed(range(wfn_sim.n_qubits)):
                    if q in qubits:
                        continue
                    marginal = np.sum(marginal, axis=q)

                probabilities = marginal.reshape(-1)

                median_prob = median(probabilities)

                # store the integer indices, which implicitly represent the bitstring outcome.
                heavy_outputs = [idx for idx, prob in enumerate(probabilities) if
                                 prob > median_prob]
                heavy_output_array[w][d].append(heavy_outputs)

    return heavy_output_array


def get_success_probabilities(noisy_results, ideal_results):
    """
    For circuit results of various width and depth, calculate the fraction of noisy results
    that are also found in the collection of ideal results for each circuit.

    Quantum volume employs this method to calculate success_probabilities where the ideal_results
    are the heavy hitters of each circuit.

    :param noisy_results: noisy shots from each circuit sampled for each width and depth
    :param ideal_results: a collection of ideal results for each circuit; membership of a noisy
        shot from a particular circuit in the corresponding set of ideal_results constitutes a
        success.
    :return: the estimated success probability for each circuit.
    """
    prob_success = {width: {depth: [] for depth in depth_array.keys()}
                    for width, depth_array in noisy_results.items()}

    assert set(noisy_results.keys()) == set(ideal_results.keys())

    for width, depth_array in prob_success.items():
        for depth in depth_array.keys():

            noisy_ckt_sample_results = noisy_results[width][depth]
            ideal_ckt_sample_results = ideal_results[width][depth]

            # iterate over circuits
            for noisy_shots, targets in zip(noisy_ckt_sample_results, ideal_ckt_sample_results):
                if not isinstance(targets[0], int):
                    targets = [bit_array_to_int(res) for res in targets]

                pr_success = 0
                # determine if each result bitstring is a success, i.e. matches an ideal_result
                for result in noisy_shots:
                    # convert result to int for comparison with heavy outputs.
                    output = bit_array_to_int(result)
                    if output in targets:
                        pr_success += 1 / len(noisy_shots)
                prob_success[width][depth].append(pr_success)

    return prob_success


def calculate_success_prob_est_and_err(num_success: int, num_circuits: int, num_shots: int) \
        -> Tuple[float, float]:
    """
    Helper to calculate the estimate for the probability of sampling a successful output at a
    particular depth as well as the 2 sigma one-sided confidence interval on this estimate.

    :param num_success: total number of successful outputs sampled at particular depth across all
        circuits and shots
    :param num_circuits: the total number of model circuits of a particular depth and width whose
        output was sampled
    :param num_shots: the total number of shots taken for each circuit
    :return: estimate for the probability of sampling a successful output at a particular depth as
        well as the 2 sigma one-sided confidence interval on this estimate.
    """
    total_sampled_outputs = num_circuits * num_shots
    prob_sample_heavy = num_success / total_sampled_outputs

    # Eq. (C3) of [QVol]. Assume that num_heavy/num_shots is worst-case binomial with param
    # num_circuits and take gaussian approximation. Get 2 sigma one-sided confidence interval.
    sigma = np.sqrt(num_success * (num_shots - num_success / num_circuits)) / total_sampled_outputs
    one_sided_confidence_interval = prob_sample_heavy - 2 * sigma

    return prob_sample_heavy, one_sided_confidence_interval


def determine_prob_success_lower_bounds(ckt_success_probs, num_shots_per_ckt):
    """
    Wrapper around `calculate_success_prob_est_and_err` to determine success lower bounds for a
    collection of circuits at various depths and widths.

    :param ckt_success_probs:
    :param num_shots_per_ckt:
    :return:
    """
    return {w: {d: calculate_success_prob_est_and_err(
        sum(np.asarray(succ_probs) * num_shots_per_ckt), len(succ_probs), num_shots_per_ckt)[1]
                for d, succ_probs in d_ckt_succ_probs.items()}
            for w, d_ckt_succ_probs in ckt_success_probs.items()}


def determine_successes(ckt_success_probs: Dict[int, Dict[int, List[float]]], num_shots_per_ckt,
                        success_threshold: float = 2 / 3):
    """
    Indicate whether the collection of circuit success probabilities for given width and depth
    recorded in `ckt_success_probs` is considered a success with respect to the specified
    `success_threshold` and given the number of shots used to estimate each success probability.

    :param ckt_success_probs:
    :param num_shots_per_ckt:
    :param success_threshold:
    :return:
    """
    lower_bounds = determine_prob_success_lower_bounds(ckt_success_probs, num_shots_per_ckt)
    return {w: {d: lb > success_threshold for d, lb in d_lower_bounds.items()}
            for w, d_lower_bounds in lower_bounds.items()}
