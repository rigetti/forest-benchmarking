from tqdm import tqdm
from forest.benchmarking.readout import estimate_joint_confusion_in_set
from forest.benchmarking.direct_fidelity_estimation import \
    generate_exhaustive_process_dfe_experiment, acquire_dfe_data, estimate_dfe
from forest.benchmarking.randomized_benchmarking import average_gate_error_to_rb_decay
from pyquil import Program
from pyquil.api import QuantumComputer, get_benchmarker
from pyquil.noise import NoiseModel, KrausModel, pauli_kraus_map, combine_kraus_maps
from pyquil.device import gates_in_isa
from pyquil.gate_matrices import QUANTUM_GATES


def _do_dfe(qc: QuantumComputer, program: Program, num_shots: int = 10_000):
    bm = get_benchmarker()
    dfe_expt = generate_exhaustive_process_dfe_experiment(program, list(program.get_qubits()), bm)
    dfe_results = acquire_dfe_data(qc, dfe_expt, num_shots=num_shots)
    return estimate_dfe(dfe_results, 'process')[0]


def _depolarizing_channel(fidelity: float, dim: int):
    p = average_gate_error_to_rb_decay(1 - fidelity, dim)
    pr_id = p + (1 - p) / dim
    pr_other = (1 - pr_id) / (dim ** 2 - 1)
    probabilities = [pr_id] + [pr_other] * (dim ** 2 - 1)
    return pauli_kraus_map(probabilities)


def generate_depolarizing_noise_model(qc: QuantumComputer, show_progress_bar: bool = True,
                                      num_readout_shots: int = 10_000, num_dfe_shots: int = 10_000)\
        -> NoiseModel:
    """
    Generate a noise model with depolarizing channel on each native gate and single qubit readout
    error using data gathered on the input qc.

    The readout error is modeled as single qubit uncorrelated classical confusion matrices.

    A depolarizing channel is assumed for each gate with parameter determined from the gate
    fidelity as estimated by DFE with error mitigation.

    Note that the output noise model can be assigned to a QuantumComputer via `qc.qam.noise_model`.

    :param qc: the quantum computer whose noise you would like to simplistically model
    :param show_progress_bar: if true then a tqdm progress bar will be displayed to STDOUT
    :param num_readout_shots: number of shots when estimating each single qubit confusion matrix
    :param num_dfe_shots: number of shots for process DFE experiment on each gate.
    :return: a simplistic noise model with depolarizing channels on each gate as determined by
        data collected on the input qc
    """
    single_qubit_conf_matrices = estimate_joint_confusion_in_set(qc, num_shots=num_readout_shots,
                                                                 joint_group_size=1)
    # the keys need to be lone qubit indices
    single_qubit_conf_matrices = {key[0]: val for key, val in single_qubit_conf_matrices.items()}

    gates = gates_in_isa(qc.get_isa())
    kraus_maps = []
    for g in tqdm(gates, disable=not show_progress_bar):
        if g.name in ('RZ', 'I'):
            # assume these are noiseless
            continue

        targets = tuple(t.index for t in g.qubits)
        if len(g.params) > 0:
            matrix = QUANTUM_GATES[g.name](*g.params)
        else:
            matrix = QUANTUM_GATES[g.name]

        # estimate the fidelity of the gate using DFE
        fidelity = min(_do_dfe(qc, Program(g), num_dfe_shots), 1.0)

        # get kraus operators for the depolarizing channel that would give rise to this fidelity
        noise_kraus = _depolarizing_channel(fidelity, 2 ** len(targets))

        km = (KrausModel(g.name, tuple(g.params), targets,
                         combine_kraus_maps(noise_kraus, [matrix]), fidelity))
        kraus_maps.append(km)

    return NoiseModel(kraus_maps, single_qubit_conf_matrices)
