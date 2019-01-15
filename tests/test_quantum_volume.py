import numpy as np
import warnings
from forest_qcvv.quantum_volume import measure_quantum_volume

np.random.seed(1)


def test_ideal_sim_heavy_probs(qvm):
    qvm.qam.random_seed = 1

    # silence warning from too few circuits, since 100 circuits is too slow
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quantum_volume, outcomes = measure_quantum_volume(qvm, num_circuits=80, num_shots=50,
                                                          qubits=[0, 1, 2])

    assert quantum_volume == 8
    target_probs = [0.788765, 0.852895]
    probs = [val[1] for val in outcomes]
    assert np.allclose(probs, target_probs, atol=.02)
