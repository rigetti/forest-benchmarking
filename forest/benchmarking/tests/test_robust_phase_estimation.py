import numpy as np
from numpy import pi
from pandas import Series
from pyquil.gates import I, H, RY, RZ
from pyquil.noise import damping_after_dephasing
from pyquil.quil import Program
from pyquil.quilbase import Measurement

import forest.benchmarking.robust_phase_estimation as rpe
from forest.benchmarking.stratified_experiment import StratifiedExperiment, Layer, \
    acquire_stratified_data


def test_expectations_at_depth(qvm):
    qvm.qam.random_seed = 5
    q = 0
    qubits = (q, )
    expected_outcomes = [1., 0, -1., 0]
    for depth in [0, 1, 2, 3, 4]:
        prep, meas, settings = rpe.all_eigenvector_prep_meas_settings(qubits, I(q))
        layers = [Layer(depth, [RZ(pi/2, q) for _ in range(depth)], tuple(settings), qubits)]
        expt = StratifiedExperiment(tuple(layers), qubits)
        result = acquire_stratified_data(qvm, [expt], num_shots=500, parallelize_layers=True)[0]
        for meas_dir in ['X', 'Y']:
            idx = ((depth - 1) if meas_dir == 'Y' else depth) % 4
            expected = expected_outcomes[idx]
            exp = [res.expectation for res in result.layers[0].results if
                   res.setting.out_operator[q] == meas_dir][0]
            assert np.allclose(expected, exp, atol=.05)


def test_noiseless_rpe(qvm):
    qvm.qam.random_seed = 5
    angle = pi / 4 - .5  # pick arbitrary angle
    q = 0
    expt = rpe.generate_rpe_experiment(RZ(angle, q),
                                       *rpe.all_eigenvector_prep_meas_settings([q], I(q)),
                                       num_depths=7)
    res = rpe.acquire_rpe_data(qvm, [expt], multiplicative_factor=10.)[0]
    result = rpe.robust_phase_estimate(res)
    assert np.abs(angle - result) < 2 * np.sqrt(rpe.get_variance_upper_bound(res))


def test_noisy_rpe(qvm):
    qvm.qam.random_seed = 5
    angles = pi * np.linspace(2 / 9, 2.0 - 2 / 9, 3)
    add_error = .15
    q = 0

    def add_damping_dephasing_noise(prog, T1, T2, gate_time):
        p = Program()
        p.defgate("noise", np.eye(2))
        p.define_noisy_gate("noise", [q], damping_after_dephasing(T1, T2, gate_time))
        for elem in prog:
            p.inst(elem)
            if isinstance(elem, Measurement):
                continue  # skip measurement
            p.inst(("noise", q))
        return p

    def add_noise_to_experiments(expt, t1, t2, p00, p11, q):
        gate_time = 200 * 10 ** (-9)
        for layer in expt.layers:
            layer.sequence = [
                add_damping_dephasing_noise(prog, t1, t2, gate_time).define_noisy_readout(q, p00,
                                                                                          p11)
                for prog in layer.sequence]

    tolerance = .1
    # scan over each angle and check that RPE correctly predicts the angle to within .1 radians
    for angle in angles:
        RH = Program(RY(-pi / 4, q)).inst(RZ(angle, q)).inst(RY(pi / 4, q))
        evecs = rpe.bloch_rotation_to_eigenvectors(pi / 4, q)
        cob_matrix = rpe.get_change_of_basis_from_eigvecs(evecs)
        cob = rpe.change_of_basis_matrix_to_quil(qvm, [q], cob_matrix)
        prep, meas, settings = rpe.all_eigenvector_prep_meas_settings([q], cob)
        expt = rpe.generate_rpe_experiment(RH, prep, meas, settings, num_depths=7)
        add_noise_to_experiments(expt, 25 * 10 ** (-6.), 20 * 10 ** (-6.), .92, .87, q)
        expt = rpe.acquire_rpe_data(qvm, [expt], multiplicative_factor=5.,
                                    additive_error=add_error)[0]
        phase_estimate = rpe.robust_phase_estimate(expt)
        assert np.allclose(phase_estimate, angle, atol=tolerance)
