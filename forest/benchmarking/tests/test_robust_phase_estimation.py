import numpy as np
from numpy import pi
from pyquil.gates import I, H, RY, RZ, RX
from pyquil.noise import damping_after_dephasing
from pyquil.quil import Program
from pyquil.quilbase import Measurement

import forest.benchmarking.robust_phase_estimation as rpe
from forest.benchmarking.observable_estimation import ObservablesExperiment, estimate_observables


def test_expectations_at_depth(qvm):
    qvm.qam.random_seed = 5
    q = 0
    qubits = (q, )
    expected_outcomes = [1., 0, -1., 0]
    for depth in [0, 1, 2, 3, 4]:
        prep, meas, settings = rpe.all_eigenvector_prep_meas_settings(qubits, I(q))
        depth_many_rot = [RZ(pi/2, q) for _ in range(depth)]
        program = Program(prep) + sum(depth_many_rot, Program()) + Program(meas)
        expt = ObservablesExperiment(list(settings), program)

        results = list(estimate_observables(qvm, expt))

        for res in results:
            meas_dir = res.setting.observable[q]
            idx = ((depth - 1) if meas_dir == 'Y' else depth) % 4
            expected = expected_outcomes[idx]
            exp = res.expectation
            assert np.allclose(expected, exp, atol=.05)


def test_noiseless_rpe(qvm):
    qvm.qam.random_seed = 5
    angle = pi / 4 - .5  # pick arbitrary angle
    q = 0
    num_depths = 7
    mult_factor = 10
    expts = rpe.generate_rpe_experiments(RZ(angle, q),
                                         *rpe.all_eigenvector_prep_meas_settings([q], I(q)),
                                         num_depths=num_depths)
    results = rpe.acquire_rpe_data(qvm, expts, multiplicative_factor=mult_factor)
    est = rpe.robust_phase_estimate([q], results)
    assert np.abs(angle - est) < 2 * np.sqrt(rpe.get_variance_upper_bound(num_depths, mult_factor))


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

    def add_noise_to_experiments(expts, t1, t2, p00, p11, q):
        gate_time = 200 * 10 ** (-9)
        for ex in expts:
            ex.program = add_damping_dephasing_noise(ex.program, t1, t2,
                                                     gate_time).define_noisy_readout(q, p00, p11)

    tolerance = .1
    # scan over each angle and check that RPE correctly predicts the angle to within .1 radians
    for angle in angles:
        RH = Program(RY(-pi / 4, q)).inst(RZ(angle, q)).inst(RY(pi / 4, q))
        evecs = rpe.bloch_rotation_to_eigenvectors(pi / 4, q)
        cob_matrix = rpe.get_change_of_basis_from_eigvecs(evecs)
        cob = rpe.change_of_basis_matrix_to_quil(qvm, [q], cob_matrix)
        prep, meas, settings = rpe.all_eigenvector_prep_meas_settings([q], cob)
        expts = rpe.generate_rpe_experiments(RH, prep, meas, settings, num_depths=7)
        add_noise_to_experiments(expts, 25 * 10 ** (-6.), 20 * 10 ** (-6.), .92, .87, q)
        results = rpe.acquire_rpe_data(qvm, expts, multiplicative_factor=5.,
                                       additive_error=add_error)
        phase_estimate = rpe.robust_phase_estimate([q], results)
        assert np.allclose(phase_estimate, angle, atol=tolerance)


def test_do_rpe(qvm):
    angles = [-pi / 2, pi]
    qubits = [0, 1]
    qubit_groups = [(qubit,) for qubit in qubits]
    changes_of_basis = [H(qubit) for qubit in qubits]

    for angle in angles:
        rotation = Program([RX(angle, qubit) for qubit in qubits])
        phases, expts, ress = rpe.do_rpe(qvm, rotation, changes_of_basis, qubit_groups,
                                         num_depths=6)
        for group in qubit_groups:
            assert np.allclose(phases[group], angle % (2*pi), atol=.1)
