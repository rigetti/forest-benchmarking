import numpy as np
from forest.benchmarking.compilation import basic_compile
from numpy import pi
from pandas import Series
from pyquil.gates import I, H, RY, RZ
from pyquil.noise import damping_after_dephasing
from pyquil.quil import Program
from pyquil.quilbase import Measurement

import forest.benchmarking.robust_phase_estimation as rpe
from forest.benchmarking.robust_phase_estimation import _run_rpe_program, _make_prog_from_df


def test_single_depth(qvm):
    qvm.qam.random_seed = 5
    expected_outcomes = [0., .5, 1.0, .5]
    for depth in [0, 1, 2, 3, 4]:
        for meas_dir in ['X', 'Y']:
            row = Series({'Depth': depth,
                          "Measure Direction": meas_dir,
                          "Rotation": RZ(np.pi / 2, 0),
                          "Change of Basis": Program(I(0)),
                          "Measure Qubits": [0]
                          })
            prog = _make_prog_from_df(row)
            idx = ((depth - 1) if meas_dir == 'Y' else depth) % 4
            expected = expected_outcomes[idx]
            result = np.average(_run_rpe_program(qvm, prog, [[0]], 5000))
            assert np.allclose(expected, result, atol=.005)


def test_noiseless_RPE(qvm):
    qvm.qam.random_seed = 5
    angle = pi / 4 - .5  # pick arbitrary angle
    expt = rpe.generate_rpe_experiment(RZ(angle, 0), I(0), num_depths=7)
    expt = rpe.acquire_rpe_data(qvm, expt, multiplicative_factor=10.)
    xs, ys, x_stds, y_stds = rpe.get_moments(expt)
    result = rpe.estimate_phase_from_moments(xs, ys, x_stds, y_stds)
    assert np.abs(angle - result) < 2 * np.sqrt(rpe.get_variance_upper_bound(expt))
    # test that wrapper yields same result
    result = rpe.robust_phase_estimate(expt)
    assert np.abs(angle - result) < 2 * np.sqrt(rpe.get_variance_upper_bound(expt))



def test_noisy_RPE(qvm):
    qvm.qam.random_seed = 5
    angles = pi * np.linspace(2 / 9, 2.0 - 2 / 9, 3)
    add_error = .15

    def add_damping_dephasing_noise(prog, T1, T2, gate_time):
        p = Program()
        p.defgate("noise", np.eye(2))
        p.define_noisy_gate("noise", [0], damping_after_dephasing(T1, T2, gate_time))
        for elem in prog:
            p.inst(elem)
            if isinstance(elem, Measurement):
                continue  # skip measurement
            p.inst(("noise", 0))
        return p

    def add_noise_to_experiments(df, t1, t2, p00, p11):
        gate_time = 200 * 10 ** (-9)
        df["Program"] = Series([
            add_damping_dephasing_noise(prog, t1, t2, gate_time).define_noisy_readout(0, p00, p11)
            for prog in df["Program"].values])

    tolerance = .1
    # scan over each angle and check that RPE correctly predicts the angle to within .1 radians
    for angle in angles:
        RH = Program(RY(-pi / 4, 0)).inst(RZ(angle, 0)).inst(RY(pi / 4, 0))
        evecs = rpe.bloch_rotation_to_eigenvectors(pi / 4, 0)
        cob = rpe.get_change_of_basis_from_eigvecs(evecs)
        expt = rpe.generate_rpe_experiment(RH, cob, num_depths=7)
        expt = rpe.add_programs_to_rpe_dataframe(qvm, expt)
        add_noise_to_experiments(expt, 25 * 10 ** (-6.), 20 * 10 ** (-6.), .92, .87)
        expt = rpe.acquire_rpe_data(qvm, expt, multiplicative_factor=5., additive_error=add_error)
        phase_estimate = rpe.robust_phase_estimate(expt)
        assert np.allclose(phase_estimate, angle, atol=tolerance)
