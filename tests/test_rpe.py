import numpy as np
from numpy import pi
from pyquil.gates import H, RY, RZ
from forest_benchmarking import rpe
from pyquil.noise import damping_after_dephasing
from pandas import Series
from pyquil.quil import Program
from pyquil.quilbase import Measurement
from forest_benchmarking.compilation import basic_compile


def test_state_prep(wfn):
    p = Program()
    rpe.prepare_state(p, 0)
    assert wfn.wavefunction(p).pretty_print() == wfn.wavefunction(Program(H(0))).pretty_print()


def test_generate_single_depth(qvm):
    qvm.qam.random_seed = 5
    expected_outcomes = [0., .5, 1.0, .5]
    for depth in [0, 1, 2, 3, 4]:
        for exp_type in ['X', 'Y']:
            exp = rpe.generate_single_depth_experiment(RZ(np.pi / 2, 0), depth, exp_type)
            idx = ((depth - 1) if exp_type == 'Y' else depth) % 4
            expected = expected_outcomes[idx]
            executable = qvm.compiler.native_quil_to_executable(basic_compile(exp.wrap_in_numshots_loop(5000)))
            result = np.average(qvm.run(executable))
            assert np.allclose(expected, result, atol=.005)


def test_noiseless_RPE(qvm):
    qvm.qam.random_seed = 5
    angle = pi / 4 - .5  # pick arbitrary angle
    experiments = rpe.generate_rpe_experiments(RZ(angle, 0), 7)
    experiments = rpe.acquire_rpe_data(experiments, qvm, multiplicative_factor=10.)
    xs, ys, x_stds, y_stds = rpe.find_expectation_values(experiments)
    result = rpe.robust_phase_estimate(xs, ys, x_stds, y_stds)
    assert np.abs(angle - result) < 2 * np.sqrt(rpe.get_variance_upper_bound(experiments))


def test_noisy_RPE(qvm):
    qvm.qam.random_seed = 5
    angles = pi * np.linspace(2 / 9, 2.0 - 2 / 9, 3)
    num_depths = 6  # max depth of 2^(num_depths - 1)
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
        df["Experiment"] = Series([
            add_damping_dephasing_noise(prog, t1, t2, gate_time).define_noisy_readout(0, p00, p11)
            for prog in df["Experiment"].values])

    tolerance = .1
    # scan over each angle and check that RPE correctly predicts the angle to within .1 radians
    for angle in angles:
        RH = Program(RY(-pi / 4, 0)).inst(RZ(angle, 0)).inst(RY(pi / 4, 0))
        experiments = rpe.generate_rpe_experiments(RH, num_depths, axis=(pi / 4, 0))
        add_noise_to_experiments(experiments, 25 * 10 ** (-6.), 20 * 10 ** (-6.), .92, .87)
        experiments = rpe.acquire_rpe_data(experiments, qvm, multiplicative_factor=5., additive_error=add_error)
        xs, ys, x_stds, y_stds = rpe.find_expectation_values(experiments)
        phase_estimate = rpe.robust_phase_estimate(xs, ys, x_stds, y_stds)
        assert np.allclose(phase_estimate, angle, atol=tolerance)
