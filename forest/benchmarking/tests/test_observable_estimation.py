import random
random.seed(1)  # seed random number generation for all calls to rand_ops

import pytest
from pyquil.quilbase import Pragma
from pyquil import get_qc
from pyquil.gates import *
from pyquil.api import WavefunctionSimulator
from pyquil.paulis import sX, sY, sZ, PauliSum
from forest.benchmarking.observable_estimation import *
from forest.benchmarking.observable_estimation import _OneQState,\
    _max_tpb_overlap, _max_weight_operator, _max_weight_state, _one_q_sic_prep


def _generate_random_states(n_qubits, n_terms):
    oneq_states = [SIC0, SIC1, SIC2, SIC3, plusX, minusX, plusY, minusY, plusZ, minusZ]
    all_s_inds = np.random.randint(len(oneq_states), size=(n_terms, n_qubits))
    states = []
    for s_inds in all_s_inds:
        state = functools.reduce(mul, (oneq_states[pi](i) for i, pi in enumerate(s_inds)),
                                 TensorProductState([]))
        states += [state]
    return states


def _generate_random_paulis(n_qubits, n_terms):
    paulis = [sI, sX, sY, sZ]
    all_op_inds = np.random.randint(len(paulis), size=(n_terms, n_qubits))
    operators = []
    for op_inds in all_op_inds:
        op = functools.reduce(mul, (paulis[pi](i) for i, pi in enumerate(op_inds)), sI(0))
        op *= np.random.uniform(-1, 1)
        operators += [op]
    return operators


def test_experiment_setting():
    in_states = _generate_random_states(n_qubits=4, n_terms=7)
    out_ops = _generate_random_paulis(n_qubits=4, n_terms=7)
    for ist, oop in zip(in_states, out_ops):
        expt = ExperimentSetting(ist, oop)
        assert str(expt) == expt.serializable()
        expt2 = ExperimentSetting.from_str(str(expt))
        assert expt == expt2
        assert expt2.in_state == ist
        assert expt2.observable == oop


@pytest.mark.filterwarnings("ignore:ExperimentSetting")
def test_setting_no_in_back_compat():
    out_ops = _generate_random_paulis(n_qubits=4, n_terms=7)
    for oop in out_ops:
        expt = ExperimentSetting(TensorProductState(), oop)
        expt2 = ExperimentSetting.from_str(str(expt))
        assert expt == expt2
        assert expt2.in_state == TensorProductState()
        assert expt2.observable == oop


@pytest.mark.filterwarnings("ignore:ExperimentSetting")
def test_setting_no_in():
    out_ops = _generate_random_paulis(n_qubits=4, n_terms=7)
    for oop in out_ops:
        expt = ExperimentSetting(zeros_state(oop.get_qubits()), oop)
        expt2 = ExperimentSetting.from_str(str(expt))
        assert expt == expt2
        assert expt2.in_state == functools.reduce(mul, [plusZ(q) for q in oop.get_qubits()],
                                                  TensorProductState())
        assert expt2.observable == oop


def test_tomo_experiment():
    expts = [
        ExperimentSetting(TensorProductState(), sX(0) * sY(1)),
        ExperimentSetting(plusZ(0), sZ(0)),
    ]

    suite = ObservablesExperiment(
        settings=expts,
        program=Program(X(0), Y(1))
    )
    assert len(suite) == 2
    for e1, e2 in zip(expts, suite):
        # experiment suite puts in groups of length 1
        assert len(e2) == 1
        e2 = e2[0]
        assert e1 == e2
    prog_str = str(suite).splitlines()[0]
    assert prog_str == 'X 0; Y 1'


def test_tomo_experiment_pre_grouped():
    expts = [
        [ExperimentSetting(TensorProductState(), sX(0) * sI(1)), ExperimentSetting(TensorProductState(), sI(0) * sX(1))],
        [ExperimentSetting(TensorProductState(), sZ(0) * sI(1)), ExperimentSetting(TensorProductState(), sI(0) * sZ(1))],
    ]

    suite = ObservablesExperiment(
        settings=expts,
        program=Program(X(0), Y(1))
    )
    assert len(suite) == 2  # number of groups
    for es1, es2 in zip(expts, suite):
        for e1, e2 in zip(es1, es2):
            assert e1 == e2
    prog_str = str(suite).splitlines()[0]
    assert prog_str == 'X 0; Y 1'


def test_tomo_experiment_empty():
    suite = ObservablesExperiment([], program=Program(X(0)))
    assert len(suite) == 0
    assert str(suite.program) == 'X 0\n'


def test_experiment_deser(tmpdir):
    expts = [
        [ExperimentSetting(TensorProductState(), sX(0) * sI(1)), ExperimentSetting(TensorProductState(), sI(0) * sX(1))],
        [ExperimentSetting(TensorProductState(), sZ(0) * sI(1)), ExperimentSetting(TensorProductState(), sI(0) * sZ(1))],
    ]

    suite = ObservablesExperiment(
        settings=expts,
        program=Program(X(0), Y(1))
    )
    to_json(f'{tmpdir}/suite.json', suite)
    suite2 = read_json(f'{tmpdir}/suite.json')
    assert suite == suite2


@pytest.fixture(params=['clique-removal', 'greedy'])
def grouping_method(request):
    return request.param


def test_expt_settings_share_ntpb():
    expts = [[ExperimentSetting(zeros_state([0, 1]), sX(0) * sI(1)), ExperimentSetting(zeros_state([0, 1]), sI(0) * sX(1))],
             [ExperimentSetting(zeros_state([0, 1]), sZ(0) * sI(1)), ExperimentSetting(zeros_state([0, 1]), sI(0) * sZ(1))]]
    for group in expts:
        for e1, e2 in itertools.combinations(group, 2):
            assert _max_weight_state([e1.in_state, e2.in_state]) is not None
            assert _max_weight_operator([e1.observable, e2.observable]) is not None


def test_group_experiments(grouping_method):
    expts = [  # cf above, I removed the inner nesting. Still grouped visually
        ExperimentSetting(TensorProductState(), sX(0) * sI(1)), ExperimentSetting(TensorProductState(), sI(0) * sX(1)),
        ExperimentSetting(TensorProductState(), sZ(0) * sI(1)), ExperimentSetting(TensorProductState(), sI(0) * sZ(1)),
    ]
    suite = ObservablesExperiment(expts, Program())
    grouped_suite = group_settings(suite, method=grouping_method)
    assert len(suite) == 4
    assert len(grouped_suite) == 2


def test_experiment_result_compat():
    er = ExperimentResult(
        setting=ExperimentSetting(plusX(0), sZ(0)),
        expectation=0.9,
        std_err=0.05,
        total_counts=100,
    )
    assert str(er) == 'X+_0→(1+0j)*Z0: 0.9 +- 0.05'


def test_experiment_result():
    er = ExperimentResult(
        setting=ExperimentSetting(plusX(0), sZ(0)),
        expectation=0.9,
        std_err=0.05,
        total_counts=100,
    )
    assert str(er) == 'X+_0→(1+0j)*Z0: 0.9 +- 0.05'


def test_estimate_observables():
    expts = [
        ExperimentSetting(TensorProductState(), o1 * o2)
        for o1, o2 in itertools.product([sI(0), sX(0), sY(0), sZ(0)], [sI(1), sX(1), sY(1), sZ(1)])
    ]
    suite = ObservablesExperiment(expts, program=Program(X(0), CNOT(0, 1)))
    assert len(suite) == 4 * 4
    gsuite = group_settings(suite)
    assert len(gsuite) == 3 * 3  # can get all the terms with I for free in this case

    qc = get_qc('2q-qvm')
    qc.qam.random_seed = 1
    for res in estimate_observables(qc, gsuite, num_shots=1000):
        if res.setting.observable in [sI(), sZ(0), sZ(1), sZ(0) * sZ(1)]:
            assert np.abs(res.expectation) > 0.9
        else:
            assert np.abs(res.expectation) < 0.1



def _random_2q_programs(n_progs=3):
    """Generate random programs that consist of single qubit rotations, a CZ, and single
    qubit rotations.
    """
    r = random.Random(52)

    def RI(qubit, angle):
        # throw away angle so we can randomly choose the identity
        return I(qubit)

    def _random_1q_gate(qubit):
        return r.choice([RI, RX, RY, RZ])(qubit=qubit, angle=r.uniform(0, 2 * pi))

    for _ in range(n_progs):
        prog = Program()
        prog += _random_1q_gate(0)
        prog += _random_1q_gate(1)
        prog += CZ(0, 1)
        prog += _random_1q_gate(0)
        prog += _random_1q_gate(1)
        yield prog


def test_estimate_observables_many_progs():
    # for "random programs" calculate wfn.expectation see if operator estimation gets it right
    expts = [
        ExperimentSetting(TensorProductState(), o1 * o2)
        for o1, o2 in itertools.product([sI(0), sX(0), sY(0), sZ(0)], [sI(1), sX(1), sY(1), sZ(1)])
    ]

    qc = get_qc('2q-qvm')
    qc.qam.random_seed = 1
    qc.qam.random_seed = 0
    # default n_progs in _random_2q_programs is now three
    for prog in _random_2q_programs():
        suite = ObservablesExperiment(expts, program=prog)
        assert len(suite) == 4 * 4
        gsuite = group_settings(suite)
        assert len(gsuite) == 3 * 3  # can get all the terms with I for free in this case

        wfn = WavefunctionSimulator()
        wfn_exps = {}
        for expt in expts:
            wfn_exps[expt] = wfn.expectation(gsuite.program, PauliSum([expt.observable]))

        for res in estimate_observables(qc, gsuite, num_shots=1000):
            np.testing.assert_allclose(wfn_exps[res.setting], res.expectation, atol=3*res.std_err)


def test_append():
    expts = [
        [ExperimentSetting(TensorProductState(), sX(0) * sI(1)), ExperimentSetting(TensorProductState(), sI(0) * sX(1))],
        [ExperimentSetting(TensorProductState(), sZ(0) * sI(1)), ExperimentSetting(TensorProductState(), sI(0) * sZ(1))],
    ]
    suite = ObservablesExperiment(
        settings=expts,
        program=Program(X(0), Y(1))
    )
    suite.append(ExperimentSetting(TensorProductState(), sY(0) * sX(1)))
    assert (len(str(suite))) > 0


def test_no_complex_coeffs():
    qc = get_qc('2q-qvm')
    qc.qam.random_seed = 1
    suite = ObservablesExperiment([ExperimentSetting(TensorProductState(), 1.j * sY(0))], program=Program(X(0)))
    with pytest.raises(ValueError):
        res = list(estimate_observables(qc, suite, num_shots=1000))


def test_max_weight_operator_1():
    pauli_terms = [sZ(0),
                   sX(1) * sZ(0),
                   sY(2) * sX(1)]
    assert _max_weight_operator(pauli_terms) == sY(2) * sX(1) * sZ(0)


def test_max_weight_operator_2():
    pauli_terms = [sZ(0),
                   sX(1) * sZ(0),
                   sY(2) * sX(1),
                   sZ(5) * sI(3)]
    assert _max_weight_operator(pauli_terms) == sZ(5) * sY(2) * sX(1) * sZ(0)


def test_max_weight_operator_3():
    pauli_terms = [sZ(0) * sX(5),
                   sX(1) * sZ(0),
                   sY(2) * sX(1),
                   sZ(5) * sI(3)]
    assert _max_weight_operator(pauli_terms) is None


def test_max_weight_operator_misc():
    assert _max_weight_operator([sZ(0), sZ(0) * sZ(1)]) is not None
    assert _max_weight_operator([sX(5), sZ(4)]) is not None
    assert _max_weight_operator([sX(0), sY(0) * sZ(2)]) is None

    x_term = sX(0) * sX(1)
    z1_term = sZ(1)
    z0_term = sZ(0)
    z0z1_term = sZ(0) * sZ(1)
    assert _max_weight_operator([x_term, z1_term]) is None
    assert _max_weight_operator([z0z1_term, x_term]) is None

    assert _max_weight_operator([z1_term, z0_term]) is not None
    assert _max_weight_operator([z0z1_term, z0_term]) is not None
    assert _max_weight_operator([z0z1_term, z1_term]) is not None
    assert _max_weight_operator([z0z1_term, sI(1)]) is not None
    assert _max_weight_operator([z0z1_term, sI(2)]) is not None
    assert _max_weight_operator([z0z1_term, sX(5) * sZ(7)]) is not None

    xxxx_terms = sX(1) * sX(2) + sX(2) + sX(3) * sX(4) + sX(4) + \
        sX(1) * sX(3) * sX(4) + sX(1) * sX(4) + sX(1) * sX(2) * sX(3)
    true_term = sX(1) * sX(2) * sX(3) * sX(4)
    assert _max_weight_operator(xxxx_terms.terms) == true_term

    zzzz_terms = sZ(1) * sZ(2) + sZ(3) * sZ(4) + \
        sZ(1) * sZ(3) + sZ(1) * sZ(3) * sZ(4)
    assert _max_weight_operator(zzzz_terms.terms) == sZ(1) * sZ(2) * \
        sZ(3) * sZ(4)

    pauli_terms = [sZ(0), sX(1) * sZ(0), sY(2) * sX(1), sZ(5) * sI(3)]
    assert _max_weight_operator(pauli_terms) == sZ(5) * sY(2) * sX(1) * sZ(0)


def test_max_weight_operator_4():
    # this last example illustrates that a pair of commuting operators
    # need not be diagonal in the same tpb
    assert _max_weight_operator([sX(1) * sZ(0), sZ(1) * sX(0)]) is None


def test_max_weight_state_1():
    states = [plusX(0) * plusZ(1),
              plusX(0),
              plusZ(1),
              ]
    assert _max_weight_state(states) == states[0]


def test_max_weight_state_2():
    states = [plusX(1) * plusZ(0),
              plusX(0),
              plusZ(1),
              ]
    assert _max_weight_state(states) is None


def test_max_weight_state_3():
    states = [plusX(0) * minusZ(1),
              plusX(0),
              minusZ(1),
              ]
    assert _max_weight_state(states) == states[0]


def test_max_weight_state_4():
    states = [plusX(1) * minusZ(0),
              plusX(0),
              minusZ(1),
              ]
    assert _max_weight_state(states) is None


def test_max_tpb_overlap_1():
    obs_expt_settings = [ExperimentSetting(plusZ(1) * plusX(0), sY(2) * sY(1)),
                          ExperimentSetting(plusX(2) * plusZ(1), sY(2) * sZ(0))]
    obs_expt_program = Program(H(0), H(1), H(2))
    obs_expt = ObservablesExperiment(obs_expt_settings, obs_expt_program)
    expected_dict = {
        ExperimentSetting(plusX(0) * plusZ(1) * plusX(2), sZ(0) * sY(1) * sY(2)): [
            ExperimentSetting(plusZ(1) * plusX(0), sY(2) * sY(1)),
            ExperimentSetting(plusX(2) * plusZ(1), sY(2) * sZ(0))
        ]
    }
    assert expected_dict == _max_tpb_overlap(obs_expt)


def test_max_tpb_overlap_2():
    in_state =  TensorProductState(_OneQState(label=label, index=0, qubit=q)
                    for q, label in PauliTerm.from_compact_str('(1+0j)*Z7Y8Z1Y4Z2Y5Y0X6'))
    expt_setting = ExperimentSetting(in_state,
                                     PauliTerm.from_compact_str('(1+0j)*Z4X8Y5X3Y7Y1'))
    p = Program(H(0), H(1), H(2))
    obs_expt = ObservablesExperiment([expt_setting], p)
    expected_dict = {expt_setting: [expt_setting]}
    assert expected_dict == _max_tpb_overlap(obs_expt)


def test_max_tpb_overlap_3():
    # add another ExperimentSetting to the above
    in_state =  TensorProductState(_OneQState(label=label, index=0, qubit=q)
                    for q, label in PauliTerm.from_compact_str('(1+0j)*Z7Y8Z1Y4Z2Y5Y0X6'))
    expt_setting = ExperimentSetting(in_state,
                                     PauliTerm.from_compact_str('(1+0j)*Z4X8Y5X3Y7Y1'))
    expt_setting2 = ExperimentSetting(plusZ(7), sY(1))
    p = Program(H(0), H(1), H(2))
    obs_expt2 = ObservablesExperiment([expt_setting, expt_setting2], p)
    expected_dict2 = {expt_setting: [expt_setting, expt_setting2]}
    assert expected_dict2 == _max_tpb_overlap(obs_expt2)


def test_group_experiments_greedy():
    in_state =  TensorProductState(_OneQState(label=label, index=0, qubit=q)
                    for q, label in PauliTerm.from_compact_str('(1+0j)*Z7Y8Z1Y4Z2Y5Y0X6'))
    ungrouped_obs_expt = ObservablesExperiment(
        [[ExperimentSetting(in_state, PauliTerm.from_compact_str('(1+0j)*Z4X8Y5X3Y7Y1'))],
         [ExperimentSetting(plusZ(7), sY(1))]], program=Program(H(0), H(1), H(2)))
    grouped_obs_expt = group_settings(ungrouped_obs_expt, method='greedy')
    expected_grouped_obs_expt = ObservablesExperiment(
        [[
            ExperimentSetting(TensorProductState.from_str('Z0_7 * Y0_8 * Z0_1 * Y0_4 * '
                                                          'Z0_2 * Y0_5 * Y0_0 * X0_6'),
                              PauliTerm.from_compact_str('(1+0j)*Z4X8Y5X3Y7Y1')),
            ExperimentSetting(plusZ(7), sY(1))
        ]],
        program=Program(H(0), H(1), H(2)))
    assert grouped_obs_expt == expected_grouped_obs_expt


def test_expt_settings_diagonal_in_tpb():
    def _expt_settings_diagonal_in_tpb(es1: ExperimentSetting, es2: ExperimentSetting):
        """
        Extends the concept of being diagonal in the same tpb to ExperimentSettings, by
        determining if the pairs of in_states and observables are separately diagonal in the same
        tpb
        """
        max_weight_in = _max_weight_state([es1.in_state, es2.in_state])
        max_weight_out = _max_weight_operator([es1.observable, es2.observable])
        return max_weight_in is not None and max_weight_out is not None

    expt_setting1 = ExperimentSetting(plusZ(1) * plusX(0), sY(1) * sZ(0))
    expt_setting2 = ExperimentSetting(plusY(2) * plusZ(1), sZ(2) * sY(1))
    assert _expt_settings_diagonal_in_tpb(expt_setting1, expt_setting2)
    expt_setting3 = ExperimentSetting(plusX(2) * plusZ(1), sZ(2) * sY(1))
    expt_setting4 = ExperimentSetting(plusY(2) * plusZ(1), sX(2) * sY(1))
    assert not _expt_settings_diagonal_in_tpb(expt_setting2, expt_setting3)
    assert not _expt_settings_diagonal_in_tpb(expt_setting2, expt_setting4)


def test_identity():
    qc = get_qc('2q-qvm')
    qc.qam.random_seed = 1
    suite = ObservablesExperiment([ExperimentSetting(plusZ(0), 0.123 * sI(0))],
                                 program=Program(X(0)))
    result = list(estimate_observables(qc, suite))[0]
    assert result.expectation == 0.123


def test_sic_process_tomo():
    qc = get_qc('2q-qvm')
    qc.qam.random_seed = 1
    process = Program(X(0))
    settings = []
    for in_state in [SIC0, SIC1, SIC2, SIC3]:
        for out_op in [sI, sX, sY, sZ]:
            settings += [ExperimentSetting(
                in_state=in_state(q=0),
                observable=out_op(q=0)
            )]

    experiment = ObservablesExperiment(settings=settings, program=process)
    results = list(estimate_observables(qc, experiment))
    assert len(results) == 4 * 4


def test_estimate_observables_symmetrize():
    """
    Symmetrization alone should not change the outcome on the QVM
    """
    expts = [
        ExperimentSetting(TensorProductState(), o1 * o2)
        for o1, o2 in itertools.product([sI(0), sX(0), sY(0), sZ(0)], [sI(1), sX(1), sY(1), sZ(1)])
    ]
    suite = ObservablesExperiment(expts, program=Program(X(0), CNOT(0, 1)))
    assert len(suite) == 4 * 4
    gsuite = group_settings(suite)
    assert len(gsuite) == 3 * 3  # can get all the terms with I for free in this case

    qc = get_qc('2q-qvm')
    qc.qam.random_seed = 1
    for res in estimate_observables(qc, gsuite, symm_type=-1):
        if res.setting.observable in [sI(), sZ(0), sZ(1), sZ(0) * sZ(1)]:
            assert np.abs(res.expectation) > 0.9
        else:
            assert np.abs(res.expectation) < 0.1


def test_estimate_observables_symmetrize_calibrate():
    """
    Symmetrization + calibration should not change the outcome on the QVM
    """
    expts = [
        ExperimentSetting(TensorProductState(), o1 * o2)
        for o1, o2 in itertools.product([sI(0), sX(0), sY(0), sZ(0)], [sI(1), sX(1), sY(1), sZ(1)])
    ]
    suite = ObservablesExperiment(expts, program=Program(X(0), CNOT(0, 1)))
    assert len(suite) == 4 * 4
    gsuite = group_settings(suite)
    assert len(gsuite) == 3 * 3  # can get all the terms with I for free in this case

    qc = get_qc('2q-qvm')
    qc.qam.random_seed = 1
    results = list(estimate_observables(qc, gsuite,
                                        symm_type=-1))
    for res in calibrate_observable_estimates(qc, results):
        if res.setting.observable in [sI(), sZ(0), sZ(1), sZ(0) * sZ(1)]:
            assert np.abs(res.expectation) > 0.9
        else:
            assert np.abs(res.expectation) < 0.1


def test_estimate_observables_zero_expectation():
    """
    Testing case when expectation value of observable should be close to zero
    """
    qc = get_qc('2q-qvm')
    qc.qam.random_seed = 1
    exptsetting = ExperimentSetting(plusZ(0), sX(0))
    suite = ObservablesExperiment([exptsetting],
                                 program=Program(I(0)))
    result = list(estimate_observables(qc, suite))[0]
    np.testing.assert_almost_equal(result.expectation, 0.0, decimal=1)


def test_shots_to_obs_moments():
    num_shots = 1000
    bs_results = np.array([[0, 1]]*num_shots)
    qubits = [0, 1]
    obs = sZ(0) * sX(1)

    obs_mean, obs_var = shots_to_obs_moments(bs_results, qubits, obs)
    assert obs_mean == -1.0
    assert obs_var == 0.0


def test_ratio_variance_float():
    a, b, var_a, var_b = 1.0, 2.0, 0.1, 0.05
    ab_ratio_var = ratio_variance(a, var_a, b, var_b)
    assert ab_ratio_var == 0.028125


def test_ratio_variance_numerator_zero():
    # denominator can't be zero, but numerator can be
    a, b, var_a, var_b = 0.0, 2.0, 0.1, 0.05
    ab_ratio_var = ratio_variance(a, var_a, b, var_b)
    assert ab_ratio_var == 0.025


def test_ratio_variance_array():
    a = np.array([1.0, 10.0, 100.0])
    b = np.array([2.0, 20.0, 200.0])
    var_a = np.array([0.1, 1.0, 10.0])
    var_b = np.array([0.05, 0.5, 5.0])
    ab_ratio_var = ratio_variance(a, var_a, b, var_b)
    np.testing.assert_allclose(ab_ratio_var, np.array([0.028125, 0.0028125, 0.00028125]))


def test_estimate_observables_uncalibrated_asymmetric_readout():
    qc = get_qc('1q-qvm')
    qc.qam.random_seed = 1
    expt1 = ExperimentSetting(TensorProductState(plusX(0)), sX(0))
    expt2 = ExperimentSetting(TensorProductState(plusY(0)), sY(0))
    expt3 = ExperimentSetting(TensorProductState(plusZ(0)), sZ(0))
    p = Program()
    p00, p11 = 0.90, 0.80
    p.define_noisy_readout(0, p00=p00, p11=p11)
    runs = 25
    expt_list = [expt1, expt2, expt3]
    obs_expt = ObservablesExperiment(settings=expt_list * runs, program=p)
    expected_expectation_z_basis = 2 * p00 - 1

    expect_arr = np.zeros(runs * len(expt_list))

    for idx, res in enumerate(estimate_observables(qc, obs_expt, num_shots=1000)):
        expect_arr[idx] = res.expectation

    assert np.isclose(np.mean(expect_arr[::3]), expected_expectation_z_basis, atol=3e-2)
    assert np.isclose(np.mean(expect_arr[1::3]), expected_expectation_z_basis, atol=3e-2)
    assert np.isclose(np.mean(expect_arr[2::3]), expected_expectation_z_basis, atol=3e-2)


def test_estimate_observables_uncalibrated_symmetric_readout():
    #
    qc = get_qc('1q-qvm')
    qc.qam.random_seed = 1
    expt1 = ExperimentSetting(TensorProductState(plusX(0)), sX(0)) # prep |+> measure X
    expt2 = ExperimentSetting(TensorProductState(plusY(0)), sY(0)) # prep |+i> measure Y
    expt3 = ExperimentSetting(TensorProductState(plusZ(0)), sZ(0)) # prep |0> measure Z
    p = Program()
    p00, p11 = 0.90, 0.80
    p.define_noisy_readout(0, p00=p00, p11=p11)
    runs = 25
    expt_list = [expt1, expt2, expt3]
    obs_expt = ObservablesExperiment(settings=expt_list * runs, program=p)
    expected_symm_error = (p00 + p11) / 2
    expected_expectation_z_basis = expected_symm_error * (1) + (1 - expected_symm_error) * (-1)

    uncalibr_e = np.zeros(runs * len(expt_list))

    for idx, res in enumerate(estimate_observables(qc, obs_expt, num_shots=1000,
                                                   symm_type=-1)):
        uncalibr_e[idx] = res.expectation

    assert np.isclose(np.mean(uncalibr_e[::3]), expected_expectation_z_basis, atol=3e-2)
    assert np.isclose(np.mean(uncalibr_e[1::3]), expected_expectation_z_basis, atol=3e-2)
    assert np.isclose(np.mean(uncalibr_e[2::3]), expected_expectation_z_basis, atol=3e-2)


def test_estimate_observables_calibrated_symmetric_readout():
    # expecting the result +1 for calibrated readout
    qc = get_qc('1q-qvm')
    expt1 = ExperimentSetting(TensorProductState(plusX(0)), sX(0))
    expt2 = ExperimentSetting(TensorProductState(plusY(0)), sY(0))
    expt3 = ExperimentSetting(TensorProductState(plusZ(0)), sZ(0))
    p = Program()
    p.define_noisy_readout(0, p00=0.99, p11=0.80)
    obs_expt = ObservablesExperiment(settings=[expt1, expt2, expt3], program=p)

    num_simulations = 10

    expectations = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = list(estimate_observables(qc, obs_expt, num_shots=1000,
                                                 symm_type=-1))
        expt_results = list(calibrate_observable_estimates(qc, expt_results, noisy_program=p))
        expectations.append([res.expectation for res in expt_results])
    expectations = np.array(expectations)
    results = np.mean(expectations, axis=0)
    np.testing.assert_allclose(results, 1.0, atol=3e-2)


def test_estimate_observables_result_zero_symmetrization_calibration():
    # expecting expectation value to be 0 with symmetrization/calibration
    qc = get_qc('9q-qvm')
    expt1 = ExperimentSetting(TensorProductState(plusX(0)), sZ(0))
    expt2 = ExperimentSetting(TensorProductState(minusZ(0)), sY(0))
    expt3 = ExperimentSetting(TensorProductState(minusY(0)), sX(0))
    expt_settings = [expt1, expt2, expt3]
    p = Program()
    p00, p11 = 0.99, 0.80
    p.define_noisy_readout(0, p00=p00, p11=p11)
    obs_expt = ObservablesExperiment(settings=expt_settings, program=p)

    num_simulations = 10

    expectations = []
    raw_expectations = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = list(estimate_observables(qc, obs_expt, num_shots=1000,
                                                 symm_type=-1))
        expt_results = list(calibrate_observable_estimates(qc, expt_results, noisy_program=p))
        expectations.append([res.expectation for res in expt_results])
        raw_expectations.append([res.raw_expectation for res in expt_results])
    expectations = np.array(expectations)
    raw_expectations = np.array(raw_expectations)
    results = np.mean(expectations, axis=0)
    raw_results = np.mean(raw_expectations)
    np.testing.assert_allclose(results, 0.0, atol=3e-2)
    np.testing.assert_allclose(raw_results, 0.0, atol=3e-2)


def test_estimate_observables_result_zero_no_noisy_readout():
    # expecting expectation value to be 0 with no symmetrization/calibration
    # and no noisy readout
    qc = get_qc('9q-qvm')
    expt1 = ExperimentSetting(TensorProductState(plusX(0)), sZ(0))
    expt2 = ExperimentSetting(TensorProductState(minusZ(0)), sY(0))
    expt3 = ExperimentSetting(TensorProductState(plusY(0)), sX(0))
    expt_settings = [expt1, expt2, expt3]
    p = Program()
    obs_expt = ObservablesExperiment(settings=expt_settings, program=p)

    num_simulations = 10

    expectations = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = list(estimate_observables(qc, obs_expt, num_shots=1000))
        expectations.append([res.expectation for res in expt_results])
    expectations = np.array(expectations)
    results = np.mean(expectations, axis=0)
    np.testing.assert_allclose(results, 0.0, atol=3e-2)


def test_estimate_observables_result_zero_no_symm_calibr():
    # expecting expectation value to be nonzero with symmetrization/calibration
    qc = get_qc('9q-qvm')
    qc.qam.random_seed = 1
    expt1 = ExperimentSetting(TensorProductState(plusX(0)), sZ(0))
    expt2 = ExperimentSetting(TensorProductState(minusZ(0)), sY(0))
    expt3 = ExperimentSetting(TensorProductState(minusY(0)), sX(0))
    expt_settings = [expt1, expt2, expt3]
    p = Program()
    p00, p11 = 0.99, 0.80
    p.define_noisy_readout(0, p00=p00, p11=p11)
    obs_expt = ObservablesExperiment(settings=expt_settings, program=p)

    num_simulations = 10

    expectations = []
    expected_result = (p00 * 0.5 + (1 - p11) * 0.5) - ((1 - p00) * 0.5 + p11 * 0.5)
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = list(estimate_observables(qc, obs_expt, num_shots=1000))
        expectations.append([res.expectation for res in expt_results])
    expectations = np.array(expectations)
    results = np.mean(expectations, axis=0)
    np.testing.assert_allclose(results, expected_result, atol=3e-2)


def test_estimate_observables_2q_readout_error_one_measured():
    # 2q readout errors, but only 1 qubit measured
    qc = get_qc('9q-qvm')
    qc.qam.random_seed = 1
    runs = 25
    qubs = [0, 1]
    setting = ExperimentSetting(TensorProductState(plusZ(qubs[0]) * plusZ(qubs[1])), sZ(qubs[0]))
    p = Program()
    p.define_noisy_readout(0, 0.999, 0.85)
    p.define_noisy_readout(1, 0.999, 0.75)
    expt = ObservablesExperiment(settings=[setting] * runs, program=p)

    raw_e = np.zeros(runs)
    obs_e = np.zeros(runs)
    cal_e = np.zeros(runs)

    results = calibrate_observable_estimates(qc, list(estimate_observables(qc, expt, num_shots=500,
                                                  symm_type=-1)),
                                             num_shots=500*runs, noisy_program=p)
    for idx, res in enumerate(results):
        raw_e[idx] = res.raw_expectation
        obs_e[idx] = res.expectation
        cal_e[idx] = res.calibration_expectation

    assert np.isclose(np.mean(raw_e), 0.849, atol=3e-2)
    assert np.isclose(np.mean(obs_e), 1.0, atol=3e-2)
    assert np.isclose(np.mean(cal_e), 0.849, atol=3e-2)


def test_estimate_observables_inherit_noise_errors():
    qc = get_qc('3q-qvm')
    qc.qam.random_seed = 1
    # specify simplest experiments
    expt1 = ExperimentSetting(TensorProductState(), sZ(0))
    expt2 = ExperimentSetting(TensorProductState(), sZ(1))
    expt3 = ExperimentSetting(TensorProductState(), sZ(2))
    # specify a Program with multiple sources of noise
    p = Program(X(0), Y(1), H(2))
    # defining several bit-flip channels
    kraus_ops_X = [np.sqrt(1 - 0.3) * np.array([[1, 0], [0, 1]]),
                   np.sqrt(0.3) * np.array([[0, 1], [1, 0]])]
    kraus_ops_Y = [np.sqrt(1 - 0.2) * np.array([[1, 0], [0, 1]]),
                   np.sqrt(0.2) * np.array([[0, 1], [1, 0]])]
    kraus_ops_H = [np.sqrt(1 - 0.1) * np.array([[1, 0], [0, 1]]),
                   np.sqrt(0.1) * np.array([[0, 1], [1, 0]])]
    # replacing all the gates with bit-flip channels
    p.define_noisy_gate("X", [0], kraus_ops_X)
    p.define_noisy_gate("Y", [1], kraus_ops_Y)
    p.define_noisy_gate("H", [2], kraus_ops_H)
    # defining readout errors
    p.define_noisy_readout(0, 0.99, 0.80)
    p.define_noisy_readout(1, 0.95, 0.85)
    p.define_noisy_readout(2, 0.97, 0.78)

    calibr_prog1 = get_calibration_program(expt1.observable, p)
    calibr_prog2 = get_calibration_program(expt2.observable, p)
    calibr_prog3 = get_calibration_program(expt3.observable, p)
    expected_prog = '''PRAGMA READOUT-POVM 0 "(0.99 0.19999999999999996 0.010000000000000009 0.8)"
PRAGMA READOUT-POVM 1 "(0.95 0.15000000000000002 0.050000000000000044 0.85)"
PRAGMA READOUT-POVM 2 "(0.97 0.21999999999999997 0.030000000000000027 0.78)"
PRAGMA ADD-KRAUS X 0 "(0.8366600265340756 0.0 0.0 0.8366600265340756)"
PRAGMA ADD-KRAUS X 0 "(0.0 0.5477225575051661 0.5477225575051661 0.0)"
PRAGMA ADD-KRAUS Y 1 "(0.8944271909999159 0.0 0.0 0.8944271909999159)"
PRAGMA ADD-KRAUS Y 1 "(0.0 0.4472135954999579 0.4472135954999579 0.0)"
PRAGMA ADD-KRAUS H 2 "(0.9486832980505138 0.0 0.0 0.9486832980505138)"
PRAGMA ADD-KRAUS H 2 "(0.0 0.31622776601683794 0.31622776601683794 0.0)"
'''
    assert calibr_prog1.out() == Program(expected_prog).out()
    assert calibr_prog2.out() == Program(expected_prog).out()
    assert calibr_prog3.out() == Program(expected_prog).out()


def test_expectations_sic0():
    qc = get_qc('1q-qvm')
    expt1 = ExperimentSetting(SIC0(0), sX(0))
    expt2 = ExperimentSetting(SIC0(0), sY(0))
    expt3 = ExperimentSetting(SIC0(0), sZ(0))
    obs_expt = ObservablesExperiment(settings=[expt1, expt2, expt3], program=Program())

    num_simulations = 10
    results_unavged = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        measured_results = []
        for res in estimate_observables(qc, obs_expt, num_shots=1000):
            measured_results.append(res.expectation)
        results_unavged.append(measured_results)

    results_unavged = np.array(results_unavged)
    results = np.mean(results_unavged, axis=0)
    expected_results = np.array([0, 0, 1])
    np.testing.assert_allclose(results, expected_results, atol=3e-2)


def test_expectations_sic1():
    qc = get_qc('1q-qvm')
    expt1 = ExperimentSetting(SIC1(0), sX(0))
    expt2 = ExperimentSetting(SIC1(0), sY(0))
    expt3 = ExperimentSetting(SIC1(0), sZ(0))
    obs_expt = ObservablesExperiment(settings=[expt1, expt2, expt3], program=Program())

    num_simulations = 10
    results_unavged = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        measured_results = []
        for res in estimate_observables(qc, obs_expt, num_shots=1000):
            measured_results.append(res.expectation)
        results_unavged.append(measured_results)

    results_unavged = np.array(results_unavged)
    results = np.mean(results_unavged, axis=0)
    expected_results = np.array([2 * np.sqrt(2) / 3, 0, -1 / 3])
    np.testing.assert_allclose(results, expected_results, atol=3e-2)


def test_expectations_sic2():
    qc = get_qc('1q-qvm')
    expt1 = ExperimentSetting(SIC2(0), sX(0))
    expt2 = ExperimentSetting(SIC2(0), sY(0))
    expt3 = ExperimentSetting(SIC2(0), sZ(0))
    obs_expt = ObservablesExperiment(settings=[expt1, expt2, expt3], program=Program())

    num_simulations = 10
    results_unavged = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        measured_results = []
        for res in estimate_observables(qc, obs_expt, num_shots=1000):
            measured_results.append(res.expectation)
        results_unavged.append(measured_results)

    results_unavged = np.array(results_unavged)
    results = np.mean(results_unavged, axis=0)
    expected_results = np.array([(2 * np.sqrt(2) / 3) * np.cos(2 * np.pi / 3),
                                 -(2 * np.sqrt(2) / 3) * np.sin(2 * np.pi / 3),
                                 -1 / 3])
    np.testing.assert_allclose(results, expected_results, atol=3e-2)


def test_expectations_sic3():
    qc = get_qc('1q-qvm')
    expt1 = ExperimentSetting(SIC3(0), sX(0))
    expt2 = ExperimentSetting(SIC3(0), sY(0))
    expt3 = ExperimentSetting(SIC3(0), sZ(0))
    obs_expt = ObservablesExperiment(settings=[expt1, expt2, expt3], program=Program())

    num_simulations = 10
    results_unavged = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        measured_results = []
        for res in estimate_observables(qc, obs_expt, num_shots=1000):
            measured_results.append(res.expectation)
        results_unavged.append(measured_results)

    results_unavged = np.array(results_unavged)
    results = np.mean(results_unavged, axis=0)
    expected_results = np.array([(2 * np.sqrt(2) / 3) * np.cos(2 * np.pi / 3),
                                 (2 * np.sqrt(2) / 3) * np.sin(2 * np.pi / 3),
                                 -1 / 3])
    np.testing.assert_allclose(results, expected_results, atol=3e-2)


def test_sic_conditions():
    """
    Test that the SIC states indeed yield SIC-POVMs
    """
    wfn_sim = WavefunctionSimulator()

    # condition (i) -- sum of all projectors equal identity times dimensionality
    result = np.zeros((2, 2))

    for i in range(4):
        if i == 0:
            amps = np.array([1, 0])
        else:
            sic = _one_q_sic_prep(i, 0)
            wfn = wfn_sim.wavefunction(sic)
            amps = wfn.amplitudes
        proj = np.outer(amps, amps.conj())
        result = np.add(result, proj)
    np.testing.assert_allclose(result / 2, np.eye(2), atol=3e-2)

    # condition (ii) -- tr(proj_a . proj_b) = 1 / 3, for a != b
    for comb in itertools.combinations([0, 1, 2, 3], 2):
        if comb[0] == 0:
            sic_a = Program(I(0))
        else:
            sic_a = _one_q_sic_prep(comb[0], 0)
        sic_b = _one_q_sic_prep(comb[1], 0)

        wfn_a = wfn_sim.wavefunction(sic_a)
        wfn_b = wfn_sim.wavefunction(sic_b)

        amps_a = wfn_a.amplitudes
        amps_b = wfn_b.amplitudes

        proj_a = np.outer(amps_a, amps_a.conj())
        proj_b = np.outer(amps_b, amps_b.conj())

        assert np.isclose(np.trace(proj_a.dot(proj_b)), 1 / 3)


def test_estimate_observables_grouped_expts():
    qc = get_qc('3q-qvm')
    # this more explicitly uses the list-of-lists-of-ExperimentSettings in ObservablesExperiment
    # create experiments in different groups
    expt1_group1 = ExperimentSetting(SIC1(0) * plusX(1), sZ(0) * sX(1))
    expt2_group1 = ExperimentSetting(plusX(1) * minusY(2), sX(1) * sY(2))
    expts_group1 = [expt1_group1, expt2_group1]

    expt1_group2 = ExperimentSetting(plusX(0) * SIC0(1), sX(0) * sZ(1))
    expt2_group2 = ExperimentSetting(SIC0(1) * minusY(2), sZ(1) * sY(2))
    expt3_group2 = ExperimentSetting(plusX(0) * minusY(2), sX(0) * sY(2))
    expts_group2 = [expt1_group2, expt2_group2, expt3_group2]
    # create a list-of-lists-of-ExperimentSettings
    expt_settings = [expts_group1, expts_group2]
    # and use this to create a ObservablesExperiment suite
    obs_expt = ObservablesExperiment(settings=expt_settings, program=Program())

    num_simulations = 10
    results_unavged = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        measured_results = []
        for res in estimate_observables(qc, obs_expt, num_shots=1000):
            measured_results.append(res.expectation)
        results_unavged.append(measured_results)

    results_unavged = np.array(results_unavged)
    results = np.mean(results_unavged, axis=0)
    expected_results = np.array([-1 / 3, -1, 1, -1, -1])
    np.testing.assert_allclose(results, expected_results, atol=3e-2)


def _point_channel_fidelity_estimate(v, dim=2):
    """:param v: array of expectation values
    :param dim: dimensionality of the Hilbert space"""
    return (1.0 + np.sum(v) + dim) / (dim * (dim + 1))


def test_bit_flip_channel_fidelity():
    """
    We use Eqn (5) of https://arxiv.org/abs/quant-ph/0701138 to compare the fidelity
    """
    qc = get_qc('1q-qvm')
    # prepare experiment settings
    expt1 = ExperimentSetting(TensorProductState(plusX(0)), sX(0))
    expt2 = ExperimentSetting(TensorProductState(plusY(0)), sY(0))
    expt3 = ExperimentSetting(TensorProductState(plusZ(0)), sZ(0))
    expt_list = [expt1, expt2, expt3]

    # prepare noisy bit-flip channel as program for some random value of probability
    prob = np.random.uniform(0.1, 0.5)
    # the bit flip channel is composed of two Kraus operations --
    # applying the X gate with probability `prob`, and applying the identity gate
    # with probability `1 - prob`
    kraus_ops = [np.sqrt(1 - prob) * np.array([[1, 0], [0, 1]]), np.sqrt(prob) * np.array([[0, 1], [1, 0]])]
    p = Program(Pragma("PRESERVE_BLOCK"), I(0), Pragma("END_PRESERVE_BLOCK"))
    p.define_noisy_gate("I", [0], kraus_ops)

    # prepare ObservablesExperiment
    process_exp = ObservablesExperiment(settings=expt_list, program=p)
    # list to store experiment results
    num_simulations = 10
    expts = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = []
        for res in estimate_observables(qc, process_exp, num_shots=1000):
            expt_results.append(res.expectation)
        expts.append(expt_results)

    expts = np.array(expts)
    results = np.mean(expts, axis=0)
    estimated_fidelity = _point_channel_fidelity_estimate(results)
    # how close is this channel to the identity operator
    expected_fidelity = 1 - (2 / 3) * prob
    np.testing.assert_allclose(expected_fidelity, estimated_fidelity, atol=3e-2)


def test_depolarizing_channel_fidelity():
    """
    We use Eqn (5) of https://arxiv.org/abs/quant-ph/0701138 to compare the fidelity
    """
    qc = get_qc('1q-qvm')
    # prepare experiment settings
    expt1 = ExperimentSetting(TensorProductState(plusX(0)), sX(0))
    expt2 = ExperimentSetting(TensorProductState(plusY(0)), sY(0))
    expt3 = ExperimentSetting(TensorProductState(plusZ(0)), sZ(0))
    expt_list = [expt1, expt2, expt3]

    # prepare noisy depolarizing channel as program for some random value of probability
    prob = np.random.uniform(0.1, 0.5)
    # Kraus operators for the depolarizing channel
    kraus_ops = [np.sqrt(3 * prob + 1) / 2 * np.array([[1, 0], [0, 1]]),
                 np.sqrt(1 - prob) / 2 * np.array([[0, 1], [1, 0]]),
                 np.sqrt(1 - prob) / 2 * np.array([[0, -1j], [1j, 0]]),
                 np.sqrt(1 - prob) / 2 * np.array([[1, 0], [0, -1]])]
    p = Program(Pragma("PRESERVE_BLOCK"), I(0), Pragma("END_PRESERVE_BLOCK"))
    p.define_noisy_gate("I", [0], kraus_ops)

    # prepare ObservablesExperiment
    process_exp = ObservablesExperiment(settings=expt_list, program=p)
    # list to store experiment results
    num_simulations = 10
    expts = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = []
        for res in estimate_observables(qc, process_exp, num_shots=1000):
            expt_results.append(res.expectation)
        expts.append(expt_results)

    expts = np.array(expts)
    results = np.mean(expts, axis=0)
    estimated_fidelity = _point_channel_fidelity_estimate(results)
    # how close is this channel to the identity operator
    expected_fidelity = (1 + prob) / 2
    np.testing.assert_allclose(expected_fidelity, estimated_fidelity, atol=3e-2)


def test_unitary_channel_fidelity():
    """
    We use Eqn (5) of https://arxiv.org/abs/quant-ph/0701138 to compare the fidelity
    """
    qc = get_qc('1q-qvm')
    # prepare experiment settings
    expt1 = ExperimentSetting(TensorProductState(plusX(0)), sX(0))
    expt2 = ExperimentSetting(TensorProductState(plusY(0)), sY(0))
    expt3 = ExperimentSetting(TensorProductState(plusZ(0)), sZ(0))
    expt_list = [expt1, expt2, expt3]

    # prepare unitary channel as an RY rotation program for some random angle
    theta = np.random.uniform(0.0, 2 * np.pi)
    # unitary (RY) channel
    p = Program(RY(theta, 0))
    # prepare ObservablesExperiment
    process_exp = ObservablesExperiment(settings=expt_list, program=p)
    # list to store experiment results
    num_simulations = 10
    expts = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = []
        for res in estimate_observables(qc, process_exp, num_shots=1000):
            expt_results.append(res.expectation)
        expts.append(expt_results)

    expts = np.array(expts)
    results = np.mean(expts, axis=0)
    estimated_fidelity = _point_channel_fidelity_estimate(results)
    # how close is this channel to the identity operator
    expected_fidelity = (1 / 6) * ((2 * np.cos(theta / 2)) ** 2 + 2)
    np.testing.assert_allclose(expected_fidelity, estimated_fidelity, atol=3e-2)


def test_bit_flip_channel_fidelity_readout_error():
    """
    We use Eqn (5) of https://arxiv.org/abs/quant-ph/0701138 to compare the fidelity
    """
    qc = get_qc('1q-qvm')
    # prepare experiment settings
    expt1 = ExperimentSetting(TensorProductState(plusX(0)), sX(0))
    expt2 = ExperimentSetting(TensorProductState(plusY(0)), sY(0))
    expt3 = ExperimentSetting(TensorProductState(plusZ(0)), sZ(0))
    expt_list = [expt1, expt2, expt3]

    # prepare noisy bit-flip channel as program for some random value of probability
    prob = np.random.uniform(0.1, 0.5)
    # the bit flip channel is composed of two Kraus operations --
    # applying the X gate with probability `prob`, and applying the identity gate
    # with probability `1 - prob`
    kraus_ops = [np.sqrt(1 - prob) * np.array([[1, 0], [0, 1]]), np.sqrt(prob) * np.array([[0, 1], [1, 0]])]
    p = Program(Pragma("PRESERVE_BLOCK"), I(0), Pragma("END_PRESERVE_BLOCK"))
    p.define_noisy_gate("I", [0], kraus_ops)
    # add some readout error
    p.define_noisy_readout(0, 0.95, 0.82)

    # prepare ObservablesExperiment
    process_exp = ObservablesExperiment(settings=expt_list, program=p)
    # list to store experiment results
    num_simulations = 10
    expts = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = []
        results = estimate_observables(qc, process_exp,
                                       symm_type=-1)
        for res in calibrate_observable_estimates(qc, list(results), noisy_program=p):
            expt_results.append(res.expectation)
        expts.append(expt_results)

    expts = np.array(expts)
    results = np.mean(expts, axis=0)
    estimated_fidelity = _point_channel_fidelity_estimate(results)
    # how close is this channel to the identity operator
    expected_fidelity = 1 - (2 / 3) * prob
    np.testing.assert_allclose(expected_fidelity, estimated_fidelity, atol=3e-2)


def test_depolarizing_channel_fidelity_readout_error():
    """
    We use Eqn (5) of https://arxiv.org/abs/quant-ph/0701138 to compare the fidelity
    """
    qc = get_qc('1q-qvm')
    # prepare experiment settings
    expt1 = ExperimentSetting(TensorProductState(plusX(0)), sX(0))
    expt2 = ExperimentSetting(TensorProductState(plusY(0)), sY(0))
    expt3 = ExperimentSetting(TensorProductState(plusZ(0)), sZ(0))
    expt_list = [expt1, expt2, expt3]

    # prepare noisy depolarizing channel as program for some random value of probability
    prob = np.random.uniform(0.1, 0.5)
    # Kraus operators for the depolarizing channel
    kraus_ops = [np.sqrt(3 * prob + 1) / 2 * np.array([[1, 0], [0, 1]]),
                 np.sqrt(1 - prob) / 2 * np.array([[0, 1], [1, 0]]),
                 np.sqrt(1 - prob) / 2 * np.array([[0, -1j], [1j, 0]]),
                 np.sqrt(1 - prob) / 2 * np.array([[1, 0], [0, -1]])]
    p = Program(Pragma("PRESERVE_BLOCK"), I(0), Pragma("END_PRESERVE_BLOCK"))
    p.define_noisy_gate("I", [0], kraus_ops)
    # add some readout error
    p.define_noisy_readout(0, 0.95, 0.82)

    # prepare ObservablesExperiment
    process_exp = ObservablesExperiment(settings=expt_list, program=p)
    # list to store experiment results
    num_simulations = 10
    expts = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = []
        results = estimate_observables(qc, process_exp,
                                       symm_type=-1)
        for res in calibrate_observable_estimates(qc, list(results), noisy_program=p):
            expt_results.append(res.expectation)
        expts.append(expt_results)

    expts = np.array(expts)
    results = np.mean(expts, axis=0)
    estimated_fidelity = _point_channel_fidelity_estimate(results)
    # how close is this channel to the identity operator
    expected_fidelity = (1 + prob) / 2
    np.testing.assert_allclose(expected_fidelity, estimated_fidelity, atol=3e-2)


def test_unitary_channel_fidelity_readout_error():
    """
    We use Eqn (5) of https://arxiv.org/abs/quant-ph/0701138 to compare the fidelity
    """
    qc = get_qc('1q-qvm')
    # prepare experiment settings
    expt1 = ExperimentSetting(TensorProductState(plusX(0)), sX(0))
    expt2 = ExperimentSetting(TensorProductState(plusY(0)), sY(0))
    expt3 = ExperimentSetting(TensorProductState(plusZ(0)), sZ(0))
    expt_list = [expt1, expt2, expt3]

    # prepare unitary channel as an RY rotation program for some random angle
    theta = np.random.uniform(0.0, 2 * np.pi)
    # unitary (RY) channel
    p = Program(RY(theta, 0))
    # add some readout error
    p.define_noisy_readout(0, 0.95, 0.82)
    # prepare ObservablesExperiment
    process_exp = ObservablesExperiment(settings=expt_list, program=p)
    # list to store experiment results
    num_simulations = 10
    expts = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = []
        results = estimate_observables(qc, process_exp,
                                       symm_type=-1)
        for res in calibrate_observable_estimates(qc, list(results), noisy_program=p):
            expt_results.append(res.expectation)
        expts.append(expt_results)

    expts = np.array(expts)
    results = np.mean(expts, axis=0)
    estimated_fidelity = _point_channel_fidelity_estimate(results)
    # how close is this channel to the identity operator
    expected_fidelity = (1 / 6) * ((2 * np.cos(theta / 2)) ** 2 + 2)
    np.testing.assert_allclose(expected_fidelity, estimated_fidelity, atol=3e-2)


@pytest.mark.slow
def test_2q_unitary_channel_fidelity_readout_error():
    """
    We use Eqn (5) of https://arxiv.org/abs/quant-ph/0701138 to compare the fidelity
    This tests if our dimensionality factors are correct, even in the presence
    of readout errors
    """
    qc = get_qc('2q-qvm')
    # prepare experiment settings

    expt1 = ExperimentSetting(TensorProductState(plusX(0)), sX(0))
    expt2 = ExperimentSetting(TensorProductState(plusY(0)), sY(0))
    expt3 = ExperimentSetting(TensorProductState(plusZ(0)), sZ(0))

    expt4 = ExperimentSetting(TensorProductState(plusX(1)), sX(1))
    expt5 = ExperimentSetting(TensorProductState(plusX(0) * plusX(1)), sX(0) * sX(1))
    expt6 = ExperimentSetting(TensorProductState(plusY(0) * plusX(1)), sY(0) * sX(1))
    expt7 = ExperimentSetting(TensorProductState(plusZ(0) * plusX(1)), sZ(0) * sX(1))

    expt8 = ExperimentSetting(TensorProductState(plusY(1)), sY(1))
    expt9 = ExperimentSetting(TensorProductState(plusX(0) * plusY(1)), sX(0) * sY(1))
    expt10 = ExperimentSetting(TensorProductState(plusY(0) * plusY(1)), sY(0) * sY(1))
    expt11 = ExperimentSetting(TensorProductState(plusZ(0) * plusY(1)), sZ(0) * sY(1))

    expt12 = ExperimentSetting(TensorProductState(plusZ(1)), sZ(1))
    expt13 = ExperimentSetting(TensorProductState(plusX(0) * plusZ(1)), sX(0) * sZ(1))
    expt14 = ExperimentSetting(TensorProductState(plusY(0) * plusZ(1)), sY(0) * sZ(1))
    expt15 = ExperimentSetting(TensorProductState(plusZ(0) * plusZ(1)), sZ(0) * sZ(1))

    expt_list = [expt1, expt2, expt3, expt4, expt5, expt6, expt7, expt8, expt9, expt10, expt11, expt12, expt13, expt14, expt15]

    # prepare unitary channel as an RY rotation program for some random angle
    # theta1, theta2 = np.random.uniform(0.0, 2 * np.pi, size=2)
    theta1, theta2 = np.pi / 4, np.pi / 5
    # unitary (RY) channel
    p = Program(RY(theta1, 0), RY(theta2, 1))
    # add some readout error
    p.define_noisy_readout(0, 0.95, 0.82)
    p.define_noisy_readout(1, 0.99, 0.73)
    # prepare ObservablesExperiment
    process_exp = ObservablesExperiment(settings=expt_list, program=p)
    # list to store experiment results
    num_simulations = 10
    expts = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = []
        results = estimate_observables(qc, process_exp,
                                       symm_type=-1)
        for res in calibrate_observable_estimates(qc, list(results), noisy_program=p):
            expt_results.append(res.expectation)
        expts.append(expt_results)

    expts = np.array(expts)
    results = np.mean(expts, axis=0)
    estimated_fidelity = _point_channel_fidelity_estimate(results, dim=4)
    # how close is this channel to the identity operator
    expected_fidelity = (1 / 5) * ((2 * np.cos(theta1 / 2) * np.cos(theta2 / 2)) ** 2 + 1)
    np.testing.assert_allclose(expected_fidelity, estimated_fidelity, atol=3e-2)


def test_measure_1q_observable_raw_expectation():
    # testing that we get correct raw expectation in terms of readout errors
    qc = get_qc('1q-qvm')
    expt = ExperimentSetting(TensorProductState(plusZ(0)), sZ(0))
    p = Program()
    p00, p11 = 0.99, 0.80
    p.define_noisy_readout(0, p00=p00, p11=p11)
    obs_expt = ObservablesExperiment(settings=[expt], program=p)

    num_simulations = 10

    raw_expectations = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = list(estimate_observables(qc, obs_expt, num_shots=1000,
                                                 symm_type=-1))
        expt_results = list(calibrate_observable_estimates(qc, expt_results, noisy_program=p))
        raw_expectations.append([res.raw_expectation for res in expt_results])
    raw_expectations = np.array(raw_expectations)
    result = np.mean(raw_expectations, axis=0)

    # calculate expected raw_expectation
    eps_not = (p00 + p11) / 2
    eps = 1 - eps_not
    expected_result = 1 - 2 * eps
    np.testing.assert_allclose(result, expected_result, atol=3e-2)


def test_measure_1q_observable_raw_variance():
    # testing that we get correct raw std_err in terms of readout errors
    qc = get_qc('1q-qvm')
    expt = ExperimentSetting(TensorProductState(plusZ(0)), sZ(0))
    p = Program()
    p00, p11 = 0.99, 0.80
    p.define_noisy_readout(0, p00=p00, p11=p11)
    obs_expt = ObservablesExperiment(settings=[expt], program=p)

    num_simulations = 10
    num_shots = 1000

    raw_std_errs = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = list(estimate_observables(qc, obs_expt, num_shots=num_shots,
                                                 symm_type=-1))
        expt_results = list(calibrate_observable_estimates(qc, expt_results, noisy_program=p))
        raw_std_errs.append([res.raw_std_err for res in expt_results])
    raw_std_errs = np.array(raw_std_errs)
    result = np.mean(raw_std_errs, axis=0)

    # calculate expected raw_expectation
    eps_not = (p00 + p11) / 2
    eps = 1 - eps_not
    expected_result = np.sqrt((1 - (1 - 2 * eps) ** 2) / num_shots)
    np.testing.assert_allclose(result, expected_result, atol=3e-2)


def test_measure_1q_observable_calibration_expectation():
    # testing that we get correct calibration expectation in terms of readout errors
    qc = get_qc('1q-qvm')
    expt = ExperimentSetting(TensorProductState(plusZ(0)), sZ(0))
    p = Program()
    p00, p11 = 0.93, 0.77
    p.define_noisy_readout(0, p00=p00, p11=p11)
    obs_expt = ObservablesExperiment(settings=[expt], program=p)

    num_simulations = 10

    calibration_expectations = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = list(estimate_observables(qc, obs_expt, num_shots=1000,
                                                 symm_type=-1))
        expt_results = list(calibrate_observable_estimates(qc, expt_results, noisy_program=p))
        calibration_expectations.append([res.calibration_expectation for res in expt_results])
    calibration_expectations = np.array(calibration_expectations)
    result = np.mean(calibration_expectations, axis=0)

    # calculate expected raw_expectation
    eps_not = (p00 + p11) / 2
    eps = 1 - eps_not
    expected_result = 1 - 2 * eps
    np.testing.assert_allclose(result, expected_result, atol=3e-2)


def test_measure_1q_observable_calibration_variance():
    # testing that we get correct calibration std_err in terms of readout errors
    qc = get_qc('1q-qvm')
    expt = ExperimentSetting(TensorProductState(plusZ(0)), sZ(0))
    p = Program()
    p00, p11 = 0.93, 0.77
    p.define_noisy_readout(0, p00=p00, p11=p11)
    obs_expt = ObservablesExperiment(settings=[expt], program=p)

    num_simulations = 10
    num_shots = 1000

    raw_std_errs = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = list(estimate_observables(qc, obs_expt, num_shots=num_shots,
                                                 symm_type=-1))
        expt_results = list(calibrate_observable_estimates(qc, expt_results, noisy_program=p))
        raw_std_errs.append([res.raw_std_err for res in expt_results])
    raw_std_errs = np.array(raw_std_errs)
    result = np.mean(raw_std_errs, axis=0)

    # calculate expected raw_expectation
    eps_not = (p00 + p11) / 2
    eps = 1 - eps_not
    expected_result = np.sqrt((1 - (1 - 2 * eps) ** 2) / num_shots)
    np.testing.assert_allclose(result, expected_result, atol=3e-2)


def test_uncalibrated_asymmetric_readout_nontrivial_1q_state():
    qc = get_qc('1q-qvm')
    expt = ExperimentSetting(TensorProductState(), sZ(0))
    # pick some random value for RX rotation
    theta = np.random.uniform(0.0, 2 * np.pi)
    p = Program(RX(theta, 0))
    # pick some random (but sufficiently large) asymmetric readout errors
    p00, p11 = np.random.uniform(0.7, 0.99, size=2)
    p.define_noisy_readout(0, p00=p00, p11=p11)
    runs = 25
    expt_list = [expt]
    obs_expt = ObservablesExperiment(settings=expt_list * runs, program=p)
    # calculate expected expectation value
    amp_sqr0 = (np.cos(theta / 2)) ** 2
    amp_sqr1 = (np.sin(theta / 2)) ** 2
    expected_expectation = (p00 * amp_sqr0 + (1 - p11) * amp_sqr1) - \
                           ((1 - p00) * amp_sqr0 + p11 * amp_sqr1)

    expect_arr = np.zeros(runs * len(expt_list))

    for idx, res in enumerate(estimate_observables(qc,
                                                  obs_expt, num_shots=1000)):
        expect_arr[idx] = res.expectation

    assert np.isclose(np.mean(expect_arr), expected_expectation, atol=3e-2)


def test_uncalibrated_symmetric_readout_nontrivial_1q_state():
    qc = get_qc('1q-qvm')
    qc.qam.random_seed = 1
    expt = ExperimentSetting(TensorProductState(), sZ(0))
    # pick some random value for RX rotation
    theta = np.random.uniform(0.0, 2 * np.pi)
    p = Program(RX(theta, 0))
    # pick some random (but sufficiently large) asymmetric readout errors
    p00, p11 = np.random.uniform(0.7, 0.99, size=2)
    p.define_noisy_readout(0, p00=p00, p11=p11)
    runs = 25
    expt_list = [expt]
    obs_expt = ObservablesExperiment(settings=expt_list * runs, program=p)
    # calculate expected expectation value
    amp_sqr0 = (np.cos(theta / 2)) ** 2
    amp_sqr1 = (np.sin(theta / 2)) ** 2
    symm_prob = (p00 + p11) / 2
    expected_expectation = (symm_prob * amp_sqr0 + (1 - symm_prob) * amp_sqr1) - \
                           ((1 - symm_prob) * amp_sqr0 + symm_prob * amp_sqr1)

    expect_arr = np.zeros(runs * len(expt_list))

    for idx, res in enumerate(estimate_observables(qc,
                                                  obs_expt, num_shots=1000,
                                                  symm_type=-1)):
        expect_arr[idx] = res.expectation

    assert np.isclose(np.mean(expect_arr), expected_expectation, atol=3e-2)


def test_calibrated_symmetric_readout_nontrivial_1q_state():
    qc = get_qc('1q-qvm')
    qc.qam.random_seed = 1
    expt = ExperimentSetting(TensorProductState(), sZ(0))
    # pick some random value for RX rotation
    theta = np.random.uniform(0.0, 2 * np.pi)
    p = Program(RX(theta, 0))
    # pick some random (but sufficiently large) asymmetric readout errors
    p00, p11 = np.random.uniform(0.7, 0.99, size=2)
    p.define_noisy_readout(0, p00=p00, p11=p11)
    runs = 25
    expt_list = [expt]
    obs_expt = ObservablesExperiment(settings=expt_list * runs, program=p)
    # calculate expected expectation value
    amp_sqr0 = (np.cos(theta / 2)) ** 2
    amp_sqr1 = (np.sin(theta / 2)) ** 2
    expected_expectation = amp_sqr0 - amp_sqr1

    expect_arr = np.zeros(runs * len(expt_list))
    z_cal_expect_arr = np.zeros(runs * len(expt_list))

    results = calibrate_observable_estimates(qc, list(estimate_observables(qc,
                                                  obs_expt, num_shots=750,
                                                  symm_type=-1)),
                                             num_shots=750, noisy_program=p)
    for idx, res in enumerate(results):
        expect_arr[idx] = res.expectation
        z_cal_expect_arr[idx] = res.calibration_expectation

    assert np.isclose(np.mean(z_cal_expect_arr), p00 + p11 - 1, atol=3e-2)
    assert np.isclose(np.mean(expect_arr), expected_expectation, atol=3e-2)


def test_measure_2q_observable_raw_statistics():
    # testing that we get correct exhaustively symmetrized statistics
    # in terms of readout errors
    # Note: this only tests for exhaustive symmetrization in the presence
    #       of uncorrelated errors
    qc = get_qc('2q-qvm')
    expt = ExperimentSetting(TensorProductState(), sZ(0) * sZ(1))
    p = Program()
    p00, p11 = 0.99, 0.80
    q00, q11 = 0.93, 0.76
    p.define_noisy_readout(0, p00=p00, p11=p11)
    p.define_noisy_readout(1, p00=q00, p11=q11)
    obs_expt = ObservablesExperiment(settings=[expt], program=p)

    num_simulations = 10
    num_shots = 5000

    raw_expectations = []
    raw_std_errs = []

    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = list(estimate_observables(qc, obs_expt, num_shots=num_shots,
                                                 symm_type=-1))
        expt_results = list(calibrate_observable_estimates(qc, expt_results, noisy_program=p))
        raw_expectations.append([res.raw_expectation for res in expt_results])
        raw_std_errs.append([res.raw_std_err for res in expt_results])

    raw_expectations = np.array(raw_expectations)
    raw_std_errs = np.array(raw_std_errs)
    result_expectation = np.mean(raw_expectations, axis=0)
    result_std_err = np.mean(raw_std_errs, axis=0)

    # calculate relevant conditional probabilities, given |00> state
    # notation used: pijmn means p(ij|mn)
    p0000 = (p00 + p11) * (q00 + q11) / 4
    p0100 = (p00 + p11) * (2 - q00 - q11) / 4
    p1000 = (q00 + q11) * (2 - p00 - p11) / 4
    p1100 = (2 - p00 - p11) * (2 - q00 - q11) / 4
    # calculate expectation value of Z^{\otimes 2}
    z_expectation = (p0000 + p1100) - (p0100 + p1000)
    # calculate standard deviation of the mean
    simulated_std_err = np.sqrt((1 - z_expectation ** 2) / num_shots)
    # compare against simulated results
    np.testing.assert_allclose(result_expectation, z_expectation, atol=3e-2)
    np.testing.assert_allclose(result_std_err, simulated_std_err, atol=3e-2)


def test_raw_statistics_2q_nontrivial_nonentangled_state():
    # testing that we get correct exhaustively symmetrized statistics
    # in terms of readout errors, even for non-trivial 2q nonentangled states
    # Note: this only tests for exhaustive symmetrization in the presence
    #       of uncorrelated errors
    qc = get_qc('2q-qvm')
    expt = ExperimentSetting(TensorProductState(), sZ(0) * sZ(1))
    theta1, theta2 = np.random.uniform(0.0, 2 * np.pi, size=2)
    p = Program(RX(theta1, 0), RX(theta2, 1))
    p00, p11, q00, q11 = np.random.uniform(0.70, 0.99, size=4)
    p.define_noisy_readout(0, p00=p00, p11=p11)
    p.define_noisy_readout(1, p00=q00, p11=q11)
    obs_expt = ObservablesExperiment(settings=[expt], program=p)

    num_simulations = 10
    num_shots = 5000

    raw_expectations = []
    raw_std_errs = []

    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = list(estimate_observables(qc, obs_expt, num_shots=num_shots,
                                                 symm_type=-1))
        expt_results = list(calibrate_observable_estimates(qc, expt_results, noisy_program=p))
        raw_expectations.append([res.raw_expectation for res in expt_results])
        raw_std_errs.append([res.raw_std_err for res in expt_results])
    raw_expectations = np.array(raw_expectations)
    raw_std_errs = np.array(raw_std_errs)
    result_expectation = np.mean(raw_expectations, axis=0)
    result_std_err = np.mean(raw_std_errs, axis=0)

    # calculate relevant conditional probabilities, given |00> state
    # notation used: pijmn means p(ij|mn)
    p0000 = (p00 + p11) * (q00 + q11) / 4
    p0100 = (p00 + p11) * (2 - q00 - q11) / 4
    p1000 = (q00 + q11) * (2 - p00 - p11) / 4
    p1100 = (2 - p00 - p11) * (2 - q00 - q11) / 4
    # calculate relevant conditional probabilities, given |01> state
    p0001 = p0100
    p0101 = p0000
    p1001 = (2 - p00 - p11) * (2 - q00 - q11) / 4
    p1101 = (2 - p00 - p11) * (q00 + q11) / 4
    # calculate relevant conditional probabilities, given |10> state
    p0010 = p1000
    p0110 = p1001
    p1010 = p0000
    p1110 = (p00 + p11) * (2 - q00 - q11) / 4
    # calculate relevant conditional probabilities, given |11> state
    p0011 = p1100
    p0111 = p1101
    p1011 = p1110
    p1111 = p0000
    # calculate amplitudes squared of pure state
    alph00 = (np.cos(theta1 / 2) * np.cos(theta2 / 2)) ** 2
    alph01 = (np.cos(theta1 / 2) * np.sin(theta2 / 2)) ** 2
    alph10 = (np.sin(theta1 / 2) * np.cos(theta2 / 2)) ** 2
    alph11 = (np.sin(theta1 / 2) * np.sin(theta2 / 2)) ** 2
    # calculate probabilities of various bitstrings
    pr00 = p0000 * alph00 + p0001 * alph01 + p0010 * alph10 + p0011 * alph11
    pr01 = p0100 * alph00 + p0101 * alph01 + p0110 * alph10 + p0111 * alph11
    pr10 = p1000 * alph00 + p1001 * alph01 + p1010 * alph10 + p1011 * alph11
    pr11 = p1100 * alph00 + p1101 * alph01 + p1110 * alph10 + p1111 * alph11
    # calculate Z^{\otimes 2} expectation, and error of the mean
    z_expectation = (pr00 + pr11) - (pr01 + pr10)
    simulated_std_err = np.sqrt((1 - z_expectation ** 2) / num_shots)
    # compare against simulated results
    np.testing.assert_allclose(result_expectation, z_expectation, atol=3e-2)
    np.testing.assert_allclose(result_std_err, simulated_std_err, atol=3e-2)


def test_raw_statistics_2q_nontrivial_entangled_state():
    # testing that we get correct exhaustively symmetrized statistics
    # in terms of readout errors, even for non-trivial 2q entangled states
    # Note: this only tests for exhaustive symmetrization in the presence
    #       of uncorrelated errors
    qc = get_qc('2q-qvm')
    expt = ExperimentSetting(TensorProductState(), sZ(0) * sZ(1))
    theta = np.random.uniform(0.0, 2 * np.pi)
    p = Program(RX(theta, 0), CNOT(0, 1))
    p00, p11, q00, q11 = np.random.uniform(0.70, 0.99, size=4)
    p.define_noisy_readout(0, p00=p00, p11=p11)
    p.define_noisy_readout(1, p00=q00, p11=q11)
    obs_expt = ObservablesExperiment(settings=[expt], program=p)

    num_simulations = 10
    num_shots = 5000

    raw_expectations = []
    raw_std_errs = []

    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = list(estimate_observables(qc, obs_expt, num_shots=num_shots,
                                                 symm_type=-1))
        expt_results = list(calibrate_observable_estimates(qc, expt_results, noisy_program=p))
        raw_expectations.append([res.raw_expectation for res in expt_results])
        raw_std_errs.append([res.raw_std_err for res in expt_results])
    raw_expectations = np.array(raw_expectations)
    raw_std_errs = np.array(raw_std_errs)
    result_expectation = np.mean(raw_expectations, axis=0)
    result_std_err = np.mean(raw_std_errs, axis=0)

    # calculate relevant conditional probabilities, given |00> state
    # notation used: pijmn means p(ij|mn)
    p0000 = (p00 + p11) * (q00 + q11) / 4
    p0100 = (p00 + p11) * (2 - q00 - q11) / 4
    p1000 = (q00 + q11) * (2 - p00 - p11) / 4
    p1100 = (2 - p00 - p11) * (2 - q00 - q11) / 4
    # calculate relevant conditional probabilities, given |11> state
    p0011 = p1100
    p0111 = (2 - p00 - p11) * (q00 + q11) / 4
    p1011 = (p00 + p11) * (2 - q00 - q11) / 4
    p1111 = p0000
    # calculate amplitudes squared of pure state
    alph00 = (np.cos(theta / 2)) ** 2
    alph11 = (np.sin(theta / 2)) ** 2
    # calculate probabilities of various bitstrings
    pr00 = p0000 * alph00 + p0011 * alph11
    pr01 = p0100 * alph00 + p0111 * alph11
    pr10 = p1000 * alph00 + p1011 * alph11
    pr11 = p1100 * alph00 + p1111 * alph11
    # calculate Z^{\otimes 2} expectation, and error of the mean
    z_expectation = (pr00 + pr11) - (pr01 + pr10)
    simulated_std_err = np.sqrt((1 - z_expectation ** 2) / num_shots)
    # compare against simulated results
    np.testing.assert_allclose(result_expectation, z_expectation, atol=3e-2)
    np.testing.assert_allclose(result_std_err, simulated_std_err, atol=3e-2)


def _point_state_fidelity_estimate(v, dim=2):
    """:param v: array of expectation values
    :param dim: dimensionality of the Hilbert space"""
    return (1.0 + np.sum(v)) / dim


def test_bit_flip_state_fidelity():
    qc = get_qc('1q-qvm')
    qc.qam.random_seed = 1
    # prepare experiment setting
    expt = ExperimentSetting(TensorProductState(), sZ(0))

    # prepare noisy bit-flip channel as program for some random value of probability
    prob = np.random.uniform(0.1, 0.5)
    # the bit flip channel is composed of two Kraus operations --
    # applying the X gate with probability `prob`, and applying the identity gate
    # with probability `1 - prob`
    kraus_ops = [np.sqrt(1 - prob) * np.array([[1, 0], [0, 1]]), np.sqrt(prob) * np.array([[0, 1], [1, 0]])]
    p = Program(Pragma("PRESERVE_BLOCK"), I(0), Pragma("END_PRESERVE_BLOCK"))
    p.define_noisy_gate("I", [0], kraus_ops)

    # prepare ObservablesExperiment
    process_exp = ObservablesExperiment(settings=[expt], program=p)
    # list to store experiment results
    num_simulations = 10
    expts = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = []
        for res in estimate_observables(qc, process_exp, num_shots=1000):
            expt_results.append(res.expectation)
        expts.append(expt_results)

    expts = np.array(expts)
    results = np.mean(expts, axis=0)
    estimated_fidelity = _point_state_fidelity_estimate(results)
    # how close is the mixed state to |0>
    expected_fidelity = 1 - prob
    np.testing.assert_allclose(expected_fidelity, estimated_fidelity, atol=3e-2)


def test_dephasing_state_fidelity():
    qc = get_qc('1q-qvm')
    # prepare experiment setting
    expt = ExperimentSetting(TensorProductState(), sZ(0))

    # prepare noisy dephasing channel as program for some random value of probability
    prob = np.random.uniform(0.1, 0.5)
    # Kraus operators for dephasing channel
    kraus_ops = [np.sqrt(1 - prob) * np.array([[1, 0], [0, 1]]),
                 np.sqrt(prob) * np.array([[1, 0], [0, -1]])]
    p = Program(Pragma("PRESERVE_BLOCK"), I(0), Pragma("END_PRESERVE_BLOCK"))
    p.define_noisy_gate("I", [0], kraus_ops)

    # prepare ObservablesExperiment
    process_exp = ObservablesExperiment(settings=[expt], program=p)
    # list to store experiment results
    num_simulations = 10
    expts = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = []
        for res in estimate_observables(qc, process_exp, num_shots=1000):
            expt_results.append(res.expectation)
        expts.append(expt_results)

    expts = np.array(expts)
    results = np.mean(expts, axis=0)
    estimated_fidelity = _point_state_fidelity_estimate(results)
    # how close is the mixed state to |0>
    expected_fidelity = 1
    np.testing.assert_allclose(expected_fidelity, estimated_fidelity, atol=3e-2)


def test_depolarizing_state_fidelity():
    qc = get_qc('1q-qvm')
    # prepare experiment setting
    expt = ExperimentSetting(TensorProductState(), sZ(0))

    # prepare noisy depolarizing channel as program for some random value of probability
    prob = np.random.uniform(0.1, 0.5)
    # Kraus operators for depolarizing channel
    kraus_ops = [np.sqrt(3 * prob + 1) / 2 * np.array([[1, 0], [0, 1]]),
                 np.sqrt(1 - prob) / 2 * np.array([[0, 1], [1, 0]]),
                 np.sqrt(1 - prob) / 2 * np.array([[0, -1j], [1j, 0]]),
                 np.sqrt(1 - prob) / 2 * np.array([[1, 0], [0, -1]])]
    p = Program(Pragma("PRESERVE_BLOCK"), I(0), Pragma("END_PRESERVE_BLOCK"))
    p.define_noisy_gate("I", [0], kraus_ops)

    # prepare ObservablesExperiment
    process_exp = ObservablesExperiment(settings=[expt], program=p)
    # list to store experiment results
    num_simulations = 10
    expts = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = []
        for res in estimate_observables(qc, process_exp, num_shots=1000):
            expt_results.append(res.expectation)
        expts.append(expt_results)

    expts = np.array(expts)
    results = np.mean(expts, axis=0)
    estimated_fidelity = _point_state_fidelity_estimate(results)
    # how close is the mixed state to |0>
    expected_fidelity = (1 + prob) / 2
    np.testing.assert_allclose(expected_fidelity, estimated_fidelity, atol=3e-2)


def test_unitary_state_fidelity():
    qc = get_qc('1q-qvm')
    # prepare experiment setting
    expt = ExperimentSetting(TensorProductState(), sZ(0))

    # rotate |0> state by some random angle about X axis
    theta = np.random.uniform(0.0, 2 * np.pi)
    p = Program(RX(theta, 0))

    # prepare ObservablesExperiment
    process_exp = ObservablesExperiment(settings=[expt], program=p)
    # list to store experiment results
    num_simulations = 10
    expts = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = []
        for res in estimate_observables(qc, process_exp, num_shots=1000):
            expt_results.append(res.expectation)
        expts.append(expt_results)

    expts = np.array(expts)
    results = np.mean(expts, axis=0)
    estimated_fidelity = _point_state_fidelity_estimate(results)
    # how close is this state to |0>
    expected_fidelity = (np.cos(theta / 2)) ** 2
    np.testing.assert_allclose(expected_fidelity, estimated_fidelity, atol=3e-2)


def test_bit_flip_state_fidelity_readout_error():
    qc = get_qc('1q-qvm')
    # prepare experiment setting
    expt = ExperimentSetting(TensorProductState(), sZ(0))

    # prepare noisy bit-flip channel as program for some random value of probability
    prob = np.random.uniform(0.1, 0.5)
    # the bit flip channel is composed of two Kraus operations --
    # applying the X gate with probability `prob`, and applying the identity gate
    # with probability `1 - prob`
    kraus_ops = [np.sqrt(1 - prob) * np.array([[1, 0], [0, 1]]), np.sqrt(prob) * np.array([[0, 1], [1, 0]])]
    p = Program(Pragma("PRESERVE_BLOCK"), I(0), Pragma("END_PRESERVE_BLOCK"))
    p.define_noisy_gate("I", [0], kraus_ops)
    p.define_noisy_readout(0, 0.95, 0.76)

    # prepare ObservablesExperiment
    process_exp = ObservablesExperiment(settings=[expt], program=p)
    # list to store experiment results
    num_simulations = 10
    expts = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = []
        results = estimate_observables(qc, process_exp,
                                       symm_type=-1)
        for res in calibrate_observable_estimates(qc, list(results), noisy_program=p):
            expt_results.append(res.expectation)
        expts.append(expt_results)

    expts = np.array(expts)
    results = np.mean(expts, axis=0)
    estimated_fidelity = _point_state_fidelity_estimate(results)
    # how close is the mixed state to |0>
    expected_fidelity = 1 - prob
    np.testing.assert_allclose(expected_fidelity, estimated_fidelity, atol=3e-2)


def test_dephasing_state_fidelity_readout_error():
    qc = get_qc('1q-qvm')
    # prepare experiment setting
    expt = ExperimentSetting(TensorProductState(), sZ(0))

    # prepare noisy dephasing channel as program for some random value of probability
    prob = np.random.uniform(0.1, 0.5)
    # Kraus operators for dephasing channel
    kraus_ops = [np.sqrt(1 - prob) * np.array([[1, 0], [0, 1]]),
                 np.sqrt(prob) * np.array([[1, 0], [0, -1]])]
    p = Program(Pragma("PRESERVE_BLOCK"), I(0), Pragma("END_PRESERVE_BLOCK"))
    p.define_noisy_gate("I", [0], kraus_ops)
    p.define_noisy_readout(0, 0.95, 0.76)

    # prepare ObservablesExperiment
    process_exp = ObservablesExperiment(settings=[expt], program=p)
    # list to store experiment results
    num_simulations = 10
    expts = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = []
        results = estimate_observables(qc, process_exp,
                                       symm_type=-1)
        for res in calibrate_observable_estimates(qc, list(results), noisy_program=p):
            expt_results.append(res.expectation)
        expts.append(expt_results)

    expts = np.array(expts)
    results = np.mean(expts, axis=0)
    estimated_fidelity = _point_state_fidelity_estimate(results)
    # how close is the mixed state to |0>
    expected_fidelity = 1
    np.testing.assert_allclose(expected_fidelity, estimated_fidelity, atol=3e-2)


def test_depolarizing_state_fidelity_readout_error():
    qc = get_qc('1q-qvm')
    # prepare experiment setting
    expt = ExperimentSetting(TensorProductState(), sZ(0))

    # prepare noisy depolarizing channel as program for some random value of probability
    prob = np.random.uniform(0.1, 0.5)
    # Kraus operators for depolarizing channel
    kraus_ops = [np.sqrt(3 * prob + 1) / 2 * np.array([[1, 0], [0, 1]]),
                 np.sqrt(1 - prob) / 2 * np.array([[0, 1], [1, 0]]),
                 np.sqrt(1 - prob) / 2 * np.array([[0, -1j], [1j, 0]]),
                 np.sqrt(1 - prob) / 2 * np.array([[1, 0], [0, -1]])]
    p = Program(Pragma("PRESERVE_BLOCK"), I(0), Pragma("END_PRESERVE_BLOCK"))
    p.define_noisy_gate("I", [0], kraus_ops)
    p.define_noisy_readout(0, 0.95, 0.76)

    # prepare ObservablesExperiment
    process_exp = ObservablesExperiment(settings=[expt], program=p)
    # list to store experiment results
    num_simulations = 10
    expts = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = []
        results = estimate_observables(qc, process_exp,
                                       symm_type=-1)
        for res in calibrate_observable_estimates(qc, list(results), noisy_program=p):
            expt_results.append(res.expectation)
        expts.append(expt_results)

    expts = np.array(expts)
    results = np.mean(expts, axis=0)
    estimated_fidelity = _point_state_fidelity_estimate(results)
    # how close is the mixed state to |0>
    expected_fidelity = (1 + prob) / 2
    np.testing.assert_allclose(expected_fidelity, estimated_fidelity, atol=3e-2)


def test_unitary_state_fidelity_readout_error():
    qc = get_qc('1q-qvm')
    # prepare experiment setting
    expt = ExperimentSetting(TensorProductState(), sZ(0))

    # rotate |0> state by some random angle about X axis
    theta = np.random.uniform(0.0, 2 * np.pi)
    p = Program(RX(theta, 0))
    p.define_noisy_readout(0, 0.95, 0.76)

    # prepare ObservablesExperiment
    process_exp = ObservablesExperiment(settings=[expt], program=p)
    # list to store experiment results
    num_simulations = 10
    expts = []
    for sim_num in range(num_simulations):
        qc.qam.random_seed = sim_num+1
        expt_results = []
        results = estimate_observables(qc, process_exp,
                                       symm_type=-1)
        for res in calibrate_observable_estimates(qc, list(results), noisy_program=p):
            expt_results.append(res.expectation)
        expts.append(expt_results)

    expts = np.array(expts)
    results = np.mean(expts, axis=0)
    estimated_fidelity = _point_state_fidelity_estimate(results)
    # how close is this state to |0>
    expected_fidelity = (np.cos(theta / 2)) ** 2
    np.testing.assert_allclose(expected_fidelity, estimated_fidelity, atol=3e-2)


def test_merge_disjoint_experiments():
    sett1 = ExperimentSetting(TensorProductState(), sX(0) * sY(1))
    sett2 = ExperimentSetting(plusZ(1), sY(1))
    sett3 = ExperimentSetting(plusZ(0), sX(0))
    sett4 = ExperimentSetting(minusX(1), sY(1))
    sett5 = ExperimentSetting(TensorProductState(), sZ(2))

    expt1 = ObservablesExperiment(settings=[sett1, sett2], program=Program(X(1)))
    expt2 = ObservablesExperiment(settings=[sett3, sett4], program=Program(Z(0)))
    expt3 = ObservablesExperiment(settings=[sett5], program=Program())

    merged_expt = merge_disjoint_experiments([expt1, expt2, expt3])
    assert len(merged_expt) == 2


def test_results_by_qubit_groups():
    er1 = ExperimentResult(
        setting=ExperimentSetting(plusX(0), sZ(0)),
        expectation=0.,
        std_err=0.,
        total_counts=1,
    )

    er2 = ExperimentResult(
        setting=ExperimentSetting(plusX(0), sZ(1)),
        expectation=0.,
        std_err=0.,
        total_counts=1,
    )

    er3 = ExperimentResult(
        setting=ExperimentSetting(plusX(0), sX(0)*sZ(1)),
        expectation=0.,
        std_err=0.,
        total_counts=1,
    )

    er4 = ExperimentResult(
        setting=ExperimentSetting(plusX(0), sX(0)*sZ(2)),
        expectation=0.,
        std_err=0.,
        total_counts=1,
    )
    groups = [(0,), (1,), (2, 0)]
    res_by_group = get_results_by_qubit_groups([er1, er2, er3, er4], groups)

    assert res_by_group == {(0,): [er1], (1,): [er2], (0, 2): [er1, er4]}
