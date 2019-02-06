import networkx as nx
import numpy as np
import pytest
from rpcq.messages import PyQuilExecutableResponse

from forest_benchmarking.compilation import basic_compile
from forest_benchmarking.random_operators import haar_rand_unitary
from forest_benchmarking.superop_conversion import vec, unvec, kraus2choi
from forest_benchmarking.tomography import _constraint_project, linear_inv_process_estimate
from forest_benchmarking.tomography import proj_to_cp, proj_to_tni, \
    generate_process_tomography_experiment, pgdb_process_estimate, \
    proj_to_tp
from forest_benchmarking.utils import sigma_x, partial_trace
from pyquil import Program
from pyquil.api import QuantumComputer
from pyquil.api._compiler import _extract_attribute_dictionary_from_program
from pyquil.api._qac import AbstractCompiler
from pyquil.device import NxDevice
from pyquil.gates import CNOT, I, RZ
from pyquil.numpy_simulator import NumpyWavefunctionSimulator
from pyquil.operator_estimation import measure_observables, ExperimentResult, TomographyExperiment, \
    _one_q_state_prep
from pyquil.pyqvm import PyQVM
from pyquil.unitary_tools import program_unitary


def test_proj_to_cp():
    state = vec(np.array([[1., 0], [0, 1.]]))
    assert np.allclose(state, proj_to_cp(state))

    state = vec(np.array([[1.5, 0], [0, 10]]))
    assert np.allclose(state, proj_to_cp(state))

    state = vec(np.array([[-1, 0], [0, 1.]]))
    cp_state = vec(np.array([[0, 0], [0, 1.]]))
    assert np.allclose(cp_state, proj_to_cp(state))

    state = vec(np.array([[0, 1], [1, 0]]))
    cp_state = vec(np.array([[.5, .5], [.5, .5]]))
    assert np.allclose(cp_state, proj_to_cp(state))

    state = vec(np.array([[0, -1j], [1j, 0]]))
    cp_state = vec(np.array([[.5, -.5j], [.5j, .5]]))
    assert np.allclose(cp_state, proj_to_cp(state))


def test_proj_to_tp():
    # Identity process is trace preserving, so no change
    state = vec(kraus2choi(np.eye(2)))
    assert np.allclose(state, proj_to_tp(state))

    # Bit flip process is trace preserving, so no change
    state = vec(kraus2choi(sigma_x))
    assert np.allclose(state, proj_to_tp(state))


def test_cptp():
    # Identity process is cptp, so no change
    state = np.array(kraus2choi(np.eye(2)))
    assert np.allclose(state, _constraint_project(state))

    # Small perturbation shouldn't change too much
    state = np.array([[1.001, 0., 0., .99], [0., 0., 0., 0.],
                      [0., 0., 0., 0.], [1.004, 0., 0., 1.01]])
    assert np.allclose(state, _constraint_project(state), atol=.01)

    # Bit flip process is cptp, so no change
    state = kraus2choi(sigma_x)
    assert np.allclose(state, _constraint_project(state))


def test_proj_to_tni():
    state = np.array([[0., 0., 0., 0.], [0., 1.01, 1.01, 0.], [0., 1., 1., 0.], [0., 0., 0., 0.]])
    trace_non_increasing = unvec(proj_to_tni(vec(state)))
    pt = partial_trace(trace_non_increasing, dims=[2, 2], keep=[0])
    assert np.allclose(pt, np.eye(2))


def get_test_qc(n_qubits):
    class BasicQVMCompiler(AbstractCompiler):
        def quil_to_native_quil(self, program: Program):
            return basic_compile(program)

        def native_quil_to_executable(self, nq_program: Program):
            return PyQuilExecutableResponse(
                program=nq_program.out(),
                attributes=_extract_attribute_dictionary_from_program(nq_program))

    return QuantumComputer(
        name='testing-qc',
        qam=PyQVM(n_qubits=n_qubits, seed=52),
        device=NxDevice(nx.complete_graph(n_qubits)),
        compiler=BasicQVMCompiler(),
    )


def wfn_measure_observables(n_qubits, tomo_expt: TomographyExperiment):
    wfn = NumpyWavefunctionSimulator(n_qubits)
    for settings in tomo_expt:
        for setting in settings:
            prog = Program()
            for oneq_state in setting.in_state.states:
                prog += _one_q_state_prep(oneq_state)
            prog += tomo_expt.program

            yield ExperimentResult(
                setting=setting,
                expectation=wfn.reset().do_program(prog).expectation(setting.out_operator),
                stddev=0.,
                total_counts=1,  # don't set to zero unless you want nans
            )


@pytest.fixture(params=['pauli', 'sic'])
def basis(request):
    return request.param


@pytest.fixture(params=['sampling', 'wfn'])
def measurement_func(request):
    if request.param == 'wfn':
        return lambda expt: list(wfn_measure_observables(n_qubits=2, tomo_expt=expt))
    elif request.param == 'sampling':
        return lambda expt: list(measure_observables(qc=get_test_qc(n_qubits=2),
                                                     tomo_experiment=expt, n_shots=100_000))
    else:
        raise ValueError()


@pytest.fixture()
def single_q_tomo_fixture(basis, measurement_func):
    qubits = [0]

    # Generate random unitary
    u_rand = haar_rand_unitary(2 ** 1, rs=np.random.RandomState(52))
    process = Program().defgate("RandUnitary", u_rand)
    process.inst([("RandUnitary", q) for q in qubits])

    process = Program()
    process += RZ(np.pi, 0)
    u_rand = np.array([[1, 0], [0, -1]])

    # Get data from QVM
    tomo_expt = generate_process_tomography_experiment(process, qubits, in_basis=basis)
    results = measurement_func(tomo_expt)

    return qubits, results, u_rand


def test_single_q_pgdb(single_q_tomo_fixture):
    qubits, results, u_rand = single_q_tomo_fixture

    process_choi_est = pgdb_process_estimate(results, qubits=qubits)
    process_choi_true = kraus2choi(u_rand)
    np.testing.assert_allclose(process_choi_true, process_choi_est, atol=1e-2)


def test_single_q_linear_inversion(single_q_tomo_fixture):
    qubits, results, u_rand = single_q_tomo_fixture

    process_choi_est = linear_inv_process_estimate(results, qubits[::-1])
    process_choi_true = kraus2choi(u_rand)

    print(np.real_if_close(np.round(process_choi_est, 2)))
    print(np.real_if_close(np.round(process_choi_true, 2)))

    np.testing.assert_allclose(process_choi_true, process_choi_est, atol=1e-2)


@pytest.fixture()
def two_q_tomo_fixture(basis, measurement_func):
    qubits = [0, 1]

    # Generate random unitary
    u_rand = haar_rand_unitary(2 ** len(qubits), rs=np.random.RandomState(52))
    process = Program().defgate("RandUnitary", u_rand)
    process += ("RandUnitary", qubits[0], qubits[1])

    # testing
    process = Program()
    process += I(0)
    process += I(1)
    u_rand = program_unitary(process, 2)

    tomo_expt = generate_process_tomography_experiment(process, qubits, in_basis=basis)
    results = measurement_func(tomo_expt)

    return qubits, results, u_rand


def test_two_q_pgdb(two_q_tomo_fixture):
    qubits, results, u_rand = two_q_tomo_fixture

    process_choi_est = pgdb_process_estimate(results, qubits=qubits)
    process_choi_true = kraus2choi(u_rand)

    print()
    print(np.real_if_close(np.round(process_choi_est, 1)))
    print(np.real_if_close(np.round(process_choi_true, 1)))

    np.testing.assert_allclose(process_choi_true, process_choi_est, atol=0.05)
