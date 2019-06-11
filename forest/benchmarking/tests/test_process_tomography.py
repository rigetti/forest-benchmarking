import networkx as nx
import numpy as np
import pytest
from requests.exceptions import RequestException
from rpcq.messages import PyQuilExecutableResponse

from forest.benchmarking.compilation import basic_compile
from forest.benchmarking.random_operators import haar_rand_unitary
from forest.benchmarking.superoperator_tools import kraus2choi
from forest.benchmarking.tomography import generate_process_tomography_experiment, \
    pgdb_process_estimate
from forest.benchmarking.observable_estimation import estimate_observables, ExperimentResult, \
    ObservablesExperiment, \
    _one_q_state_prep
from pyquil import Program
from pyquil import gate_matrices as mat
from pyquil.api import QVM
from pyquil.gates import CNOT, X
from pyquil.numpy_simulator import NumpyWavefunctionSimulator


@pytest.fixture
def test_qc():
    from pyquil.api import ForestConnection, QuantumComputer
    from pyquil.api._compiler import _extract_attribute_dictionary_from_program
    from pyquil.api._qac import AbstractCompiler
    from pyquil.device import NxDevice
    from pyquil.gates import I

    class BasicQVMCompiler(AbstractCompiler):

        def quil_to_native_quil(self, program: Program):
            return basic_compile(program)

        def native_quil_to_executable(self, nq_program: Program):
            return PyQuilExecutableResponse(
                program=nq_program.out(),
                attributes=_extract_attribute_dictionary_from_program(nq_program))
    try:
        qc = QuantumComputer(
            name='testing-qc',
            qam=QVM(connection=ForestConnection(), random_seed=52),
            device=NxDevice(nx.complete_graph(2)),
            compiler=BasicQVMCompiler(),
        )
        qc.run_and_measure(Program(I(0)), trials=1)
        return qc
    except (RequestException, TimeoutError) as e:
        return pytest.skip("This test requires a running local QVM: {}".format(e))


def wfn_estimate_observables(n_qubits, tomo_expt: ObservablesExperiment):
    if len(tomo_expt.program.defined_gates) > 0:
        raise pytest.skip("Can't do wfn on defined gates yet")
    wfn = NumpyWavefunctionSimulator(n_qubits)
    for settings in tomo_expt:
        for setting in settings:
            prog = Program()
            for oneq_state in setting.in_state.states:
                prog += _one_q_state_prep(oneq_state)
            prog += tomo_expt.program

            yield ExperimentResult(
                setting=setting,
                expectation=wfn.reset().do_program(prog).expectation(setting.observable),
                std_err=0.,
                total_counts=1,  # don't set to zero unless you want nans
            )


@pytest.fixture(params=['pauli', 'sic'])
def basis(request):
    return request.param


@pytest.fixture(params=['sampling', 'wfn'])
def measurement_func(request, test_qc):
    if request.param == 'wfn':
        return lambda expt: list(wfn_estimate_observables(n_qubits=2, tomo_expt=expt))
    elif request.param == 'sampling':
        return lambda expt: list(estimate_observables(qc=test_qc, obs_expt=expt, num_shots=500))
    else:
        raise ValueError()


@pytest.fixture(params=['X', 'haar'])
def single_q_process(request):
    if request.param == 'X':
        return Program(X(0)), mat.X
    elif request.param == 'haar':
        u_rand = haar_rand_unitary(2 ** 1, rs=np.random.RandomState(52))
        process = Program().defgate("RandUnitary", u_rand)
        process += ("RandUnitary", 0)
        return process, u_rand


@pytest.fixture()
def single_q_tomo_fixture(basis, single_q_process, measurement_func):
    qubits = [0]
    process, u_rand = single_q_process
    tomo_expt = generate_process_tomography_experiment(process, qubits, in_basis=basis)
    results = measurement_func(tomo_expt)

    return qubits, results, u_rand


def test_single_q_pgdb(single_q_tomo_fixture):
    qubits, results, u_rand = single_q_tomo_fixture

    process_choi_est = pgdb_process_estimate(results, qubits=qubits)
    process_choi_true = kraus2choi(u_rand)
    np.testing.assert_allclose(process_choi_true, process_choi_est, atol=.05)


@pytest.fixture(params=['CNOT', 'haar'])
def two_q_process(request):
    if request.param == 'CNOT':
        return Program(CNOT(0, 1)), mat.CNOT
    elif request.param == 'haar':
        u_rand = haar_rand_unitary(2 ** 2, rs=np.random.RandomState(52))
        process = Program().defgate("RandUnitary", u_rand)
        process += ("RandUnitary", 0, 1)
        return process, u_rand
    else:
        raise ValueError()


@pytest.fixture()
def two_q_tomo_fixture(basis, two_q_process, measurement_func):
    qubits = [0, 1]
    process, u_rand = two_q_process
    tomo_expt = generate_process_tomography_experiment(process, qubits, in_basis=basis)
    results = measurement_func(tomo_expt)
    return qubits, results, u_rand


def test_two_q_pgdb(two_q_tomo_fixture):
    qubits, results, u_rand = two_q_tomo_fixture
    process_choi_est = pgdb_process_estimate(results, qubits=qubits)
    process_choi_true = kraus2choi(u_rand)
    np.testing.assert_allclose(process_choi_true, process_choi_est, atol=0.05)
