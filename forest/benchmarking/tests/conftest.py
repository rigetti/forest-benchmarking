import os

from requests.exceptions import RequestException
import pytest
from unittest.mock import create_autospec, Mock

from pyquil.api import WavefunctionSimulator, ForestConnection, QVMConnection, get_benchmarker
from pyquil.api._errors import UnknownApiError
from pyquil.paulis import sX
from pyquil import Program, get_qc
from pyquil.gates import I, MEASURE
from pyquil.device import ISA, Device


PATH = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(scope='module')
def test_qc():
    import networkx as nx
    from requests.exceptions import RequestException
    from rpcq.messages import PyQuilExecutableResponse
    from forest.benchmarking.compilation import basic_compile
    from pyquil.api import ForestConnection, QuantumComputer, QVM
    from pyquil.api._compiler import _extract_attribute_dictionary_from_program
    from pyquil.api._qac import AbstractCompiler
    from pyquil.device import NxDevice
    from pyquil.gates import I

    class BasicQVMCompiler(AbstractCompiler):

        def quil_to_native_quil(self, program: Program, protoquil=None):
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


@pytest.fixture(scope='module')
def qvm():
    try:
        qc = get_qc('9q-square-qvm')
        qc.compiler.client.timeout = 1
        qc.run_and_measure(Program(I(0)), trials=1)
        return qc
    except (RequestException, TimeoutError) as e:
        return pytest.skip("This test requires a running local QVM and quilc: {}".format(e))


@pytest.fixture(scope='session')
def forest():
    try:
        connection = ForestConnection()
        connection._wavefunction(Program(I(0)), 52)
        return connection
    except (RequestException, UnknownApiError) as e:
        return pytest.skip("This test requires a Forest connection: {}".format(e))


@pytest.fixture(scope='session')
def wfn(forest):
    return WavefunctionSimulator(connection=forest)


@pytest.fixture(scope='session')
def cxn():
    # DEPRECATED
    try:
        cxn = QVMConnection(endpoint='http://localhost:5000')
        cxn.run(Program(I(0), MEASURE(0, 0)), [0])
        return cxn
    except RequestException as e:
        return pytest.skip("This test requires a running local QVM: {}".format(e))


@pytest.fixture(scope='session')
def benchmarker():
    try:
        benchmarker = get_benchmarker(timeout=1)
        benchmarker.apply_clifford_to_pauli(Program(I(0)), sX(0))
        return benchmarker
    except (RequestException, TimeoutError) as e:
        return pytest.skip("This test requires a running local benchmarker endpoint (ie quilc): {}"
                           .format(e))


@pytest.fixture(scope='session')
def mock_get_devices():
    # Generated from ISA.to_dict.
    acorn = {'1Q': {'0': {}, '1': {}, '2': {}, '3': {'dead': True}, '4': {}, '5': {}, '6': {}, '7': {}, '8': {},
                    '9': {}, '10': {}, '11': {}, '12': {}, '13': {}, '14': {}, '15': {}, '16': {}, '17': {}, '18': {},
                    '19': {}},
     '2Q': {'0-5': {}, '0-6': {}, '1-6': {}, '1-7': {}, '2-7': {}, '2-8': {}, '4-9': {}, '5-10': {}, '6-11': {},
            '7-12': {}, '8-13': {}, '9-14': {}, '10-15': {}, '10-16': {}, '11-16': {}, '11-17': {}, '12-17': {},
            '12-18': {}, '13-18': {}, '13-19': {}, '14-19': {}}}
    agave = {'1Q': {'0': {}, '1': {}, '2': {}, '3': {}, '4': {}, '5': {}, '6': {}, '7': {}},
             '2Q': {'0-1': {}, '1-2': {}, '2-3': {}, '3-4': {}, '4-5': {}, '5-6': {}, '6-7': {}, '7-0': {}}}
    mock_acorn = Mock(spec=Device)
    mock_agave = Mock(spec=Device)
    mock_acorn.isa = ISA.from_dict(acorn)
    mock_agave.isa = ISA.from_dict(agave)

    # Make sure we are matching the signature.
    mocked_function = create_autospec(get_devices)

    def side_effect(as_dict=True):
        if as_dict:
            return {'19Q-Acorn': mock_acorn,
                    '8Q-Agave': mock_agave}
        else:
            return {acorn, agave}
    mocked_function.side_effect = side_effect
    return mocked_function


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: mark test to run only with --runslow option."
    )
