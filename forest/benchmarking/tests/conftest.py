import os

import pytest
from httpx import RequestError
from pyquil import Program, get_qc
from pyquil.api import WavefunctionSimulator, BenchmarkConnection
from pyquil.gates import I
from pyquil.paulis import sX
from pyquil.quantum_processor import NxQuantumProcessor

PATH = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(scope="module")
def test_qc():
    import networkx as nx
    from forest.benchmarking.compilation import basic_compile
    from pyquil.api import QuantumComputer, QVM
    from pyquil.api._compiler import AbstractCompiler
    from pyquil.gates import I

    class BasicQVMCompiler(AbstractCompiler):
        def quil_to_native_quil(self, program: Program, protoquil=None):
            return basic_compile(program)

        def native_quil_to_executable(self, nq_program: Program):
            return nq_program

        def reset(self):
            pass

    try:
        qc = QuantumComputer(
            name="testing-qc",
            qam=QVM(random_seed=52),
            compiler=BasicQVMCompiler(
                quantum_processor=NxQuantumProcessor(nx.complete_graph(2)),
                timeout=20.0,
                client_configuration=None,
            ),
        )
        qc.run(Program(I(0)))
        return qc
    except (RequestError, TimeoutError) as e:
        return pytest.skip("This test requires a running local QVM: {}".format(e))


@pytest.fixture(scope="module")
def qvm():
    try:
        qc = get_qc("9q-square-qvm", compiler_timeout=10.0)
        qc.run(Program(I(0)))
        return qc
    except (RequestError, TimeoutError) as e:
        return pytest.skip(
            "This test requires a running local QVM and quilc: {}".format(e)
        )


@pytest.fixture(scope="session")
def wfn():
    return WavefunctionSimulator()


@pytest.fixture(scope="session")
def benchmarker():
    try:
        benchmarker = BenchmarkConnection(timeout=30)
        benchmarker.apply_clifford_to_pauli(Program(I(0)), sX(0))
        return benchmarker
    except (RequestError, TimeoutError) as e:
        return pytest.skip(
            "This test requires a running local benchmarker endpoint (ie quilc): {}".format(
                e
            )
        )


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
