import inspect
import random
from math import pi

import numpy as np
import pytest

from pyquil.gates import *
from pyquil.quil import Program
from pyquil.gates import RX, RZ
from forest_benchmarking.compilation import _RY, basic_compile, _CNOT, _H, _X, match_global_phase

try:
    from pyquil.unitary_tools import program_unitary

    unitary_tools = True
except ImportError:
    unitary_tools = False


def assert_all_close_up_to_global_phase(actual, desired, rtol: float = 1e-7, atol: float = 0):
    actual, desired = match_global_phase(actual, desired)
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)


def test_basic_compile_defgate():
    p = Program()
    p.inst(RX(pi, 0))
    p.defgate("test", [[0, 1], [1, 0]])
    p.inst(("test", 2))
    p.inst(RZ(pi / 2, 0))

    assert p == basic_compile(p)


@pytest.mark.skipif(not unitary_tools, reason='Requires unitary_tools')
def test_RY():
    for theta in np.linspace(-2 * np.pi, 2 * np.pi):
        u1 = program_unitary(Program(RY(theta, 0)), n_qubits=1)
        u2 = program_unitary(_RY(theta, 0), n_qubits=1)
        assert_all_close_up_to_global_phase(u1, u2)


@pytest.mark.skipif(not unitary_tools, reason='Requires unitary_tools')
def test_X():
    u1 = program_unitary(Program(X(0)), n_qubits=1)
    u2 = program_unitary(_X(0), n_qubits=1)
    assert_all_close_up_to_global_phase(u1, u2, atol=1e-12)


@pytest.mark.skipif(not unitary_tools, reason='Requires unitary_tools')
def test_H():
    u1 = program_unitary(Program(H(0)), n_qubits=1)
    u2 = program_unitary(_H(0), n_qubits=1)
    assert_all_close_up_to_global_phase(u1, u2, atol=1e-12)


@pytest.mark.skipif(not unitary_tools, reason='Requires unitary_tools')
def test_CNOT():
    u1 = program_unitary(Program(CNOT(0, 1)), n_qubits=2)
    u2 = program_unitary(_CNOT(0, 1), n_qubits=2)
    assert_all_close_up_to_global_phase(u1, u2, atol=1e-12)

    u1 = program_unitary(Program(CNOT(1, 0)), n_qubits=2)
    u2 = program_unitary(_CNOT(1, 0), n_qubits=2)
    assert_all_close_up_to_global_phase(u1, u2, atol=1e-12)


# Note to developers: unsupported gates are commented out.
QUANTUM_GATES = {'I': I,
                 'X': X,
                 # 'Y': Y,
                 # 'Z': Z,
                 'H': H,
                 # 'S': S,
                 # 'T': T,
                 # 'PHASE': PHASE,
                 'RX': RX,
                 'RY': RY,
                 'RZ': RZ,
                 'CZ': CZ,
                 'CNOT': CNOT,
                 # 'CCNOT': CCNOT,
                 # 'CPHASE00': CPHASE00,
                 # 'CPHASE01': CPHASE01,
                 # 'CPHASE10': CPHASE10,
                 # 'CPHASE': CPHASE,
                 # 'SWAP': SWAP,
                 # 'CSWAP': CSWAP,
                 # 'ISWAP': ISWAP,
                 # 'PSWAP': PSWAP
                 }


def _generate_random_program(n_qubits, length):
    """Randomly sample gates and arguments (qubits, angles)"""
    if n_qubits < 2:
        raise ValueError("Please request n_qubits >= 2 so we can use 2-qubit gates.")

    gates = list(QUANTUM_GATES.values())
    prog = Program()
    for _ in range(length):
        gate = random.choice(gates)
        possible_qubits = set(range(n_qubits))
        sig = inspect.signature(gate)

        param_vals = []
        for param in sig.parameters:
            if param in ['qubit', 'q1', 'q2', 'control',
                         'control1', 'control2', 'target', 'target_1', 'target_2',
                         'classical_reg']:
                param_val = random.choice(list(possible_qubits))
                possible_qubits.remove(param_val)
            elif param == 'angle':
                # TODO: support rx(theta)
                if gate == RX:
                    param_val = random.choice([-1, -0.5, 0, 0.5, 1]) * pi
                else:
                    param_val = random.uniform(-2 * pi, 2 * pi)
            else:
                raise ValueError("Unknown gate parameter {}".format(param))

            param_vals.append(param_val)

        prog += gate(*param_vals)

    return prog


@pytest.fixture(params=list(range(2, 5)))
def n_qubits(request):
    return request.param


@pytest.fixture(params=[2, 50, 100])
def prog_length(request):
    return request.param


@pytest.mark.skipif(not unitary_tools, reason='Requires unitary_tools')
def test_random_progs(n_qubits, prog_length):
    for repeat_i in range(10):
        prog = _generate_random_program(n_qubits=n_qubits, length=prog_length)
        u1 = program_unitary(prog, n_qubits=n_qubits)
        u2 = program_unitary(basic_compile(prog), n_qubits=n_qubits)

        assert_all_close_up_to_global_phase(u1, u2, atol=1e-12)
