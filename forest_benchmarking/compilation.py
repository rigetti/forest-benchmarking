from math import pi

import numpy as np
from typing import Tuple

from pyquil.gates import RX, RZ, CZ
from pyquil.quil import Program
from pyquil.quilbase import Gate


# This function is taken from cirq. License: apache 2.
def match_global_phase(a: np.ndarray,
                       b: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Phases the given matrices so that they agree on the phase of one entry.

    To maximize precision, the position with the largest entry from one of the
    matrices is used when attempting to compute the phase difference between
    the two matrices.

    :param a: The first matrix
    :param b: The second matrix
    :return: A tuple (a', b') where a' == b' implies a == b*exp(i t) for some t.
    """

    # Not much point when they have different shapes.
    if a.shape != b.shape:
        return a, b

    # Find the entry with the largest magnitude in one of the matrices.
    k = max(np.ndindex(*a.shape), key=lambda t: abs(b[t]))

    def dephase(v):
        r = np.real(v)
        i = np.imag(v)

        # Avoid introducing floating point error when axis-aligned.
        if i == 0:
            return -1 if r < 0 else 1
        if r == 0:
            return 1j if i < 0 else -1j

        return np.exp(-1j * np.arctan2(i, r))

    # Zero the phase at this entry in both matrices.
    return a * dephase(a[k]), b * dephase(b[k])


def _RY(angle, q):
    """
    A RY in terms of RX(+-pi/2) and RZ(theta)
    """
    p = Program()
    p += RX(pi / 2, q)
    p += RZ(angle, q)
    p += RX(-pi / 2, q)
    return p


def _X(q1):
    """
    An RX in terms of RX(pi/2)

    .. note:
        This introduces a global phase! Don't control this gate.

    :param q1:
    :return:
    """
    p = Program()
    p += RX(np.pi / 2, q1)
    p += RX(np.pi / 2, q1)
    return p


def _H(q1):
    """
    A Hadamard in terms of RX(+-pi/2) and RZ(theta)

    .. note:
        This introduces a different global phase! Don't control this gate.
    """
    p = Program()
    p.inst(_RY(-np.pi / 2, q1))
    p.inst(RZ(np.pi, q1))
    return p


def _CNOT(q1, q2):
    """
    A CNOT in terms of RX(+-pi/2), RZ(theta), and CZ

    .. note:
        This uses two of :py:func:`_H`, so it picks up twice the global phase.
        Don't control this gate.

    """
    p = Program()
    p.inst(_H(q2))
    p.inst(CZ(q1, q2))
    p.inst(_H(q2))
    return p


def _T(q1):
    """
    A T in terms of RZ(theta)
    """
    return Program(RZ(np.pi/4, q1))


def _SWAP(q1, q2):
    """
     A SWAP in terms of _CNOT

         .. note:
        This uses :py:func:`_CNOT`, so it picks up a global phase.
        Don't control this gate.
    """
    p = Program()
    p.inst(_CNOT(q1, q2))
    p.inst(_CNOT(q2, q1))
    p.inst(_CNOT(q1, q2))
    return p


def _CCNOT(q1, q2, q3):
    """
    A CCNOT in terms of RX(+-pi/2), RZ(theta), and CZ

    .. note:
        Don't control this gate.

    """
    p = Program()
    p.inst(_H(q3))
    p.inst(_CNOT(q2, q3))
    p.inst(_T(q3).dagger())
    p.inst(_SWAP(q2, q3))
    p.inst(_CNOT(q1, q2))
    p.inst(_T(q2))
    p.inst(_CNOT(q3, q2))
    p.inst(_T(q2).dagger())
    p.inst(_CNOT(q1, q2))
    p.inst(_SWAP(q2, q3))
    p.inst(_T(q2))
    p.inst(_T(q3))
    p.inst(_CNOT(q1, q2))
    p.inst(_H(q3))
    p.inst(_T(q1))
    p.inst(_T(q2).dagger())
    p.inst(_CNOT(q1, q2))

    return p


def is_magic_angle(angle):
    return (np.isclose(np.abs(angle), pi / 2)
            or np.isclose(np.abs(angle), pi)
            or np.isclose(angle, 0.0))


def basic_compile(program):
    new_prog = Program()
    new_prog.num_shots = program.num_shots
    new_prog.inst(program.defined_gates)
    for inst in program:
        if isinstance(inst, Gate):
            if inst.name in ['RZ', 'CZ', 'I']:
                new_prog += inst
            elif inst.name == 'RX' and is_magic_angle(inst.params[0]):
                new_prog += inst
            elif inst.name == 'RY':
                new_prog += _RY(inst.params[0], inst.qubits[0])
            elif inst.name == 'CNOT':
                new_prog += _CNOT(*inst.qubits)
            elif inst.name == 'CCNOT':
                new_prog += _CCNOT(*inst.qubits)
            elif inst.name == 'SWAP':
                new_prog += _SWAP(*inst.qubits)
            elif inst.name == 'T':
                new_prog += _T(inst.qubits[0])
            elif inst.name == "H":
                new_prog += _H(inst.qubits[0])
            elif inst.name == "X":
                new_prog += _X(inst.qubits[0])
            elif inst.name in [gate.name for gate in new_prog.defined_gates]:
                new_prog += inst
            else:
                raise ValueError(f"Unknown gate instruction {inst}")

        else:
            new_prog += inst

    new_prog.native_quil_metadata = {
        'final_rewiring': None,
        'gate_depth': None,
        'gate_volume': None,
        'multiqubit_gate_depth': None,
        'program_duration': None,
        'program_fidelity': None,
        'topological_swaps': 0,
    }
    return new_prog
