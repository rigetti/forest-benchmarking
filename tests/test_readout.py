import re

from pyquil.gates import I, RX, CNOT, MEASURE
from pyquil.quil import Program

from forest_qcvv.readout import get_flipped_program


def test_get_flipped_program():
    program = Program([
        I(0),
        RX(2.3, 1),
        CNOT(0, 1),
        MEASURE(0, 0),
        MEASURE(1, 1),
    ])

    flipped_program = get_flipped_program(program)

    lines = flipped_program.out().splitlines()
    matched = 0
    for l1, l2 in zip(lines, lines[1:]):
        ma = re.match(r'MEASURE (\d) ro\[(\d)\]', l2)
        if ma is not None:
            matched += 1
            assert int(ma.group(1)) == int(ma.group(2))
            assert l1 == 'RX(pi) {}'.format(int(ma.group(1)))

    assert matched == 2
