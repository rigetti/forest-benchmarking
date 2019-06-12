import numpy as np
from forest.benchmarking.operator_tools.project_state_matrix import project_state_matrix_to_physical


def test_project_state_matrix():
    """
    Test the wizard method. Example from fig 1 of maximum likelihood minimum effort
    https://doi.org/10.1103/PhysRevLett.108.070502

    :return:
    """
    eigs = np.diag(np.array(list(reversed([3.0 / 5, 1.0 / 2, 7.0 / 20, 1.0 / 10, -11.0 / 20]))))
    phys = project_state_matrix_to_physical(eigs)
    assert np.allclose(phys, np.diag([0, 0, 1.0 / 5, 7.0 / 20, 9.0 / 20]))
