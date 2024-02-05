import pytest
from os.path import dirname, join, realpath
from subprocess import PIPE, run

EXAMPLES_PATH = join(dirname(realpath(__file__)), '..', '..', '..', 'docs', 'examples')


# from https://stackoverflow.com/a/36058256
def out(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return result.stdout


def run_notebook(filename: str):
    return out('python -m pytest --nbval-lax ' + EXAMPLES_PATH + f'/{filename}')


def _is_not_empty_str(string: str) -> bool:
    # bool(string) tests if the string is empty
    # bool(string.strip()) tests if the string is full of spaces
    return bool(string and string.strip())


def check_for_failure_and_empty_output(output: str):
    """
    Searches the string for the word 'failed' and checks if the string is empty of full of spaces.
    If any of those conditions are met False is returned.

    :param output: stdout from 'pytest --nbval-lax notebook.ipynb'
    :return: bool
    """
    return (output.find('failed') is -1) and _is_not_empty_str(output)


# NOTES:
# (1) By using the command `out` we are accessing the command line and running the basic command:
#
# >pytest --nbval-lax some_notebook.ipynb
#
# The --nbval flag runs the notebook validation plugin (to collect and test notebook cells,
# comparing their outputs with those saved in the file). By adding the optional -lax flag we fail
# if there is an error in executing a cell.
#
# (2) We are skipping any test that is longer than 30 seconds.
#
# (3) Each test function has a print, if the test fails then you can see where it fails from stdout.


# ~ 6 min; passed 2019/05/13
@pytest.mark.slow
def test_direct_fidelity_estimation_nb():
    # print(out('ls'))
    output_str = run_notebook('direct_fidelity_estimation.ipynb')
    print(output_str)
    assert check_for_failure_and_empty_output(output_str)


# ~6 sec
def test_distance_measures_nb():
     output_str = run_notebook('distance_measures.ipynb')
     print(output_str)
     assert check_for_failure_and_empty_output(output_str)


# ~27 sec
def test_direct_entangled_states_nb():
    output_str = run_notebook('entangled_states.ipynb')
    print(output_str)
    assert check_for_failure_and_empty_output(output_str)


# ~5 sec
def test_Hinton_Plots_nb():
    output_str = run_notebook('Hinton-Plots.ipynb')
    print(output_str)
    assert check_for_failure_and_empty_output(output_str)


# ~17 min; passed 2019/05/13
@pytest.mark.slow
def test_quantum_volume_nb():
    output_str = run_notebook('quantum_volume.ipynb')
    print(output_str)
    assert check_for_failure_and_empty_output(output_str)


# ~5 sec
def test_qubit_spectroscopy_cz_ramsey_nb():
    output_str = run_notebook('qubit_spectroscopy_cz_ramsey.ipynb')
    print(output_str)
    assert check_for_failure_and_empty_output(output_str)


# ~5 sec
def test_qubit_spectroscopy_rabi_nb():
    output_str = run_notebook('qubit_spectroscopy_rabi.ipynb')
    print(output_str)
    assert check_for_failure_and_empty_output(output_str)


# ~5 sec
def test_qubit_spectroscopy_t1_nb():
    output_str = run_notebook('qubit_spectroscopy_t1.ipynb')
    print(output_str)
    assert check_for_failure_and_empty_output(output_str)


# ~5 sec
def test_qubit_spectroscopy_t2_nb():
    output_str = run_notebook('qubit_spectroscopy_t2.ipynb')
    print(output_str)
    assert check_for_failure_and_empty_output(output_str)


# ~5 sec
def test_random_operators_nb():
    output_str = run_notebook('random_operators.ipynb')
    print(output_str)
    assert check_for_failure_and_empty_output(output_str)


# ~5 min; passed 2019/05/13
@pytest.mark.slow
def test_randomized_benchmarking_interleaved_nb():
    output_str = run_notebook('randomized_benchmarking_interleaved.ipynb')
    print(output_str)
    assert check_for_failure_and_empty_output(output_str)


# ~14 sec
def test_randomized_benchmarking_simultaneous_nb():
    output_str = run_notebook('randomized_benchmarking_simultaneous.ipynb')
    print(output_str)
    assert check_for_failure_and_empty_output(output_str)


# ~7 min; passed 2019/05/13
@pytest.mark.slow
def test_randomized_benchmarking_unitarity_nb():
    output_str = run_notebook('randomized_benchmarking_unitarity.ipynb')
    print(output_str)
    assert check_for_failure_and_empty_output(output_str)


# ~3 sec
def test_readout_fidelity_nb():
    output_str = run_notebook('readout_fidelity.ipynb')
    print(output_str)
    assert check_for_failure_and_empty_output(output_str)


# ~41 sec; passed 2019/06/24
@pytest.mark.slow
def test_ripple_adder_benchmark_nb():
    output_str = run_notebook('ripple_adder_benchmark.ipynb')
    print(output_str)
    assert check_for_failure_and_empty_output(output_str)


# ~40 sec; passed 2019/05/13
@pytest.mark.slow
def test_robust_phase_estimation_nb():
    output_str = run_notebook('robust_phase_estimation.ipynb')
    print(output_str)
    assert check_for_failure_and_empty_output(output_str)


# ~10 sec
def test_state_and_process_plots_nb():
    output_str = run_notebook('state_and_process_plots.ipynb')
    print(output_str)
    assert check_for_failure_and_empty_output(output_str)


# ~4 sec
def test_superoperator_tools_nb():
    output_str = run_notebook('superoperator_tools.ipynb')
    print(output_str)
    assert check_for_failure_and_empty_output(output_str)

# ~10 sec
def test_tomography_process_nb():
    output_str = run_notebook('tomography_process.ipynb')
    print(output_str)
    assert check_for_failure_and_empty_output(output_str)


# ~4 sec
def test_tomography_state_nb():
   output_str = run_notebook('tomography_state.ipynb')
   print(output_str)
   assert check_for_failure_and_empty_output(output_str)
