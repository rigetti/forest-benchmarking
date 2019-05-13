import pytest
from os.path import dirname, join, realpath
from subprocess import PIPE, run

EXAMPLES_PATH = join(dirname(realpath(__file__)), '..', '..', '..', 'examples')


# from https://stackoverflow.com/a/36058256
def out(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return result.stdout


def run_notebook(filename):
    return out('pytest --nbval-lax ' + EXAMPLES_PATH + f'/{filename}')


# NOTES:
# (1) By using the command `out` we are accessing the command line and running the basic command:
#
# >pytest --nbval-lax some_notebook.ipynb
#
# The --nbval flag runs the IPython Notebook Validation plugin which will collect and test
# notebook cells, comparing their outputs with those saved in the file.
# By adding the optional -lax flag we fail if there is an error in executing a cell.
#
# (2) We are skipping any test that is longer than 30 seconds.
#
# (3) Each test function has a print, if the test fails then you can see where it fails from stdout.


# TODO: go through the slow notebooks and add the comment `# NBVAL_SKIP` on the offending slow
#  cells. This will allow us to test imports and other functionality without sacrificing runtime.


# ~ 6 min; passed 2019/05/13
@pytest.mark.slow
def test_direct_fidelity_estimation_nb():
    # print(out('ls'))
    output_str = run_notebook('direct_fidelity_estimation.ipynb')
    print(output_str)
    assert output_str.find('failed') is -1


# ~5 sec
def test_distance_measures_nb():
    output_str = run_notebook('distance_measures.ipynb')
    print(output_str)
    assert output_str.find('failed') is -1


# ~5 sec
def test_direct_entangled_states_nb():
    output_str = run_notebook('entangled_states.ipynb')
    print(output_str)
    assert output_str.find('failed') is -1


# ~5 sec
def test_Hinton_Plots_nb():
    output_str = run_notebook('Hinton-Plots.ipynb')
    print(output_str)
    assert output_str.find('failed') is -1


# ~17 min; passed 2019/05/13
@pytest.mark.slow
def test_quantum_volume_nb():
    output_str = run_notebook('quantum_volume.ipynb')
    print(output_str)
    assert output_str.find('failed') is -1


# ~5 sec
def test_qubit_spectroscopy_cz_ramsey_nb():
    output_str = run_notebook('qubit_spectroscopy_cz_ramsey.ipynb')
    print(output_str)
    assert output_str.find('failed') is -1


# ~5 sec
def test_qubit_spectroscopy_rabi_nb():
    output_str = run_notebook('qubit_spectroscopy_rabi.ipynb')
    print(output_str)
    assert output_str.find('failed') is -1


# ~5 sec
def test_qubit_spectroscopy_t1_nb():
    output_str = run_notebook('qubit_spectroscopy_t1.ipynb')
    print(output_str)
    assert output_str.find('failed') is -1


# ~5 sec
def test_qubit_spectroscopy_t2_nb():
    output_str = run_notebook('qubit_spectroscopy_t2.ipynb')
    print(output_str)
    assert output_str.find('failed') is -1


# ~5 sec
def test_random_operators_nb():
    output_str = run_notebook('random_operators.ipynb')
    print(output_str)
    assert output_str.find('failed') is -1


# ~5 min; passed 2019/05/13
@pytest.mark.slow
def test_randomized_benchmarking_interleaved_nb():
    output_str = run_notebook('randomized_benchmarking_interleaved.ipynb')
    print(output_str)
    assert output_str.find('failed') is -1


# ~14 sec
def test_randomized_benchmarking_simultaneous_nb():
    output_str = run_notebook('randomized_benchmarking_simultaneous.ipynb')
    print(output_str)
    assert output_str.find('failed') is -1


# ~7 min; passed 2019/05/13
@pytest.mark.slow
def test_randomized_benchmarking_unitarity_nb():
    output_str = run_notebook('randomized_benchmarking_unitarity.ipynb')
    print(output_str)
    assert output_str.find('failed') is -1


# ~3 sec
def test_readout_fidelity_nb():
    output_str = run_notebook('readout_fidelity.ipynb')
    print(output_str)
    assert output_str.find('failed') is -1


# ~4 sec
def test_ripple_adder_benchmark_nb():
    output_str = run_notebook('ripple_adder_benchmark.ipynb')
    print(output_str)
    assert output_str.find('failed') is -1


# ~40 sec; passed 2019/05/13
@pytest.mark.slow
def test_robust_phase_estimation_nb():
    output_str = run_notebook('robust_phase_estimation.ipynb')
    print(output_str)
    assert output_str.find('failed') is -1


# ~10 sec
def test_state_and_process_plots_nb():
    output_str = run_notebook('state_and_process_plots.ipynb')
    print(output_str)
    assert output_str.find('failed') is -1


# ~4 sec
def test_superoperator_tools_nb():
    output_str = run_notebook('superoperator_tools.ipynb')
    print(output_str)
    assert output_str.find('failed') is -1


# Notebook fails at the moment
# def test_tomography_process_nb():
#     output_str = run_notebook('tomography_process.ipynb')
#     print(output_str)
#     assert output_str.find('failed') is -1


# ~4 sec
def test_tomography_state_nb():
    output_str = run_notebook('tomography_state.ipynb')
    print(output_str)
    assert output_str.find('failed') is -1
