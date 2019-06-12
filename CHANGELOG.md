Changelog
=========

v0.6 (June 11, 2018)
--------------------
Breaking Changes:

- `operator_estimation.py` is entirely replaced. All changes from (gh-135) except where stated otherwise.

- `operator_estimation.py` -> `observable_estimation.py` (gh-138)

- `pyquil.operator_estimation` dependencies replaced with `forest.benchmarking.observable_estimation` (gh-129,132,133,134,135)

- `operator_estimation.TomographyExperiment.out_op` -> `observable_estimation.ObservablesExperiment.out_observable`

- `operator_estimation.measure_observables` -> `observable_estimation.estimate_observables`

- `operator_estimation.group_experiments` -> `observable_estimation.group_settings`

- `utils.all_pauli_terms` -> `utils.all_traceless_pauli_terms` 

- `DFEData` and `DFEEstimate` dataclasses removed in favor of `ExperimentResult` and tuple of results respectively (gh-134).

- plotting moved out of `qubit_spectroscopy`; instead, use `fit_*_results()` to get a `lmfit.model.ModelResult` and pass this into `analysis.fitting.make_figure()`

- `pandas.DataFrame` is no longer used in `randomized_benchmarking` (gh-133), `qubit_spectroscopy` (gh-129), and `robust_phase_estimation` (gh-135). These now make use of `observable_estimation.ObservablesExperiment`, and as such the API has changed substantially. Please refer to example notebooks for new usage.

- `pandas.DataFrame` methods removed from `quantum_volume`. See examples notebook for alternative usage (gh-136). 

- `utils.determine_simultaneous_grouping()` removed in favor of similar functionality in `observable_estimation.group_settings`

- SIC state helpers removed from `utils`

- default `utils.str_to_pauli_term` now associates left-most character of input `pauli_str` with qubit 0. If `qubit_labels` are provided then the qubits label the characters in order.  

- `utils.all_pauli_*_terms` -> `utils.all_traceless_pauli_*_terms` to reflect fact that identity term is not included.

- `utils.pauli_basis_pauli_basis_measurements` removed

Improvements and Changes:

- `analysis.fitting` has been expanded to hold each fit model used in `qubit_spectroscopy` and `randomized_benchmarking`

- `RX(angle)` for arbitrary angle now supported by `basic_compile`

- `observable_estimation.estimate_observables` (formerly `pyquil.operator_estimation.measure_observables`) has been decomposed into separate steps:
    - `generate_experiment_programs()`, which converts and experiment into a list of programs
    - optional symmetrization, which expands each program into a group of programs that accomplish symmetrization
    - data collection and optional `consolidate_symmetrization_outputs()` which collects data used for estimates
    - `calibrate_observable_estimates()` which can be used to update estimates after collecting calibration data
   
- `plotting.state_process.plot_pauli_transfer_matrix()` now automatically casts input to `np.real_if_close` 

- `_state_tomo_settings()` no longer includes all-Identity term.


Bugfixes:

- t2 experiments now implement correct echo sequence

v0.5 (June 10, 2018)
--------------------
Improvements and Changes:

- Moved bitstring prep/measure program creation helpers to utils (gh-118)
- Added functoinality to `plotting` module: two ways to visualize a quantum state in the Pauli basis, plotting of a Pauli Transfer Matrix, plotting a real matrix using a Hinton diagram, the addition of the computational basis as a predefined basis object (gh-119)
- Refactor iterative MLE to use ExperimentResults directly (gh-120)
- Combined `graph_state` and `bell_state` modules into `entangled_state` module, added deprecation warnings for the old modules (gh-122)
- Made Ipython Notebooks a part of testing (gh-123) 
- Resolve test warnings and doc string formatting issues (gh-124)
- **Breaking change.** Bump version and delete `graph_state` and `bell_state` modules (gh-125)
- Added the ability to check if the Kraus operators are valid (PR 128)


v0.4 (May 6, 2018)
------------------

Improvements and Changes:

- Updated requirements.txt and setup.py files (gh-83)
- Added support for chi superoperator conversion and augmented documentation for
  superoperator representations (gh-76)
- Renamed `superoperator_conversion` to `superoperator_tools` (gh-100) to reflect
  new functionality, including helpers to apply a Choi process to a state, compute
  the Pauli twirl (gh-101), testing properties (e.g. CP, TP) of channels in the Choi
  representation (gh-109), and projecting Choi matrices to nearby processes satisfying
  those properties (gh-114)
- Following update to pyQuil version 2.7.2, changed stddev to std_err and removed qubits
  argument when using pyquil `TomographyExperiment` (gh-107)
- Changed the default input basis from SIC to Pauli for process tomography (gh-105)
- Standardized around use of `pyquil.gate_matrices` for definitions of gate matrices (gh-103)
- Added support non-square Kraus operators to superoperator conversion tools (gh-102)
- Renamed `diamond_norm` to `diamond_norm_distance` gh-104
- Made use of superoperator tools wherever possible throughout the repo (gh-111)
- Standardized around use of dim as variable name for dimension (gh-112)
- Added `entanglement_fidelity` calculation in distance_measures.py (gh-116)

Bugfixes:

- Fixed keyword argument name `symmetrize_readout` in call to pyQuil's
  `measure_observables` (gh-115)

v0.3 (April 5, 2018)
--------------------

Initial public release of `forest-benchmarking`.
