Changelog
=========

[next](https://github.com/rigetti/forest-benchmarking/compare/v0.7.1...master) (in development)
------------------------------------------------------------------------------------

### Announcements

### Bugfixes

v0.8.0 (February 4, 2022)
------------------------------------------------------------------------------------

### Announcements

- Upgraded pyQuil to v3 (gh-215). As a result, the minimum supported Python version is now 3.7.

### Bugfixes

- Fix compiler timeout during test (@jlapeyre) (gh-208)
- Fix type error in pauli coefficient in observable calibration for recent pyquil versions (@kylegulshen) (gh-207)
- Add noise definition only once to quil program (@kylegulshen) (gh-206)


v0.7.1 (November 25, 2019)
--------------------------

Improvements and Changes:

- Add PyPI version and Slack badges (gh-196).
- Add Zenodo badge and BibTeX file for citation (gh-197).
- Accommodate XY gate in `basic_compile` (gh-202). 
- Increase PyQuil version requirement to accommodate XY gate (gh-203).


v0.7 (September 20, 2019)
----------------------
Breaking Changes:

- Major module re-org of superoperator tools into `operator_tools` also moved `random_operators` to the operator tools module. Added type checking in random operators, added new module to check plain old operators are unitary etc (gh-150, 140, 164).
- Remove symmetrization functionality from `observable_estimation` in favor of pyquil functionality (gh-194). 
- Methods in `fitting` renamed to be less ambiguous `decay_constant_param_decay` -> `decay_time_param_decay` and
`fit_decay_constant_param_decay` -> `fit_decay_time_param_decay`. Correspondingly, the fit parameter was renamed
`decay_constant` -> `decay_time`(gh-175)
- `generate_cz_phase_ramsey_experiment` was made plural, consistent with rest of `qubit_spectroscopy` (gh-175)
- `acquire_cz_phase_ramsey_data` removed in favor of `estimate_observables` and all other specific `acquire_*` methods
in `qubit_spectroscopy.py` were removed in favor of `acquire_qubit_spectroscopy_data` (gh-175)
- argument order standardized, which changed the api of `generate_exhaustive_process_dfe_experiment`,
`generate_exhaustive_state_dfe_experiment`, `generate_monte_carlo_state_dfe_experiment`, 
`generate_monte_carlo_process_dfe_experiment`, `robust_phase_estimate`, 
and positional arg name of `acquire_dfe_data` (gh-182) 

Improvements and Changes:

- Fixed the years in this Change log file 2018 -> 2019
- Added linear inversion process tomography (gh-142)
- Changed qubit tensor factor ordering of state tomography estimates to match that of process tomography, e.g. 
tomographizing the plus eigenstate of `X0 * Z1` and passing in `qubits = [0,1]` will yield the state 
estimate corresponding to `|+0> = (1, 0, 1, 0)/sqrt(2)` rather than `|0+>` (gh-142)
- Improved the `superoperator_tools` notebook and the `random_operators` notebook (gh-98)
- Improvements to Ripple carry adder notebook, added tests for non parametric bit string 
prep program in utils (gh-98)
- Added the ability to project a Choi matrix to the closest unitary (gh-159, 157)
- Reduced local test run time from 11min to 5min (gh-160)
- Major changes and improvements to all notebooks (gh-148, 149, 153, 154, 155, 156, 165, 167, 172, 182(see 183))
- Speedup the tests (gh-158, 161)
- Complete overhaul/addition of documentation in docs folder (gh-170, 174)
- Github PR and bug report templates added (gh-177, 182)
- `merge_disjoint_experiments` and `get_results_by_qubit_groups` added to `observable_estimation.py` to facilitate
running experiments 'in parallel' on a QPU and analyzing the results separately (gh-182)
- various `do_*` methods added that wrap various experiments into a single method with sensible defaults (gh-182)
- examples notebooks moved to docs/ directory to be rendered within documentation (gh-182)
- tqdm progress bars added to most data acquisition methods (gh-182)

Bugfixes:

- Dagger is now implemented in pyquil as a gate-level modifier, but this doesn't play well with noise models on the QVM
so a stop-gap was put into basic_compile to hand compile the dagger modifier (gh-169)
- Compiler sometimes failed to find connected subgraphs of qubits (gh-188)
- QPU cannot handle integer parameters (gh-188)

v0.6 (June 11, 2019)
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

v0.5 (June 10, 2019)
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


v0.4 (May 6, 2019)
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

v0.3 (April 5, 2019)
--------------------

Initial public release of `forest-benchmarking`.
