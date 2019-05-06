Changelog
=========

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
