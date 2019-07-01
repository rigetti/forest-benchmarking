Observable Estimation and Error Mitigation
==========================================

The module ``observable_estimation`` is at the heart of forest benchmarking. It provides a
convenient way to construct experiments that measure observables and mitigate errors
associated with readout (measurement) process.

Overview
---------

.. todo:: Put an overview or a mini tutorial here.


Data structures
---------------
.. currentmodule:: forest.benchmarking.observable_estimation
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    TensorProductState
    ExperimentSetting
    estimate_observables
    ObservablesExperiment
    ExperimentResult


Functions
---------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    generate_experiment_programs
    group_settings
    group_settings_clique_removal
    estimate_observables
    calibrate_observable_estimates
    ratio_variance

