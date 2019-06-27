Direct Fidelity Estimation
==========================

Direct fidelity estimation uses partial tomography to quickly certify a quantum state
or a quantum process at lower cost than full tomography.

Running DFE
-----------

.. todo:: Put a mini tutorial here.


Data structures
---------------

.. currentmodule:: forest.benchmarking.direct_fidelity_estimation
.. autoclass:: dfe_experiment
.. autoclass:: dfe_experiment_data
.. autoclass:: dfe_calibration_data


Functions
---------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    calibrate_readout_imperfections
    generate_state_dfe_experiment
    generate_process_dfe_experiment
    acquire_dfe_data
    direct_fidelity_estimate
    ratio_variance

