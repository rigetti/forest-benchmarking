Direct Fidelity Estimation
==========================

Direct fidelity estimation uses partial tomography to quickly certify a quantum state
or a quantum process at lower cost than full tomography.

Running DFE
-----------

.. todo:: Put an overview or mini tutorial here.


State DFE
---------
.. currentmodule:: forest.benchmarking.direct_fidelity_estimation
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    generate_exhaustive_state_dfe_experiment
    generate_monte_carlo_state_dfe_experiment
    acquire_dfe_data
    estimate_dfe


Process DFE
-----------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    generate_exhaustive_process_dfe_experiment
    generate_monte_carlo_process_dfe_experiment
    acquire_dfe_data
    estimate_dfe
