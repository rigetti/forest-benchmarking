.. module:: forest.benchmarking.direct_fidelity_estimation
Direct Fidelity Estimation
==========================

Direct fidelity estimation (DFE) uses knowledge about the ideal target state or process to perform
a tailored set of measurements to quickly certify a quantum state or a quantum process at lower
cost than full tomography.

.. toctree::

    dfe_example

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    do_dfe

State DFE
---------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    generate_exhaustive_state_dfe_experiment
    generate_monte_carlo_state_dfe_experiment


Process DFE
-----------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    generate_exhaustive_process_dfe_experiment
    generate_monte_carlo_process_dfe_experiment


Data Acquisition
----------------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    acquire_dfe_data


Analysis
--------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    estimate_dfe