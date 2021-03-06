.. module:: forest.benchmarking.qubit_spectroscopy

Spectroscopic and Analog Measurements of Qubits
===============================================
The protocols in the module ``qubit_spectroscopy`` are closer to analog protocols than gate based
QCVV protocols.

.. toctree::

    examples/qubit_spectroscopy_t1
    examples/qubit_spectroscopy_t2
    examples/qubit_spectroscopy_rabi
    examples/qubit_spectroscopy_cz_ramsey

General Functions
-----------------

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    acquire_qubit_spectroscopy_data
    get_stats_by_qubit
    do_t1_or_t2

T1
-----------------

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    generate_t1_experiments
    fit_t1_results

T2
-----------------

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    generate_t2_star_experiments
    generate_t2_echo_experiments
    fit_t2_results

Rabi
----

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    generate_rabi_experiments
    fit_rabi_results

CZ phase Ramsey
---------------

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    generate_cz_phase_ramsey_experiments
    fit_cz_phase_ramsey_results