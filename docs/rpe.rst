.. module:: forest.benchmarking.robust_phase_estimation

Robust Phase Estimation
=======================

Is a kind of `iterative phase estimation <https://arxiv.org/abs/0904.3426>`_ formalized by
`Kimmel, Low, Yoder Phys. Rev. A 92, 062315 (2015) <https://arxiv.org/abs/1502.02677>`_. It is
ideal for measuring gate calibration errors.

.. toctree::

    examples/robust_phase_estimation

API Reference
-------------

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    do_rpe
    bloch_rotation_to_eigenvectors
    get_change_of_basis_from_eigvecs
    change_of_basis_matrix_to_quil
    generate_rpe_experiments
    get_additive_error_factor
    num_trials
    acquire_rpe_data

.. rubric:: Analysis
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    _p_max
    _xci
    get_variance_upper_bound
    estimate_phase_from_moments
    robust_phase_estimate
    plot_rpe_iterations


