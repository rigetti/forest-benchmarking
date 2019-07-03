.. currentmodule:: forest.benchmarking.robust_phase_estimation

Robust Phase Estimation
=======================

Is a kind of `iterative phase estimation <https://arxiv.org/abs/0904.3426>`_ formalized by
`Kimmel, Low, Yoder Phys. Rev. A 92, 062315 (2015) <https://arxiv.org/abs/1502.02677>`_. It is
ideal for measuring gate calibration errors.


API Reference
-------------

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    prepare_state
    generate_single_depth_experiment
    generate_2q_single_depth_experiment
    generate_rpe_experiments
    get_additive_error_factor
    num_trials
    acquire_rpe_data

.. rubric:: Analysis
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    p_max
    xci
    get_variance_upper_bound
    robust_phase_estimate
    plot_RPE_iterations


