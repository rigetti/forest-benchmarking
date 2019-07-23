.. module:: forest.benchmarking.readout

Readout Error Estimation
========================

The ``readout.py`` module allows you to estimate the measurement confusion matrix.


.. toctree::

    examples/readout_error_estimation


Functions
---------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    get_flipped_program
    estimate_confusion_matrix
    estimate_joint_confusion_in_set
    estimate_joint_reset_confusion
    marginalize_confusion_matrix

