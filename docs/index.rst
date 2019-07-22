Welcome to the Docs for Forest-Benchmarking
===========================================

Forest-Benchmarking is an **open source** library for performing quantum characterization, verification, and validation
(QCVV) of quantum computers using pyQuil.



To get started see

* :ref:`install`
* :ref:`start`


.. note::

    To join our user community, connect to the Rigetti Slack workspace at https://rigetti-forest.slack.com.

Contents
--------

.. toctree::
    :maxdepth: 1

    install
    start


.. toctree::
   :maxdepth: 1
   :caption: Observable Estimation

   obs_est


.. toctree::
   :maxdepth: 1
   :caption: QVCC Routines:

   tomography
   dfe
   rb
   rpe
   readout
   qubit_spec


.. toctree::
   :maxdepth: 1
   :caption: Holistic Benchmarks

   class_logic
   qvol


.. toctree::
   :maxdepth: 1
   :caption: Visualization

   plotting


.. toctree::
   :maxdepth: 1
   :caption: Fitting

   fit


.. toctree::
   :maxdepth: 1
   :caption: Basic Compilation

   compilation

.. toctree::
   :maxdepth: 2
   :caption: Distance Measures

   dist_meas

.. toctree::
   :maxdepth: 1
   :caption: Operator and Superoperator tools

   superoperator_representations
   examples/superoperator_tools
   operator_tools/rand_ops
   operator_tools/app_superop
   operator_tools/calc
   operator_tools/chan_approx
   operator_tools/compose
   operator_tools/project_state
   operator_tools/project_superop
   operator_tools/transform
   operator_tools/validate_op
   operator_tools/validate_super


.. toctree::
   :maxdepth: 1
   :caption: Entangled States

   ent_states

.. toctree::
   :maxdepth: 1
   :caption: Utils

   utils


.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   :glob:

   examples/*