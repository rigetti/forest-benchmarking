Welcome to the Docs for Forest-Benchmarking
===========================================

Forest-Benchmarking is an **open source** library for performing quantum characterization, verification, and validation
(QCVV) of quantum computers using the Rigetti Forest Software Development Kit.



See our Installation and Getting Started guide :doc:`start`.


**A few terms to orient you as you get started:**

1. **pyQuil**: An `open source Python library  <http://github.com/rigetti/pyquil>`_ to help you write and run quantum programs.
2. **Quil**: The `Quantum Instruction Language <https://arxiv.org/abs/1608.03355>`__. Instructions written in Quil can be executed on any implementation of a quantum abstract machine, such as the quantum virtual machine, or on a real quantum processing unit.
3. **QVM**: The `Quantum Virtual Machine <https://github.com/rigetti/qvm>`__ is an open source implementation of a quantum abstract machine on classical hardware. The QVM lets you use a regular computer to simulate a small quantum computer and execute Quil programs.
4. **QPU**: Quantum processing unit. This refers to the physical hardware chip which we run quantum programs on.
5. **Quil Compiler**: The `open source compiler <https://github.com/rigetti/quilc>`__, ``quilc``, compiles arbitrary Quil programs to the supported lattice and instruction for our architecture. 
6. **Quantum Cloud Services**: `Quantum Cloud Services <http://rigetti.com/qcs>`_ offers users access point to our physical      	quantum computers and the ablity to schedule compute time on our QPUs.

.. note::

    To join our user community, connect to the Rigetti Slack workspace at https://rigetti-forest.slack.com.

Contents
--------


.. toctree::
   :maxdepth: 3
   :caption: QVCC Routines:

   tomography
   dfe
   rb
   rpe


.. toctree::
   :maxdepth: 3
   :caption: Utils

   utils

.. toctree::
	:maxdepth:2
	:caption: Tutorials



