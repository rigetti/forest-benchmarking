.. _start:

Quick Start Guide
=================

Below we will assume that you are developing in a jupyter notebook.


Getting ready to benchmark (QVM)
--------------------------------

First thing you need to do is open up a terminal and run:

.. code-block:: bash

    $ quilc -S
    $ qvm -S
    $ jupyter lab

Inside the notebook we need to get some basic pyQuil objects, namely a `QuantumComputer`, as well as a `BenchmarkConnection` for some routines.
We'll also need to be able to construct a pyQuil `Program`.

.. code:: python

    from pyquil import get_qc
    from pyquil.api import get_benchmarker

    noisy_qc = get_qc('2q-qvm', noisy=True)
    bm = get_benchmarker()

    from pyquil import Program
    from pyquil.gates import *

Now we are ready to run through some simple examples. We'll start with state tomography on the plus state.

.. code:: python

    from forest.benchmarking.tomography import do_tomography

    # prepare the plus state
    qubit = 1
    state_prep = Program(H(qubit))

    # tomograph the noisy plus state
    state_estimate, _, _ = do_tomography(noisy_qc, state_prep, qubits=[qubit], kind='state')

Process tomography is quite similar (note that this will take a long time with the default arguments).

.. code:: python

    # specify a process
    qubits = [0, 1]
    process = Program(CNOT(*qubits))
    # tomograph the noisy process CNOT
    process_estimate, _, _ = do_tomography(noisy_qc, process, qubits, kind='process')

If we only care about the fidelity of our state or process then we can turn to Direct Fidelity Estimation (DFE) to
save time / runs on the quantum computer. Here we use the `BenchmarkConnection` `bm` to do some of the operations with
the Clifford group.

.. code:: python

    from forest.benchmarking.direct_fidelity_estimation import do_dfe

    # fidelity of a state preparation
    (fidelity_est, std_err), _, _ = do_dfe(noisy_qc, bm, state_prep, qubits=[qubit], kind='state')

    # process fidelity
    (proc_fidelity_est, std_err), _, _ = do_dfe(noisy_qc, bm, process, qubits, kind='process')


Finally we can get estimates of the average error rate for our Clifford gates using Randomized Benchmarking (RB).
Here again we use `bm` to generate random sequences of Clifford gates (compiled to native gates).

.. code:: python

    from forest.benchmarking.randomized_benchmarking import do_rb

    # simultaneous single qubit RB on q0 and q1
    qubit_groups = [(0,), (1,)]
    num_sequences_per_depth = 20
    depths = [2, 10, 20] * num_sequences_per_depth
    rb_decays, _, _ = do_rb(noisy_qc, bm, qubit_groups, depths)

These are just a few examples! Peruse the examples notebooks to see more.

Getting ready to benchmark (QPU)
--------------------------------
.. todo:: QMI then re log into qcs and document getting forest benchmarking working

1. log into qcs
2. ?
