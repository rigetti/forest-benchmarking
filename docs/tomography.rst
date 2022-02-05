Tomography
==========

Tomography involves making many projective measurements of a quantum state or process and using
them to reconstruct the initial-state or process.

Running State Tomography
------------------------

Prepare the experiments
~~~~~~~~~~~~~~~~~~~~~~~

We wish to perform state tomography on a graph state on qubits 0-1 on a noisy 9q-square-qvm

.. code:: ipython3

    from pyquil import Program, get_qc
    from pyquil.gates import *
    qubits = [0, 1]
    qc = get_qc("9q-square-qvm", noisy=True)

    state_prep = Program([H(q) for q in qubits])
    state_prep.inst(CZ(0,1))


The state prep program is thus::

    H 0
    H 1
    CZ 0 1


We generate the required experiments::

    from forest.benchmarking.tomography import *
    exp_desc = generate_state_tomography_experiment(state_prep, qubits)

which in this case are measurements of the following operators::

    ['(1+0j)*X0',
     '(1+0j)*Y0',
     '(1+0j)*Z0',
     '(1+0j)*X1',
     '(1+0j)*X1*X0',
     '(1+0j)*X1*Y0',
     '(1+0j)*X1*Z0',
     '(1+0j)*Y1',
     '(1+0j)*Y1*X0',
     '(1+0j)*Y1*Y0',
     '(1+0j)*Y1*Z0',
     '(1+0j)*Z1',
     '(1+0j)*Z1*X0',
     '(1+0j)*Z1*Y0',
     '(1+0j)*Z1*Z0']

Data Acquisition
~~~~~~~~~~~~~~~~

We can then collect data::

    from forest.benchmarking.observable_estimation import estimate_observables
    results = list(estimate_observables(qc, exp_desc))

Estimate the State
~~~~~~~~~~~~~~~~~~

Finally, we analyze our data with one of the analysis routines::

    rho_est = linear_inv_state_estimate(results, qubits)
    print(np.real_if_close(np.round(rho_est, 3)))

.. parsed-literal::

    [[ 0.263-0.j     0.209-0.014j  0.23 -0.027j -0.203-0.01j ]
    [ 0.209+0.014j  0.231+0.j     0.175+0.j    -0.168-0.019j]
    [ 0.23 +0.027j  0.175-0.j     0.277-0.j    -0.173+0.004j]
    [-0.203+0.01j  -0.168+0.019j -0.173-0.004j  0.229-0.j   ]]

.. module:: forest.benchmarking.tomography

.. toctree::
    examples/tomography_state
    examples/tomography_process

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    do_tomography

State Tomography
----------------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    generate_state_tomography_experiment
    linear_inv_state_estimate
    iterative_mle_state_estimate
    estimate_variance


Process Tomography
------------------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    generate_process_tomography_experiment
    linear_inv_process_estimate
    pgdb_process_estimate
