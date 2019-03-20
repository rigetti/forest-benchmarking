Tomography
==========

Tomography involves making many projective measurements of a quantum state or process and using
them to reconstruct the initial full state or process.

Running State Tomography
------------------------

Prepare the experiments
~~~~~~~~~~~~~~~~~~~~~~~

We wish to perform state tomography on a graph state on qubits 0-1 on the 9q-generic-noisy-qvm

.. code:: ipython3

    from pyquil import Program, get_qc
    from pyquil.gates import *
    qubits = [0, 1]
    qc = get_qc("9q-generic-noisy-qvm")

    state_prep = Program([H(q) for q in qubits])
    state_prep.inst(CZ(0,1))


The state prep program is thus::

    H 0
    H 1
    CZ 0 1


We generate the required experiments::

    from forest.benchmarking.state_tomography import *
    exp_desc = generate_state_tomography_experiment(state_prep)

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

    exp_data = acquire_state_tomography_data(exp_desc, qc, var=0.001)

Linear inversion estimate
~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, we analyze our data with one of the analysis routines::

    rho_est = linear_inv_state_estimate(exp_data).state_point_est
    print(np.round(rho_est, 4))
    print('Purity:', np.trace(np.matmul(rho_est, rho_est)))

.. parsed-literal::

    [[ 0.2754+0.j      0.2077+0.0136j  0.2047+0.0153j -0.1869+0.0077j]
     [ 0.2077-0.0136j  0.2551+0.j      0.1919+0.0059j -0.1794+0.0188j]
     [ 0.2047-0.0153j  0.1919-0.0059j  0.2493+0.j     -0.169 +0.0169j]
     [-0.1869-0.0077j -0.1794-0.0188j -0.169 -0.0169j  0.2202-0.j    ]]
    Purity =  (0.6889520199999999+4.597017211338539e-17j)


API Reference
-------------

.. currentmodule:: forest.benchmarking.state_tomography
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    generate_state_tomography_experiment
    state_tomography_experiment_data
    acquire_state_tomography_data
    state_tomography_estimate
    linear_inv_state_estimate
    construct_pinv_measurement_matrix
    construct_projection_operators_on_n_qubits
    iterative_mle_state
    project_density_matrix
    estimate_variance

