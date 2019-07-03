
.. currentmodule:: forest.benchmarking.operator_tools.superoperator_transformations

Superoperator Transformations
=============================

``superoperator_transformations`` is module containing tools for converting between different
representations of superoperators.

We have arbitrarily decided to use a column stacking convention.


vec and unvec
---------------

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    vec
    unvec

Computational and Pauli Basis
-----------------------------

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    pauli2computational_basis_matrix
    computational2pauli_basis_matrix


Transformations from Kraus
--------------------------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    kraus2chi
    kraus2superop
    kraus2pauli_liouville
    kraus2choi


Transformations from Chi
--------------------------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    chi2kraus
    chi2superop
    chi2pauli_liouville
    chi2choi


Transformations from Liouville
------------------------------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    superop2kraus
    superop2chi
    superop2pauli_liouville
    superop2choi


Transformations from Pauli-Liouville (Pauli Transfer Matrix)
------------------------------------------------------------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    pauli_liouville2kraus
    pauli_liouville2chi
    pauli_liouville2superop
    pauli_liouville2choi


Transformations from Choi
-------------------------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    choi2kraus
    choi2chi
    choi2superop
    choi2pauli_liouville

