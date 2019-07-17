.. currentmodule:: forest.benchmarking.utils

Utilities
=========

In ``utils.py`` you will find functions that are shared among one or more modules.

Common Functions
----------------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    is_pos_pow_two
    bit_array_to_int
    int_to_bit_array
    pack_shot_data
    bloch_vector_to_standard_basis
    standard_basis_to_bloch_vector
    prepare_state_on_bloch_sphere
    transform_pauli_moments_to_bit
    transform_bit_moments_to_pauli
    parameterized_bitstring_prep
    bitstring_prep
    metadata_save


Pauli Functions
----------------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    str_to_pauli_term
    all_traceless_pauli_terms
    all_traceless_pauli_choice_terms
    all_traceless_pauli_z_terms
    local_pauli_eig_prep
    local_pauli_eigs_prep
    random_local_pauli_eig_prep
    local_pauli_eig_meas
    prepare_prod_pauli_eigenstate
    measure_prod_pauli_eigenstate
    prepare_random_prod_pauli_eigenstate
    prepare_all_prod_pauli_eigenstates


Operator Basis
---------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    OperatorBasis
    n_qubit_pauli_basis
    n_qubit_computational_basis
