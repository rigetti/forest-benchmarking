.. currentmodule:: forest.benchmarking.operator_tools.validate_superoperator

Validate Superoperators
=======================

The module ``validate_superoperator`` lets you check properties, such as physicality, of
channels. If you have a superoperator specified in a different representation convert it to the
representations below.

Kraus operators
---------------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    kraus_operators_are_valid


Choi Matrix
-----------
.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    choi_is_hermitian_preserving
    choi_is_trace_preserving
    choi_is_completely_positive
    choi_is_cptp
    choi_is_unital
    choi_is_unitary





