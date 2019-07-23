.. module:: forest.benchmarking.compilation

Basic Compilation
=================

Rigetti's native compiler ``quilc`` is a highly advanced and world class compiler. It
has many features to optimize the performance of quantum  algorithms.

In QCVV we need to be certain that the circuit we wish to run is the one that is run. For this
reason we have built the module ``compilation``.  Its functionality is rudimentary but easy to
understand.

Basic Compile
-------------

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    basic_compile

Helper Functions
----------------

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    match_global_phase
    is_magic_angle


Gates
-----

.. autosummary::
    :toctree: autogen
    :template: autosumm.rst

    _RY
    _RX
    _X
    _H
    _T
    _CNOT
    _SWAP
    _CCNOT
