Quick start guide
=================

Below we will assume that you are developing in a jupyter notebook.


Getting ready to benchmark (QVM)
--------------------------------

First thing you need to do is open up a terminal and run:

.. code-block:: bash

	$ quilc -S
	$ qvm -S
	$ juputer lab

Inside the notebook we need to get some basic pyQuil object

.. todo:: very minimal examples of importing get_qc, get_benchmarker, and run the major benchmarks (tomo, dfe, rb, t1)

.. code:: python

	from pyquil import get_qc
	from pyquil.api import get_benchmarker
	from forest.benchmarking.compilation import basic_compile
	qc.compiler.quil_to_native_quil = basic_compile



Getting ready to benchmark (QPU)
--------------------------------
.. todo:: QMI then re log into qcs and document getting forest benchmarking working

1. log into qcs
2. ?
