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

.. code:: python

	from pyquil import get_qc
	from pyquil.api import get_benchmarker
	from forest.benchmarking.compilation import basic_compile
	qc.compiler.quil_to_native_quil = basic_compile

qvm object
quilc
basic compiler
get qc

Getting ready to benchmark (QPU)
--------------------------------
1. log into qcs
2. ?
3. ? 