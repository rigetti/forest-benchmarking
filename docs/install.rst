Installation Guide
==================

**A few terms to orient you as you are installing:**

1. **pyQuil**: An `open source Python library  <http://github.com/rigetti/pyquil>`_ to help you write and run quantum programs written in *quil*.
2. **Quil**: The `Quantum Instruction Language <https://arxiv.org/abs/1608.03355>`__. 
3. **QVM**: The `Quantum Virtual Machine <https://github.com/rigetti/qvm>`__ is an open source implementation of a quantum abstract machine on classical hardware. The QVM lets you use a regular computer to simulate a small quantum computer and execute Quil programs.
4. **QPU**: Quantum processing unit. This refers to the physical hardware chip which we run quantum programs on.
5. **Quil Compiler**: The `open source compiler <https://github.com/rigetti/quilc>`__, ``quilc``, compiles arbitrary Quil programs to the supported lattice and instruction for our architecture. 
6. **Quantum Cloud Services**: `Quantum Cloud Services <http://rigetti.com/qcs>`_ offers users access point to our physical quantum computers and the ablity to schedule compute time on our QPUs.  If you’d like to access to our quantum computers for QCVV research, please email support@rigetti.com.



Step 1. Install pyQuil and the Forest SDK
-----------------------------------------
::

	pip install pyquil

Next you must install the SDK which includes the Rigetti Quil Compiler (quilc), and the Quantum
Virtual Machine (qvm). `Request the Forest SDK here <http://rigetti.com/forest>`__. You'll
receive an email right away with the download links for macOS, Linux (.deb), Linux (.rpm), and Linux (bare-bones).

If you dont already have Jupyter or Jupyter lab now would be a good time to install that too.

::

    pip install jupyterlab


Step 2. Install forest-benchmarking
-----------------------------------
`forest-benchmarking` can be installed from source or via the Python package manager PyPI.

**Note**: NumPy and SciPy must be pre-installed for installation to be successful, due to cvxpy.

**Source**

::

	git clone https://github.com/rigetti/forest-benchmarking.git
	cd forest-benchmarking/
	pip install numpy scipy
	pip install -e .


**PyPI**

::

		pip install numpy scipy
		pip install forest-benchmarking




Step 3. Build the docs (optional)
---------------------------------
We use sphinx to build the documentation. To do this, first  install the requirements

::
	
	pip install sphinx
	pip install sphinx_rtd_theme
	pip install nbsphinx
	
then navigate into Forest-Benchmarkings top-level directory and run:

::

		sphinx-build -b html docs/ docs/_build

To view the docs navigate to the newly-created docs/_build directory and open the index.html file in a browser.
