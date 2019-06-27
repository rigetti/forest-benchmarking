Installation 
============

Prerequisites
-------------

**Step 1.** install the pyQuil and the Forest SDK which include the Rigetti Quil Compiler (quilc), and the Quantum Virtual Machine (qvm).

::

    pip install pyquil


`Request the Forest SDK here <http://rigetti.com/forest>`__. You'll receive an email right away with the download links for macOS, Linux (.deb), Linux (.rpm), and Linux (bare-bones).




**Step 2.** Install forest-benchmarking.

## Installation

`forest-benchmarking` can be installed from source or via the Python package manager PyPI.

**Note**: NumPy and SciPy must be pre-installed for installation to be successful, due to cvxpy.

### Source

```bash
git clone https://github.com/rigetti/forest-benchmarking.git
cd forest-benchmarking/
pip install numpy scipy
pip install -e .
```

### PyPI

```bash
pip install numpy scipy
pip install forest-benchmarking
```



## Build the docs

We use sphinx to build the documentation. To do this, first  install the requirements

```bash
pip install sphinx
pip install sphinx_rtd_theme
pip install nbsphinx
pip install recommonmark
```
then navigate into Forest-Benchmarkings top-level directory and run:

```bash
sphinx-build -b html docs/ docs/_build
```
To view the docs navigate to the newly-created docs/_build directory and open the index.html file in a browser.



