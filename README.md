Forest Benchmarking: QCVV using pyQuil
======================================

[![pipeline status](https://gitlab.com/rigetti/forest-benchmarking/badges/master/pipeline.svg)](https://gitlab.com/rigetti/forest-benchmarking/commits/master)
[![Build Status](https://semaphoreci.com/api/v1/rigetti/forest-benchmarking/branches/master/shields_badge.svg)](https://semaphoreci.com/rigetti/forest-benchmarking)
[![Documentation Status](https://readthedocs.org/projects/forest-benchmarking/badge/?version=latest)](https://forest-benchmarking.readthedocs.io/en/latest/?badge=latest)
[![pypi version](https://img.shields.io/pypi/v/forest-benchmarking)](https://pypi.org/project/forest-benchmarking/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3455847.svg)](https://doi.org/10.5281/zenodo.3455847)
[![slack workspace](https://img.shields.io/badge/slack-rigetti--forest-812f82.svg?)](https://join.slack.com/t/rigetti-forest/shared_invite/enQtNTUyNTE1ODg3MzE2LWExZWU5OTE4YTJhMmE2NGNjMThjOTM1MjlkYTA5ZmUxNTJlOTVmMWE0YjA3Y2M2YmQzNTZhNTBlMTYyODRjMzA)

A library for quantum characterization, verification, validation (QCVV), and benchmarking using [pyQuil](https://github.com/rigetti/pyquil).

Installation
------------

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

Library Philosophy
------------------

The core philosophy of `forest-benchmarking` is to separate:

* Experiment design and or generation
* Data collection
* Data analysis
* Data visualisation

We ask that code contributed to this repository respect this separation.
We also ask that an example of how to use your contributed code is placed
in the `/examples/` directory along with the standard documentation found in `/docs/`.

Testing
-------

The unit tests can be run locally using `pytest`, but beware that the test dependencies
must be installed beforehand using `pip install -r requirements.txt`.

Disclaimer
----------

This package is currently in alpha (v0.x), and therefore you should not expect that APIs
will necessarily be stable between releases. Code that depends on this package in its current
state is very likely to break when the package version changes, so we encourage you to pin
the version you use, and update it consciously when necessary.

Citation
--------

If you use Forest Benchmarking, please cite it via the [BibTeX file](forest-benchmarking.bib).
