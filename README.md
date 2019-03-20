# Forest Benchmarking

[![pipeline status](https://gitlab.com/rigetti/forest-benchmarking/badges/master/pipeline.svg)](https://gitlab.com/rigetti/forest-benchmarking/commits/master)[![Build Status](https://semaphoreci.com/api/v1/rigetti/forest-benchmarking/branches/master/shields_badge.svg)](https://semaphoreci.com/rigetti/forest-benchmarking)

A library for quantum characterization, verification, validation (QCVV), and benchmarking using [pyQuil](https://github.com/rigetti/pyquil).

## Installation

`forest-benchmarking` is a Python package.
It is currently in pre-release and must be installed from `master`

    git clone https://github.com/rigetti/forest-benchmarking.git
    cd forest-benchmarking/
    pip install -e .

## Library Philosophy

The core philosophy of `forest-benchmarking` is to separate:

* Experiment design and or generation
* Data collection
* Data analysis
* Data visualisation

We ask that code contributed to this repository respect this separation.
We also ask that an example of how to use your contributed code is placed
in the `/examples/` directory along with the standard documentation found in `/docs/`.

## Testing

The unit tests can be run locally using `pytest`, but beware that the test dependencies
must be installed beforehand using `pip install -r requirements.txt`.
