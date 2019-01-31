# Forest-Benchmarking

A library for quantum characterization, verification, validation, and benchmarking using Rigetti's
 SDK -- Forest. 


### Installation

Forest-Benchmarking is a Python package. It is currently in pre-release and must be installed from `master`

    git clone https://github.com/rigetti/forest-benchmarking.git
    cd forest-benchmarking/
    pip install -e .

### Library Philosophy

The core philosophy of Forest-Benchmarking is to separate: 

* Experiment design and or generation
* Data collection
* Data analysis
* Data visualisation

We ask that code contributed to this repository respect this separation. We also ask that an example of how to use your contributed code is placed in the `/examples/` directory along with the standard documentation found in `/docs/`.
