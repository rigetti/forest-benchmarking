# Contributor Guidelines

We value and encourage contribution from the community. To reduce
friction in this process, we have collected some best-practices for
contributors:

* API style. We have chosen to split the API of the modules in our 
  benchmarking package in the following way:
  `exp_obj = generate_X_experiment( parameters)`
  `results_obj = acquire_data_X(exp_obj)`
  `estimates = estimate_X(results_object)`
  `plot_X(estimates, results)`.
  We have chosen this split so, for example, so that for the same 
  data set (`results_obj`) many different estimation or analysis 
  methods can be easily applied.

* Testing. Before making a pull-request (PR), please make sure that
  you have tested your new QCVV code (and any other modules you have 
  changed) for mathematical correctness by including important test 
  cases in `/tests/` and then running the tests. We use the pytest 
  framework.
  
* Documentation. Public functions *must* have docstrings. Documentation 
  should also be provided as comments to your code. Finally please include 
  references to to the relavent papers and equations in those papers.
  The recomended referernce style guide is:
  
  [TopX]  Cool title about topic X
          first author lastname, journal name, volume, firstpage (year)
          DOI Link
          arXiv Link
  e.g.
  
  [HMLE]  Hedged Maximum Likelihood Quantum State Estimation
          Blume-Kohout, PRL, 105, 200504 (2010)
          https://doi.org/10.1103/PhysRevLett.105.200504
          https://arxiv.org/abs/1001.2029
 
 * Notebooks. At the moment we are clearing the notebook metadata.
   A recomended workflow is to install `jq` (https://stedolan.github.io/jq/),
   then put this script in your home directory [clean-notebook.sh](https://gist.github.com/mpharrigan/14f3c93be5520d139ea265ad95249663#file-clean-notebook-sh).
   The workflow is then to run the command `~/clean_notebook.sh your_notebook.ipynb` before commiting.

* Code style. In general, follow the [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/).
