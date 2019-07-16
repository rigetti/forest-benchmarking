# Contributor Guidelines

We value and encourage contribution from the community. To reduce
friction in this process, we have collected some best-practices for
contributors:

* API style. We have chosen to split the API of the modules in our 
  benchmarking package in the following way: \
  `exp_obj = generate_X_experiment(parameters)` \
  `results_obj = acquire_X_data(exp_obj)` \
  `estimates = estimate_X(results_object)` \
  `plot_X(estimates, results)` \
  We have chosen this split so that, for example, many different 
  estimation or analysis methods can be easily applied to the same 
  data set (`results_obj`).

* Testing. Before making a pull-request (PR), please make sure that
  you have tested your new QCVV code (and any other modules you have 
  changed) for mathematical correctness by including important test 
  cases in `/tests/` and then running the tests. We use the pytest 
  framework.
  
* Documentation. Public functions *must* have docstrings. Documentation 
  should also be provided as comments to your code. Finally please include 
  references to to the relevant papers and equations in those papers.
  The recommended reference style guide is:
  
      .. [TOPX] Cool title about topic X.
           first author lastname, journal name, volume, firstpage (year).
           DOI Link 
           arXiv Link 
          
  e.g.
  
      .. [HMLE] Hedged Maximum Likelihood Quantum State Estimation.
           Blume-Kohout, PRL, 105, 200504 (2010).
           https://doi.org/10.1103/PhysRevLett.105.200504
           https://arxiv.org/abs/1001.2029 
  
  To refer to this reference elsewhere use the tag `[TOPX]_` with underscore, g.e. `[HMLE]_` 
 
 * Notebooks. Please only commit substantive changes to the examples notebooks, since these are used in documentation. 
 If creating or contributing to a notebook please be mindful of the markdown formatting. Try to create and maintain 
 useful headings, e.g.
    
        # Notebook Title
        ## Section heading
        ### Subsection heading
        ## Section heading
        
 and be careful to exclude white space in the start or end of Latex expressions, e.g.
 
       In line latex with no whitespace bookends $x=2$
       
       $$\textrm{longer expression, no whitespace bookends }$$
       
       Not this: $ x=2 $

* Code style. In general, follow the [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/).
