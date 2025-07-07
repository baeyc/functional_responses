# Modelling functional response using mixed-effects models

This code performs parameter estimation of a functional response model using stochastic gradient descent algorithms.

## Dependencies

All modules are available on PyPI:
 - `jax`
 - `numpy`
 - `scipy`
 - `parametrization_cookbook`
 - `matplotlib`
 - `tqdm`
 - `functools`
 
 ## Run examples

A jupyter notebook `ex_run_algo.ipynb` gives some examples of how to run the model and get predictions, or load real data.

File `models.py` contains the model specification, along with all the functions that directly depend on the model such as the log-likelihood and its derivatives but also the MCMC sampler.

File `algos.py` contains the SGD algorithm.

File `config.py` contains the configuration of the mixed-effects model, i.e. the names of the random effects as well as their covariance structure.

The other files run and/or analyze the results from the model:
 - `several_runs_sample_size.py` and `several_runs_misspe.py` run several repetitions of the SGD algorithm on sets of simulated data, for different sampling size and for a misspecified model.
  - `compute_rmse_sample_size.py` and `compute_rmse_misspe.py` compute the RMSE of the parameter estimates, and draw plots of the results
  - `real_data_analysis.py` perform analysis of Schroder et al. data on **Artemia** and **Heterandria**
  
In order to reproduce the data of the paper Baey, Billiard, Delattre (2025+), files `several_runs_sample_size.py` and `compute_rmse_sample_size.py` allows to obtain Figures 5 and 6, and `real_data_analysis.py` allows to obtain all the results (Tables and Figures) of section 3.3 
