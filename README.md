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
 - `pandas`
 - `seaborn`
 - `collections`
 - `os`
 - `re`
 - `pickle`

 
 ## Run examples

A jupyter notebook `ex_run_algo.ipynb` gives some examples of how to run the model and get predictions, or load real data.

File `models.py` contains the model specification, along with all the functions that directly depend on the model such as the log-likelihood and its derivatives but also the MCMC sampler.

File `algos.py` contains the SGD algorithm.

File `config.py` contains the configuration of the mixed-effects model, i.e. the names of the random effects as well as their covariance structure.

**In most of the above files, the path to the directory containing the results should be provided at the beginning of the script (there is a default value on the first few lines that should be replaced by the appropriate folder name.**

The other files run and/or analyze the results from the model:
  - `several_runs_sample_size.py` and `several_runs_misspe.py` run several repetitions of the SGD algorithm on sets of simulated data, for different sampling size and for a misspecified model.
  - `compute_rmse_sample_size.py` and `compute_rmse_misspe.py` compute the RMSE of the parameter estimates, and draw plots of the results
  - `several_runs_variability_level.py` runs several repetitions of the SGD algorithm on sets of simulated data, for different levels of variability in the data
  - `model_choice_variability_level.py` performs model choice on sets of simulated data, for different levels of variability in the data
  - `compute_results_variability_level.py` plots graphs of parameter estimates and generates tables of model selection results for different levels of variability in the data
  - `real_data_analysis.py` perform analysis of Schroder et al. data on **Artemia** and **Heterandria**.

The real dataset is contained in the file `schroder_et_al.csv`, which was obtained from the authors of the study. It corresponds to raw data on the number of preys ingested per unit of time (no transformation of the data was necessary).
  
In order to reproduce the data of the paper Baey, Billiard, Delattre (2025+):
  - files `several_runs_sample_size.py` and `compute_rmse_sample_size.py` allows to obtain Figures 5 and 6, 
  - files `several_runs_variability_level.py`, `model_choice_variability_level.py`, and `compute_results_variability_level.py` allow to obtain Figures 7, 8, 9, 10 and Tables 1 and 2,
  - `real_data_analysis.py` allows to obtain all the results (Tables and Figures) of section 3.3.
**Note that these scripts can be computationnally heavy and may take several hours to run. The computationnal time can be reduced by decreasing the number of repetitions Nsimus.**
