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

> [!WARNING]
> In most of the above files, the path to the directory containing the results should be provided at the beginning of the script (there is a default value on the first few lines that should be replaced by the appropriate folder name.

> [!CAUTION]
> In the current version of the code, when one wishes to change the model to take into account different rand effects structures, it is necessary to modify the files `config.py` and `models.py`. While this is not optimal, it is the simplest way to optimize compatibility with the JAX library. We are working on more sophisticated solutions.


The other files run and/or analyze the results from the model. They should be run in a specific order. We detailed below several sequences of scripts.

### Effect of the sample size 
  1. run either `several_runs_sample_size.py` or `several_runs_misspe.py`. This will generate several sets of simulated data and run the SGD algorithm on each simulated dataset, and store all the results files in a folder called `results`. The first script repeats this scenario (generation of data followed by estimation via the SGD algorithm) for several sample sizes, while the second script repeats this scenario for different sample sizes in the case where the model is misspecified, i.e. the wrong noise structure is assumed when fitting the data. The number of simulated datasets can be tuned by changing parameter `Nsimus` at the beginning of each files.
  2. once the previous scripts have been run, run `compute_rmse_sample_size.py` or `compute_rmse_misspe.py` to compute the RMSE of the parameter estimates, and draw plots of the results. These scripts will load the results from the folder `results`.

### Effect of the level of variability
  1. run the script `several_runs_variability_level.py` to generate several simulated datasets with different levels of variability, and run the SGD algorithm on each of these datasets. Results are stored in the folder `results`.
  2. run `model_choice_variability_level.py` to generate several simulated datasets and fit a misspecified model (it is similar to the script `several_runs_misspe.py` except that several levels of variability are explored). Results files are stored in the folder `results`
  3. run the script `compute_results_variability_level.py` to plots graphs of parameter estimates and generates tables of model selection results for different levels of variability in the data. This script needs to load results files stored in the folder `results`.

### Real data analysis
Run the script `real_data_analysis.py` to perform analysis of Schroder et al. data on **Artemia** and **Heterandria**. This script will load the .csv file `schroder_et_al.csv`, which contains the real dataset. These data were obtained from the authors of the study and correspond to raw data on the number of preys ingested per unit of time (no transformation of the data was necessary).


## Reproductibility of the results
  
In order to reproduce the data of the paper Baey, Billiard, Delattre (2025+):
  - files `several_runs_sample_size.py` and `compute_rmse_sample_size.py` allows to obtain Figures 5 and 6, 
  - files `several_runs_variability_level.py`, `model_choice_variability_level.py`, and `compute_results_variability_level.py` allow to obtain Figures 7, 8, 9, 10 and Tables 1 and 2,
  - `real_data_analysis.py` allows to obtain all the results (Tables and Figures) of section 3.3.
**Note that these scripts can be computationnally heavy and may take several hours to run. The computationnal time can be reduced by decreasing the number of repetitions Nsimus.**
