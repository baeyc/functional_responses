import collections
import models
import jax.numpy as jnp
import jax
import algos
import pickle
from pathlib import Path


# Ensure that "/results" is a subdirectory of the working directory.
# The working directory must contain this .py script.
# To run this script correctly, please follow these steps:
# 1. The "/results" directory must exist inside the working directory where this script is located.
# 2. If the "/results" directory does not exist, create it inside the working directory.
# Example directory structure:
# /working_directory/
# ├── several_runs_variability_level.py
# └── results/


current_path = Path.cwd()
path = current_path / "results"


# These experiments were performed with INDEPENDENT random effects. Change config.py and models.py accordingly to reproduce the results presented in the article.
# To model two INDEPENDENT random effects for lambda and h the config.py file should have the following line :
# random_eff = "lambda_h"
# and the models.py file should have:
# cov_latent=pc.MatrixDiagPosDef(dim=config.estimation_description.nindiv),

# Choose one of the following values for variable "residual":
# residual = True to simulate data using the signal-to-noise ratio setting
# residual = False to simulate data using the random effects coefficient of variation setting

residual = False

if residual is True:
    seq_cv = jnp.array([5, 10, 25, 50, 75])
else:
    seq_cv = jnp.array([25, 50, 75])


keyy = 0
Nsimus = 2  # 1000
n_vec = jnp.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
J_vec = jnp.array([50, 25, 17, 12, 10, 8, 7, 6, 5])


# Different sample sizes schemes
def sample_and_estim(theta, n, J, meca_noise, dim, prng_key, n_preheat, n_iter, prng_key_simu):
    key_simu, key_estim = jax.random.split(prng_key)
    # if a key is provided for the simulation, use it (it ensures that the dataset will be the same)
    if prng_key_simu is not None:
        key_simu = jax.random.PRNGKey(prng_key_simu)

    # fix the key in order to have the same density for each fixed value of J
    key_d = jax.random.PRNGKey(0)
    d = jnp.sort(jax.random.choice(key_d, jnp.arange(
        1, 150), shape=(1, J), replace=False))
    d = jnp.tile(d, (n, 1))

    z, y, my, t, vary = models.simu_data(
        theta, n=n, d=d, meca_noise=meca_noise, dim=dim, prng_key=key_simu)

    thetainit = jax.random.uniform(key=key_estim, shape=(
        models.parametrization.size,), minval=0.6, maxval=1.4)*thetatrue
    res = algos.fisher_sgd(y=y, d=d, delta=1, meca_noise=meca_noise, dim=dim, prng_key=key_simu,
                           pre_heating=n_preheat, Nmax=n_iter, theta0=thetainit, optim_step='AdaGrad', factor=0.85)

    return ResEstim(res.theta, y, t)


ResEstim = collections.namedtuple("ResEstim", ("theta", "y", "t"))


def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


for exp in range(len(n_vec)):
    for cv in seq_cv:

        if residual is True:
            thetatrue = {"pop": jnp.array([]),
                         "indiv": {"mean_latent": jnp.array([0.7, 0.5]), "cov_latent": jnp.array([[0.07, 0], [0, 0.05]])},
                         "var_residual": (cv/100)**2 * jnp.exp(0.5*2+0.05)}
        else:
            thetatrue = {"pop": jnp.array([]),
                         "indiv": {"mean_latent": jnp.array([0.7, 0.5]), "cov_latent": (cv/100)**2 * jnp.array([[0.7**2, 0], [0, 0.5**2]])},
                         "var_residual": 0.1**2}

        thetatrue = models.parametrization.params_to_reals1d(thetatrue)

        many_res = [sample_and_estim(thetatrue, n_vec[exp], J_vec[exp], 1, 2, key, 1000, 3500, None)
                    for key in jax.random.split(jax.random.PRNGKey(keyy), Nsimus)]

        if residual is True:
            fname = path / \
                f"allres_n{n_vec[exp]}_J{J_vec[exp]}_residual_cv{cv}.pkl"
            save_object(many_res, fname)
            ftname = path / f"residual_cv{cv}_allres_n{n_vec[exp]}.jnp"
            theta = jnp.array([res.theta for res in many_res])
            with open(ftname, 'wb') as f:
                jnp.save(f, theta)
        else:
            fname = path / \
                f"allres_n{n_vec[exp]}_J{J_vec[exp]}_rand_eff_cv{cv}.pkl"
            save_object(many_res, fname)
            ftname = path / f"rand_eff_cv{cv}_allres_n{n_vec[exp]}.jnp"
            theta = jnp.array([res.theta for res in many_res])
            with open(ftname, 'wb') as f:
                jnp.save(f, theta)
