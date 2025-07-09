import collections
import models
import jax.numpy as jnp
import jax
import algos
import pickle


# These experiments were performed with independent random effects. Change config.py and models.py accordingly to reproduce the results presented in the article.
# To model two INDEPENDENT random effects for lambda and h the config.py file should have the following line :
# random_eff = "lambda_h"
# and the models.py file should have:
# cov_latent=pc.MatrixDiagPosDef(dim=config.estimation_description.nindiv),


# residual = True to simulate data using the signal-to-noise ratio setting
# residual = False to simulate data using the random effects coefficient of variation setting
residual = False

if residual is True:
    seq_cv = jnp.array([5, 10, 25, 50, 75])
else:
    seq_cv = jnp.array([25, 50, 75])

keyy = 0
Nsimus = 1000
n_vec = jnp.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
J_vec = jnp.array([50, 25, 17, 12, 10, 8, 7, 6, 5])


ResBIC = collections.namedtuple("ResBIC", ("BIC", "Loglik", "Var_lik"))


def bic(theta, latent, meca_noise, y, d, dim, N=1000):
    (n, J) = d.shape
    indiv_lik, var_lik, ll = models.log_likelihood_is(
        theta, latent, y, d, delta=1, N=N, meca_noise=meca_noise, dim=dim)
    bic = -2*ll + 4*jnp.log(n) + jnp.log(n*J)
    return ResBIC(bic, ll, var_lik)


# Different sample sizes schemes
def model_choice_noise(theta, n, J, meca_noise, dim, prng_key, n_preheat, n_iter):
    key_simu, key_estim, key_ll = jax.random.split(prng_key, num=3)

    # fix the key in order to have the same density for each fixed value of J
    key_d = jax.random.PRNGKey(0)
    d = jnp.sort(jax.random.choice(key_d, jnp.arange(
        1, 150), shape=(1, J), replace=False))
    d = jnp.tile(d, (n, 1))
    z, y, my, t, vary = models.simu_data(
        theta, n=n, d=d, meca_noise=meca_noise, dim=dim, prng_key=key_simu)

    thetainit = jax.random.uniform(key=key_simu, shape=(
        models.parametrization.size,), minval=0.6, maxval=1.4)*thetatrue

    wrong_noise = 1 - meca_noise

    wrong_res = algos.fisher_sgd(y=y, d=d, delta=1, meca_noise=wrong_noise, dim=dim, prng_key=key_estim,
                                 pre_heating=n_preheat, Nmax=n_iter, theta0=thetainit, optim_step='AdaGrad', factor=0.85)
    correct_res = algos.fisher_sgd(y=y, d=d, delta=1, meca_noise=meca_noise, dim=dim, prng_key=key_estim,
                                   pre_heating=n_preheat, Nmax=n_iter, theta0=thetainit, optim_step='AdaGrad', factor=0.85)

    N = 1000
    theta_wrong = jnp.mean(wrong_res.theta[:, -1000:n_iter], axis=0)
    theta_correct = jnp.mean(correct_res.theta[:, -1000:n_iter], axis=0)
    latent_wrong = jnp.array(wrong_res.latent)[-1000:, :, :]
    latent_correct = jnp.array(correct_res.latent)[-1000:, :, :]
    bic_wrong, var_ll_wrong, ll_wrong = bic(
        theta_wrong, latent_wrong, wrong_noise, y, d, dim, N)
    bic_correct, var_ll_correct, ll_correct = bic(
        theta_correct, latent_correct, meca_noise, y, d, dim, N)

    return ResEstim(theta_wrong, theta_correct, ll_wrong, ll_correct, bic_wrong, bic_correct, y, t, z, wrong_res, correct_res)


ResEstim = collections.namedtuple("ResEstim", ("theta_wrong", "theta_correct",
                                  "ll_wrong", "ll_correct", "bic_wrong", "bic_correct", "y", "t", "z", "zw", "zc"))


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

        theta_wrong = []
        theta_correct = []
        ll_wrong = []
        ll_correct = []
        bic_wrong = []
        bic_correct = []

        key_all = jax.random.split(jax.random.PRNGKey(keyy), Nsimus)
        for i in range(Nsimus):
            res_comp = model_choice_noise(
                thetatrue, n_vec[exp], J_vec[exp], 1, 2, key_all[i], 500, 3000)

            theta_wrong.append(res_comp.theta_wrong)
            theta_correct.append(res_comp.theta_correct)
            ll_wrong.append(res_comp.ll_wrong)
            ll_correct.append(res_comp.ll_correct)
            bic_wrong.append(res_comp.bic_wrong)
            bic_correct.append(res_comp.bic_correct)

        if residual is True:
            with open("modchoice_misspenoise_thetawrong_n"+str(n_vec[exp])+"_J"+str(J_vec[exp])+"_residual_cv"+str(cv)+".npy", 'wb') as f:
                jnp.save(f, theta_wrong)
            with open("modchoice_misspenoise_thetacorrect_n"+str(n_vec[exp])+"_J"+str(J_vec[exp])+"_residual_cv"+str(cv)+".npy", 'wb') as f:
                jnp.save(f, theta_correct)
            with open("modchoice_misspenoise_llwrong_n"+str(n_vec[exp])+"_J"+str(J_vec[exp])+"_residual_cv"+str(cv)+".npy", 'wb') as f:
                jnp.save(f, ll_wrong)
            with open("modchoice_misspenoise_llcorrect_n"+str(n_vec[exp])+"_J"+str(J_vec[exp])+"_residual_cv"+str(cv)+".npy", 'wb') as f:
                jnp.save(f, ll_correct)
            with open("modchoice_misspenoise_bicwrong_n"+str(n_vec[exp])+"_J"+str(J_vec[exp])+"_residual_cv"+str(cv)+".npy", 'wb') as f:
                jnp.save(f, bic_wrong)
            with open("modchoice_misspenoise_biccorrect_n"+str(n_vec[exp])+"_J"+str(J_vec[exp])+"_residual_cv"+str(cv)+".npy", 'wb') as f:
                jnp.save(f, bic_correct)
            fname = "res_misspenoise_n" + \
                str(n_vec[exp])+"_J"+str(J_vec[exp])+"_residual_cv" + \
                str(cv) + ".pkl"
            save_object(res_comp, fname)
        else:
            with open("modchoice_misspenoise_thetawrong_n"+str(n_vec[exp])+"_J"+str(J_vec[exp])+"_rand_eff_cv"+str(cv)+".npy", 'wb') as f:
                jnp.save(f, theta_wrong)
            with open("modchoice_misspenoise_thetacorrect_n"+str(n_vec[exp])+"_J"+str(J_vec[exp])+"_rand_eff_cv"+str(cv)+".npy", 'wb') as f:
                jnp.save(f, theta_correct)
            with open("modchoice_misspenoise_llwrong_n"+str(n_vec[exp])+"_J"+str(J_vec[exp])+"_rand_eff_cv"+str(cv)+".npy", 'wb') as f:
                jnp.save(f, ll_wrong)
            with open("modchoice_misspenoise_llcorrect_n"+str(n_vec[exp])+"_J"+str(J_vec[exp])+"_rand_eff_cv"+str(cv)+".npy", 'wb') as f:
                jnp.save(f, ll_correct)
            with open("modchoice_misspenoise_bicwrong_n"+str(n_vec[exp])+"_J"+str(J_vec[exp])+"_rand_eff_cv"+str(cv)+".npy", 'wb') as f:
                jnp.save(f, bic_wrong)
            with open("modchoice_misspenoise_biccorrect_n"+str(n_vec[exp])+"_J"+str(J_vec[exp])+"_rand_eff_cv"+str(cv)+".npy", 'wb') as f:
                jnp.save(f, bic_correct)
            fname = "res_misspenoise_n" + \
                str(n_vec[exp])+"_J"+str(J_vec[exp])+"_rand_eff_cv" + \
                str(cv) + ".pkl"
            save_object(res_comp, fname)
