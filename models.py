import functools
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from functools import partial

from tqdm import tqdm

import parametrization_cookbook.jax as pc
from parametrization_cookbook.functions.jax import expit

import config

#### WARNING #####
# To run the model for a given random effect configuration, the config.py file should be modified before running the script
# To model two CORRELATED random effects for lambda and h the config.py file should have the following line :
# random_eff = "lambda_h"
# and the models.py file should have:
# cov_latent=pc.MatrixSymPosDef(dim=config.estimation_description.nindiv),

# To model two INDEPENDENT random effects for lambda and h the config.py file should have the following line :
# random_eff = "lambda_h"
# and the models.py file should have:
# cov_latent=pc.MatrixDiagPosDef(dim=config.estimation_description.nindiv),


parametrization = pc.NamedTuple(
    pop=pc.Real(shape=config.estimation_description.npop),
    indiv=pc.NamedTuple(
        mean_latent=pc.Real(shape=config.estimation_description.nindiv),
        cov_latent=pc.MatrixSymPosDef(dim=config.estimation_description.nindiv),
    ),    
    var_residual=pc.RealPositive(),
)


# definition of constants to be used throughout the models
c2E = (jnp.sqrt(2) + jnp.log(1+jnp.sqrt(2)))/3
c2V = 2/3
c3E = (6*jnp.sqrt(3) - jnp.pi + jnp.log(3650401+2107560*jnp.sqrt(3)))/24
c3V = 1


# individual complete log likelihood functions 
@jax.jit
def comp_log_likelihood_rows(theta, z, y, d, delta=1, meca_noise=1., dim=2):
    p = parametrization.reals1d_to_params(theta)
    n, J = y.shape
    dim_latent = len(p.indiv.mean_latent)
    assert z.shape == (n, dim_latent)
    assert d.shape == (n, J)
    
    
    # from the mean and variance of the (log-scaled) latent to the lognormal parameters
    varz = jnp.log(jnp.diag(1/p.indiv.mean_latent) @ p.indiv.cov_latent @ jnp.diag(1/p.indiv.mean_latent) + jnp.ones((dim_latent,dim_latent)))
    muz = jnp.log(p.indiv.mean_latent) - 0.5*jnp.diag(varz)
    
    # likelihood of the latent variables     
    dlogz = z - muz
    log_likli_latent = (
        - 0.5 * jnp.linalg.slogdet(varz)[1]
        - 0.5 * dim_latent * jnp.log(2 * jnp.pi)
        - 0.5 * ((dlogz @ jnp.linalg.inv(varz)) * dlogz).sum(axis=1)
        )

    # likelihood of observations given the latent
    if dim_latent == 2:
        lamb = jnp.exp(z[:, 0][:, None])
        h = jnp.exp(z[:, 1][:, None])
    elif dim_latent == 1: # at least 1 random effect
         if 'lambda' in config.estimation_description.population_model_parameters: 
             lamb = jnp.exp(p.pop)
             h = jnp.exp(z[:, 0][:, None])
         elif 'h' in config.estimation_description.population_model_parameters:
             lamb = jnp.exp(z[:, 0][:, None])
             h = jnp.exp(p.pop)
    
    ypred, my, vary = model(lamb=lamb,
                            h=h,
                            delta=delta,
                            var_res=p.var_residual,
                            d=d,
                            meca_noise=meca_noise,
                            dim=dim)    
    
    dy = y-my
    dyvar = dy**2/vary
    dyvar.shape

    log_likli_obs = (
        - 0.5 * jnp.log(2 * jnp.pi * jnp.nansum(vary,axis=1)) * 11
        - 0.5 * jnp.nansum(dyvar,axis=1)
    ) 
    
    return log_likli_latent + log_likli_obs


# individual conditional log likelihood functions 
@jax.jit
def cond_log_likelihood_rows(theta, z, y, d, delta=1, meca_noise=1.,dim=2):
    p = parametrization.reals1d_to_params(theta)
    n, J = y.shape
    dim_latent = len(p.indiv.mean_latent)
    assert z.shape == (n, dim_latent)
    assert d.shape == (n, J)
    
    # likelihood of observations given the latent
    if dim_latent == 2:
        lamb = jnp.exp(z[:, 0][:, None])
        h = jnp.exp(z[:, 1][:, None])
    elif dim_latent == 1: # at least 1 random effect
         if 'lambda' in config.estimation_description.population_model_parameters: 
             lamb = jnp.exp(p.pop)
             h = jnp.exp(z[:, 0][:, None])
         elif 'h' in config.estimation_description.population_model_parameters:
             lamb = jnp.exp(z[:, 0][:, None])
             h = jnp.exp(p.pop)
    
    ypred, my, vary = model(lamb=lamb,
                            h=h,
                            delta=delta,
                            var_res=p.var_residual,
                            d=d,
                            meca_noise=meca_noise,
                            dim=dim)    

    dy2 = (y - my)**2/vary
    log_likli_obs = - 0.5 * J * jnp.log(2 * jnp.pi) - 0.5 * jnp.sum(jnp.log(vary),axis=1) - 0.5 * jnp.sum(dy2,axis=1)
    
    return log_likli_obs


#@partial(jax.jit, static_argnums=(5,))
def log_likelihood_is(theta, latent, y, d, delta=1, N=1000, meca_noise=0., dim=2, prng_key=0):
    if jnp.ndim(prng_key) == 0:
        prng_key = jax.random.PRNGKey(prng_key)

    key1, key2 = jax.random.split(prng_key)

    p = parametrization.reals1d_to_params(theta)
    dim_latent = len(p.indiv.mean_latent)
    n, J = y.shape
    
    # from the mean and variance of the (log-scaled) latent to the lognormal parameters
    varz = jnp.log(jnp.diag(1/p.indiv.mean_latent) @ p.indiv.cov_latent @ jnp.diag(1/p.indiv.mean_latent) + jnp.ones((dim_latent,dim_latent)))
    muz = jnp.log(p.indiv.mean_latent) - 0.5*jnp.diag(varz)    
    
    # parameters of the importance sampling distribution
    muz_is = jnp.mean(latent,axis=0)
    sd_is = 25*jnp.sqrt(jnp.var(latent,axis=0)) ## mettre à 25

    # we then generate samples from a normal distribution with mean and variance
    # given by the conditional mean and variance computed above
    logzstd = jax.random.normal(key=key1, shape=(N, n, dim_latent))
    logz = jnp.array([logzstd[i,:,:] * sd_is[i,:] + muz_is[i,:] for i in range(N)])
    dlogz = (logz-muz_is)/sd_is
    # We compute quantities that will be used at the end of the IS estimator
    # 2. likelihood of the latent variables     
    likli_latent = jax.scipy.stats.multivariate_normal.pdf(logz,muz,varz)
    # 3. importance sampling function
    is_latent = jnp.prod(jax.scipy.stats.norm.pdf(dlogz),axis=2)
    indiv_fn_to_is = []
    for i in tqdm(range(N)):    
        if dim_latent == 2:
            lamb = jnp.exp(logz[i,:, 0][:, None]) 
            h = jnp.exp(logz[i,:, 1][:, None]) 
        elif dim_latent == 1: # at least 1 random effect
            if 'lambda' in config.estimation_description.population_model_parameters: 
                lamb = jnp.exp(p.pop) 
                h = jnp.exp(logz[i,:, 0][:, None]) 
            elif 'h' in config.estimation_description.population_model_parameters:
                lamb = jnp.exp(logz[i,:, 0][:, None]) 
                h = jnp.exp(p.pop) 
    
        ypred, meany, vary = model(lamb=lamb,h=h,delta=delta,var_res=p.var_residual,d=d,meca_noise=meca_noise,dim=dim)

        # quantities that are averaged by IS
        # 1. conditional likelihood of the observation given the latent
      
        indiv_cond_lik = jnp.nanprod(jax.scipy.stats.norm.pdf((y-meany)/jnp.sqrt(vary)),axis=1)
      
        indiv_fn_to_is.append(indiv_cond_lik*likli_latent[i,:]/is_latent[i,:])
   
    indiv_fn_to_is = np.array(indiv_fn_to_is)
   
    indiv_lik_is = np.mean(indiv_fn_to_is,axis=0) # average over the N samples
    var_lik_is = np.var(indiv_fn_to_is,axis=0) # variance over the N samples
    loglik_is = np.log(indiv_lik_is)
    loglik_is = np.nansum(loglik_is)
    
    return indiv_lik_is, var_lik_is, loglik_is
    

@jax.jit
def comp_log_likelihood(theta, z, y, d, delta=1, meca_noise=1., dim=2):
    return comp_log_likelihood_rows(theta, z, y, d, delta, meca_noise, dim).sum()
    
jac_log_likelihood_rows = jax.jit(jax.jacfwd(comp_log_likelihood_rows))

@jax.jit
def cond_log_likelihood(theta, z, y, d, delta=1, meca_noise=1., dim=2):
    return cond_log_likelihood_rows(theta, z, y, d, delta, meca_noise, dim).sum()


def mh_step_gibbs_prior_prop(mean_proposal, sigma_proposal, current_ar, it, theta, z, y, d, delta, meca_noise, dim, prng_key):
    p = parametrization.reals1d_to_params(theta)
    n, J = y.shape
    dim_latent = len(p.indiv.mean_latent)
    assert z.shape ==(n, dim_latent)
    assert d.shape == (n, J)
    assert sigma_proposal.shape == (dim_latent,)

    log_likli = cond_log_likelihood_rows(theta, z, y, d, delta, meca_noise, dim)    
    keys = jax.random.split(prng_key, dim_latent)

    ar = jnp.zeros(dim_latent)

    for k in range(dim_latent):
        key1, key2 = jax.random.split(keys[k], 2)
        z_propo = z.at[:,k].set(jax.random.normal(key1, shape=(n,)) * sigma_proposal[k] + mean_proposal[k])
        log_likli_propo = cond_log_likelihood_rows(theta, z_propo, y, d, delta, meca_noise, dim)

        lalpha = (log_likli_propo - log_likli)
        u = jax.random.uniform(key=key2, shape=(n,))
        mask = (u < jnp.exp(lalpha))
        z = z.at[:,k].set(z_propo[:,k] * mask + z[:,k] * (1 - mask))
        log_likli = log_likli_propo * mask + log_likli * (1 - mask)
        ar = ar.at[k].set(mask.sum())

    current_ar = (it * current_ar + (ar / n))/(it+1)

    return current_ar, z, log_likli, prng_key


def mhrw_step_gibbs(theta, z, y, d, delta, meca_noise, dim, sigma_proposal, prng_key):
    p = parametrization.reals1d_to_params(theta)
    n, J = y.shape
    dim_latent = len(p.indiv.mean_latent)
    assert z.shape ==(n, dim_latent)
    assert d.shape == (n, J)
    assert sigma_proposal.shape == (n,dim_latent)

    log_likli = comp_log_likelihood_rows(theta, z, y, d, delta, meca_noise, dim)    
    keys = jax.random.split(prng_key, dim_latent)

    ar = jnp.zeros((n,dim_latent))

    for k in range(dim_latent):
        key1, key2 = jax.random.split(keys[k], 2)
        z_propo = z.at[:, k].add(
            jax.random.normal(key1, shape=(n,)) * sigma_proposal.T[k]
            )
        log_likli_propo = comp_log_likelihood_rows(
            theta=theta, 
            z=z_propo, 
            y=y, 
            d=d, 
            delta=delta,
            meca_noise=meca_noise,
            dim=dim)

        lalpha = (log_likli_propo - log_likli)        
        mask = (jax.random.uniform(key=key2, shape=(n,)) < jnp.exp(lalpha))
        z = z.at[:, k].set(jnp.where(mask, z_propo[:, k], z[:,k]))
        log_likli = jnp.where(mask, log_likli_propo, log_likli)
        ar = ar.at[:,k].set(jnp.where(mask,1,0))

    return ar, z, log_likli


@jax.jit
def mhrw_step_gibbs_adaptative(sigma_proposal, current_ar, it, theta, z, y, d, delta, meca_noise, dim, prng_key):
    n, J = y.shape
    n, dim_latent = sigma_proposal.shape
    assert d.shape == (n, J)
    assert z.shape == (n, dim_latent)
    assert sigma_proposal.shape == (n,dim_latent)
    
    prng_key, key = jax.random.split(prng_key)
    
    accept, z, log_likli = mhrw_step_gibbs(theta, z, y, d, delta, meca_noise, dim, sigma_proposal, prng_key)      
    current_ar = ((it+1) * current_ar + accept)/(it+2)  
    mask = current_ar < 0.45
    sigma_proposal = jnp.where(mask,sigma_proposal/1.01,sigma_proposal*1.01)  
    
    return (sigma_proposal, current_ar, z, log_likli, prng_key)


@jax.jit
def model(lamb, h, delta, var_res, d, meca_noise = 1., dim = 2, prng_key=0):
    if jnp.ndim(prng_key) == 0:
        prng_key = jax.random.PRNGKey(prng_key)
        
    (n,J) = d.shape
    
    cE = jax.lax.cond(dim==2,
             lambda dim: (jnp.sqrt(dim) + jnp.log(1+jnp.sqrt(dim)))/3,
             lambda dim: (6*jnp.sqrt(dim) - jnp.pi + jnp.log(3650401+2107560*jnp.sqrt(dim)))/24,
             dim)
    cV = dim/3

    num = d**(1/dim) - 1
    denom = cE * lamb + num*h
    varmeca = ((cV - cE**2)/delta) * (lamb**2*num) / denom**3 

    my = num / denom    
    
    vary = var_res + meca_noise*varmeca 
    
    y = my + jax.random.normal(key=prng_key, shape=(n, J)) * jnp.sqrt(vary)
    return y, my, vary
    

# simulation of data 
@jax.jit
def simu_data(theta, n, d=20, delta  = 1, meca_noise = 1., dim = 2, prng_key=0):
    if jnp.ndim(prng_key) == 0:
        prng_key = jax.random.PRNGKey(prng_key)

    if isinstance(d, int):
        d = jnp.linspace(1, 25, d)
        d = jnp.tile(d,(n,1))    
    (n,J) = d.shape

    p = parametrization.reals1d_to_params(theta)
    key1, key2 = jax.random.split(prng_key)
    dim_latent = len(p.indiv.mean_latent)
    
    # from the mean and variance of the (log-scaled) latent to the lognormal parameters    
    varz = jnp.log(jnp.diag(1/p.indiv.mean_latent) @ p.indiv.cov_latent @ jnp.diag(1/p.indiv.mean_latent) + jnp.ones((dim_latent,dim_latent)))
    muz = jnp.log(p.indiv.mean_latent) - 0.5*jnp.diag(varz)
    logz = jax.random.normal(key=key1, shape=(n, dim_latent)) @ jnp.linalg.cholesky(varz).T + muz
    z = jnp.exp(logz)    
    
    if dim_latent == 2:
        lamb = z[:, 0][:, None]
        h = z[:, 1][:, None]
    elif dim_latent == 1: # at least 1 random effect
         if 'lambda' in config.estimation_description.population_model_parameters: 
             lamb = jnp.exp(p.pop)
             h = z[:, 0][:, None]
         elif 'h' in config.estimation_description.population_model_parameters:
             lamb = z[:, 0][:, None]
             h = jnp.exp(p.pop)
    
    y, my, vary = model(lamb,h,delta,p.var_residual,d,meca_noise,dim,prng_key)    
    
    return logz, y, my, d, vary
