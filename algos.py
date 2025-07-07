import functools

import jax
import jax.numpy as jnp

from tqdm import tqdm
import models


class O3filter:
    m1: jnp.array
    m2: jnp.array
    m3: jnp.array
    time_cst: float
    last_time: float
    mone: float

    def __init__(self, size, time_cst):
        self.m1 = jnp.zeros(size)
        self.m2 = jnp.zeros(size)
        self.m3 = jnp.zeros(size)
        self.time_cst = time_cst
        self.mone = 0.0
        self.factor = -jnp.expm1(-1.0 / self.time_cst)

    def update(self, val):
        self.mone += self.factor * (1 - self.mone)
        self.m1 = self.m1 + self.factor * (val - self.m1)
        self.m2 = self.m2 + self.factor * (self.m1 / self.mone - self.m2)
        self.m3 = self.m3 + self.factor * (self.m2 / self.mone - self.m3)

    @property
    def unbiaised_m3(self):
        return self.m3 / self.mone


@jax.jit
def moving_average(a, size):
    ret = jnp.cumsum(a, dtype=float, axis=0)
    ret = ret.at[size:,:].set(ret[size:,:] - ret[:-size,:])
    return ret[size - 1:] / size


@functools.partial(jax.jit, static_argnames=('optim_step',))
def one_iter(it,pre_heating,end_heating,theta,sigma_proposal,z,y,d,delta,meca_noise,dim,jac,fim_sa,current_ar,current_grad,current_step,prng_key,optim_step,factor,rho,alpha):
    n, J = y.shape
    n, dim_latent = sigma_proposal.shape
    assert d.shape == (n, J)
    assert z.shape == (n, dim_latent)  

    (sigma_proposal, current_ar, z, log_likli, prng_key) = models.mhrw_step_gibbs_adaptative(
        sigma_proposal=sigma_proposal,
        current_ar=current_ar,    
        it=it,
        theta=theta,
        z=z,
        y=y,
        d=d,
        delta=delta,
        meca_noise=meca_noise,
        dim=dim,
        prng_key=prng_key,
    )
    
    # Gradient of the log-likelihood (per individual)
    current_jac = models.jac_log_likelihood_rows(theta, z, y, d, delta, meca_noise, dim)
    grad = current_jac.mean(axis=0) 

    gamma = jax.lax.cond(
        it <= pre_heating,
        lambda it: 1.0,
        lambda it: 1/(it - pre_heating + 1),
        it,
    )

    fim_sa += gamma * (current_jac - fim_sa) # quantity needed to compute the FIM
    fisher_info_mat = fim_sa.T @ fim_sa / n # estimation of the FIM using stochastic approximation
        
    preconditioner = jnp.ones((models.parametrization.size,))
    theta_step = jnp.zeros(models.parametrization.size)

    if optim_step == 'Fisher-SGD': # Smooth step a        
        jac += current_jac.T @ current_jac 
        preconditioner = jax.lax.cond(
            it < pre_heating,
                lambda preconditioner: jac,
                lambda preconditioner: fisher_info_mat,
            preconditioner,
        )                

        theta_step = jax.lax.cond(
            it < pre_heating,
                lambda theta_step: grad * 1/(jnp.sqrt(jnp.diag(preconditioner)) + 1e-5),
                lambda theta_step: preconditioner @ grad,
            theta_step,
        ) 
        factor = gamma    
    
    if optim_step == 'AdaGrad': # AdaGrad step            
        jac += current_jac.T @ current_jac                
        preconditioner = jnp.diag(jac)
        theta_step = grad * 1/(jnp.sqrt(preconditioner) + 1e-5)

    if optim_step == 'RMSProp':                           
        jac = alpha * jac + (1-alpha) * current_jac.T @ current_jac
        preconditioner = jnp.diag(jac)
        theta_step = grad * 1/(jnp.sqrt(preconditioner) + 1e-5)

    if optim_step == 'Momentum':                        
        theta_step = rho * current_grad + grad
    
    if optim_step == 'Adam':        
        jac = alpha * jac + (1-alpha) * current_jac.T @ current_jac
        preconditioner = jnp.diag(jac)/(1-alpha**(it+1))
        grad = rho * current_grad + (1-rho) * grad
        grad = grad/(1-rho**(it+1))        
        theta_step = grad * 1/(jnp.sqrt(preconditioner) + 1e-5)

    # Update theta
    theta = theta + factor * theta_step
        

    return (theta,sigma_proposal,z,d,fim_sa,jac,current_ar,grad,theta_step,prng_key,theta_step,preconditioner,fisher_info_mat,log_likli,factor)


def fisher_sgd(y,d,delta,meca_noise,dim,prng_key=None,pre_heating=1000,Nmax=10000,theta0=None,optim_step='AdaGrad',factor=1,rho=1,alpha=1):
    # initialize parameter values 
    if prng_key is None:
        prng_key = 0
    if isinstance(prng_key, int):
        prng_key = jax.random.PRNGKey(prng_key)
    prng_key, key = jax.random.split(prng_key)
    
    # initialize parameter values if not provided
    prng_key, key = jax.random.split(prng_key)
    if theta0 is None:
        theta = jax.random.normal(key=key, shape=(models.parametrization.size,))
    else:
        theta = theta0

    psize = models.parametrization.size
    n, J = y.shape
    
    # initialize parameters of the RWMH algo
    p = models.parametrization.reals1d_to_params(theta)     
    dim_latent = len(p.indiv.mean_latent)
    varz = jnp.log(jnp.diag(1/p.indiv.mean_latent) @ p.indiv.cov_latent @ jnp.diag(1/p.indiv.mean_latent) + jnp.ones((dim_latent,dim_latent)))    
    muz = jnp.log(p.indiv.mean_latent) - 0.5*jnp.diag(varz)        
    sigma_proposal = jnp.tile(jnp.abs(muz),(n,1)) * 0.5
    current_ar = jnp.zeros((n,dim_latent))

    # pre-heating phase -> we do not compute the gradient, we just let the MCMC algorithm evolve to reach the
    # stationnary regime    
    z = jax.random.multivariate_normal(key, mean=muz, cov=varz, shape=(n,)) 
    ar_all = []
    z_all = []
    ll_all = []    
    
    for it in tqdm(range(pre_heating)):
        (sigma_proposal, current_ar, z, log_likli, prng_key) = models.mhrw_step_gibbs_adaptative(
            sigma_proposal=sigma_proposal,
            current_ar=current_ar,   
            it=it,
            theta=theta,
            z=z,
            y=y,
            d=d,
            delta=delta,
            meca_noise=meca_noise,
            dim=dim,
            prng_key=prng_key,
        )
        ar_all.append(current_ar)
        z_all.append(z)
        ll_all.append(log_likli)   


    # Now we start the Fisher-SGD algo, with a first heating phase
    theta_step = jnp.zeros(psize)
    grad = jnp.zeros(psize)
    step_all = jnp.zeros((Nmax,psize))
    grad_all = jnp.zeros((Nmax+1,psize))
    precond_all = []
    fim_all = []
    theta_all = jnp.zeros((Nmax+1,psize))
    factor_all = jnp.zeros(Nmax)
    grad_all = grad_all.at[0,:].set(grad)
    theta_all = theta_all.at[0,:].set(theta)
    
    jac = jnp.zeros((psize,psize))
    fim_sa = jnp.zeros((n,psize))
    end_heating = None
    o3_filter = O3filter(models.parametrization.size, 100)
    o3_step_mean = jnp.zeros(models.parametrization.size)


    for it in tqdm(range(Nmax)):
        (
            theta,
            sigma_proposal,
            z,
            d,
            fim_sa,
            jac,
            current_ar,
            grad,
            theta_step,
            prng_key,            
            theta_step,
            preconditioner,
            fim,
            log_likli,
            factor,
        ) = one_iter(
            it=it,
            pre_heating=pre_heating,
            end_heating=end_heating,
            theta=theta,
            sigma_proposal=sigma_proposal,
            z=z,
            y=y,
            d=d,
            meca_noise=meca_noise,
            dim=dim,
            delta=delta,
            jac=jac,
            fim_sa=fim_sa,
            current_ar=current_ar,
            current_grad=grad,
            current_step=theta_step,
            prng_key=prng_key,
            optim_step=optim_step,
            factor=factor,
            rho=rho,
            alpha=alpha,            
        )        
        
        if end_heating is None:
            o3_filter.update(theta_step)
            o3_step_mean, old_o3_step_mean = o3_filter.unbiaised_m3, o3_step_mean
            if (
                it > pre_heating
                and (o3_step_mean**2).sum() > (old_o3_step_mean**2).sum()
            ):
                end_heating = it
        if jnp.max(jnp.isnan(theta)):
            break

        
        ar_all.append(current_ar)
        z_all.append(z)
        ll_all.append(log_likli)
        theta_all = theta_all.at[it+1,:].set(theta)
        step_all = step_all.at[it+1,:].set(theta_step)
        grad_all = grad_all.at[it+1,:].set(grad)
        precond_all.append(preconditioner)
        fim_all.append(fim)
        factor_all = factor_all.at[it].set(factor)


    return ResEstim(        
        theta_all,
        ll_all,
        z_all,
        factor_all,
        ar_all,
        sigma_proposal,
        precond_all,   
        fim_all,            
        step_all,
        grad_all,        
    )


import collections

ResEstim = collections.namedtuple("ResEstim", 
                                  ("theta",                                    
                                   "loglikli",
                                   "latent", 
                                   "factor", 
                                   "ar", 
                                   "sigma_proposal", 
                                   "preconditioner",    
                                   "fim",                                
                                   "step",
                                   "grad"))



