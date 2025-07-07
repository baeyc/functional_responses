import models
import jax.numpy as jnp
import jax
from tqdm import tqdm
import algos
import math 
import pickle

import matplotlib.pyplot as plt

# Set cov=True to include correlation between the random effects
# Set thetanum=2 to use the parameter set theta2
cov = True
theta_num = 2

theta1 = (theta_num == 1)
theta2 = not theta1

thetatrue = {"pop" : jnp.array([]), 
             "indiv" : {"mean_latent" : jnp.array([0.7*theta1 + 1.75*theta2,0.5]), "cov_latent" : jnp.array([[0.07*theta1 + 0.175*theta2,0+cov*(theta1*0.03+theta2*0.045)],[0+cov*(theta1*0.03+theta2*0.045),0.05]])},              
             "var_residual" : 0.1**2}
thetatrue = models.parametrization.params_to_reals1d(thetatrue)


# Different sample sizes schemes
def sample_and_estim(theta,n,J,meca_noise,dim,prng_key,n_preheat,n_iter,prng_key_simu):
    key_simu, key_estim = jax.random.split(prng_key)
    if prng_key_simu is not None: # if a key is provided for the simulation, use it (it ensures that the dataset will be the same)
        key_simu = jax.random.PRNGKey(prng_key_simu)

    # fix the key in order to have the same density for each fixed value of J    
    key_d = jax.random.PRNGKey(0)
    d = jnp.sort(jax.random.choice(key_d,jnp.arange(1,150),shape=(1,J),replace=False))
    d = jnp.tile(d,(n,1))
    z, y, my, t, vary = models.simu_data(theta, n=n, d=d, meca_noise=meca_noise, dim=dim,prng_key=key_simu)

    thetainit = jax.random.uniform(key=key_estim, shape=(models.parametrization.size,), minval=0.6, maxval=1.4)*thetatrue  
    res = algos.fisher_sgd(y=y,d=d,delta=1,meca_noise=meca_noise,dim=dim,prng_key=key_simu,pre_heating=n_preheat,Nmax=n_iter,theta0=thetainit,optim_step='AdaGrad',factor=0.85)

    return ResEstim(res.theta, y, t)


import collections
ResEstim = collections.namedtuple("ResEstim",("theta","y","t"))

keyy = 0
Nsimus = 1000
n_vec = jnp.array([10,20,30,40,50,60,70,80,90])
J_vec = jnp.array([50,25,17,12,10,8,7,6,5])
meca_noise = 1
dim = 2

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


for exp in range(len(n_vec)):
	many_res = [sample_and_estim(thetatrue,n_vec[exp],J_vec[exp],meca_noise,dim,key,1000,3500,None) for key in jax.random.split(jax.random.PRNGKey(keyy), Nsimus)]
	fname = "mecanoise"+str(meca_noise)+"_d"+str(dim)+"_allres_n"+str(n_vec[exp])+"_J"+str(J_vec[exp])+"theta"+str(theta_num)+".pkl"
	save_object(many_res,fname)
	ftname = "theta"+str(theta_num)+"_mecanoise"+str(meca_noise)+"_d"+str(dim)+"_allres_n"+str(n_vec[exp])+".jnp"
	theta = jnp.array([res.theta for res in many_res])
	with open(ftname,'wb') as f:
		jnp.save(f,theta)
