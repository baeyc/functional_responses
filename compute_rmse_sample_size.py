#!/usr/bin/env python
# coding: utf-8

import models
import algos
import jax
import jax.numpy as jnp
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import os 
from tqdm import tqdm


theta1 = {"pop" : jnp.array([]), 
             "indiv" : {"mean_latent" : jnp.array([0.7,0.5]), "cov_latent" : jnp.array([[0.07,0.03],[0.03,0.05]])},              
             "var_residual" : 0.1**2}
theta1 = models.parametrization.params_to_reals1d(theta1)

theta2 = {"pop" : jnp.array([]), 
             "indiv" : {"mean_latent" : jnp.array([1.75,0.5]), "cov_latent" : jnp.array([[0.175,0.045],[0.045,0.05]])},              
             "var_residual" : 0.1**2}
theta2 = models.parametrization.params_to_reals1d(theta2)

n_vec = jnp.array([10,20,30,40,50,60,70,80,90])
J_vec = jnp.array([50,25,17,12,10,8,7,6,5])
thetanames = [r"$\mu_{\lambda}$",r'$\mu_h$',r'$\sigma_{\lambda}$',r'$\sigma_h$',r'$\sigma_{\lambda,h}$',r'$\sigma_0$']

# create a panda data frame to be merged with results tables
true_value = pd.DataFrame({'true_value':theta1})
true_value['variable'] = thetanames
true_value['true_val_param'] = [0.7,0.5,np.sqrt(0.07),np.sqrt(0.05),0.5,0.1]

true_value2 = pd.DataFrame({'true_value':theta2})
true_value2['variable'] = thetanames
true_value2['true_val_param'] = [1.75,0.5,np.sqrt(0.175),np.sqrt(0.05),0.5,0.1]

import re
import collections
ResEstim = collections.namedtuple("ResEstim",("theta","y","t"))

# load results
def params_to_array(param):
    arr = jnp.concatenate([param.pop,param.indiv.mean_latent,param.indiv.cov_latent[jnp.triu_indices(2)],jnp.array([param.var_residual])])
    return arr

def theta_to_params(theta):
    arr = jnp.array([theta[0],theta[1],jnp.log(1+jnp.exp(theta[2])),jnp.sqrt((jnp.log(1+jnp.exp(theta[3]))**2+theta[4]**2)/2),theta[4]/jnp.sqrt(jnp.log(1+jnp.exp(theta[3]))+theta[4]),jnp.log(1+jnp.exp(theta[5]))])
    return arr

def load(fn):
     with open(fn, 'rb') as f:
          theta = jnp.load(f)  

     (nrep, niter, nparams) = theta.shape

     # extract information from filename
     str_split = re.split(r'[_.]', fn)   
     size = [idx[1:] for idx in str_split if idx[0] == "n"]
     size = int(size[0])
     dim = [idx[1:] for idx in str_split if idx[0] == "d"]
     dim = int(dim[0])
     noise = [idx[-1] for idx in str_split if idx[0] == "m"]
     noise = int(noise[0])
     
     theta = theta.reshape((nrep*niter,nparams))
     dtheta = pd.DataFrame(theta)
     
     dtheta['N'] = jnp.repeat(size,nrep*niter)
     dtheta['d'] = jnp.repeat(dim,nrep*niter)
     dtheta['noise'] = jnp.repeat(noise,nrep*niter)
     dtheta['rep'] = jnp.repeat(jnp.arange(1,nrep+1),niter)
     dtheta['iter'] = jnp.tile(jnp.arange(1,niter+1),nrep)

     return dtheta


##---------------------------
## Theta 1
## --------------------------
## Mecanistic noise and correlation
path = 'results/'
files = [f for f in os.listdir(path) if re.match(r'theta1_mecanoise1_d2_allres_n*', f)]
params_data = [load(os.path.join(path, f)) for f in files]	
params_data = pd.concat(params_data)


# get the averaged value of the parameters
params_last = params_data.groupby(['N','rep']).mean().reset_index()
# transform back the parameters
params_last[thetanames[0]] = params_last[0]
params_last[thetanames[1]] = params_last[1]
params_last[thetanames[2]] = np.log(1+np.exp(params_last[2])) # std lambda
params_last[thetanames[3]] = np.sqrt((np.log(1+np.exp(params_last[3]))**2+params_last[4]**2)/2) # std h
params_last[thetanames[4]] = params_last[4]/(np.sqrt(2)*params_last[thetanames[3]]) # correlation between lambda and h
params_last[thetanames[5]] = np.sqrt(np.log(1+np.exp(params_last[5]))) # residual std

params_last = pd.melt(params_last.drop(['rep','iter',0,1,2,3,4,5], axis=1)
,id_vars=['N','d','noise'])
params_last = pd.merge(params_last,true_value)
params_last['bias'] = params_last['value'] - params_last['true_val_param']
params_last['bias2'] = params_last['bias']**2


rmse = params_last.groupby(['N','variable'],as_index=False).agg({'bias2' : ['mean'], 'value' : ['var']})
rmse = rmse.droplevel(axis=1, level=[1]).reset_index()
rmse = pd.merge(rmse,true_value2)
rmse['mse'] = rmse['bias2'] + rmse['value']
rmse['relmse'] = np.sqrt(rmse['mse'])/rmse['true_val_param']
rmse = rmse.drop(columns=['true_value','true_val_param'])
rmse.columns = ['index','N','Parameter','Bias (squared)','Variance','mse','relmse']



# Plots
import seaborn as sns
g = sns.FacetGrid(rmse, col="Parameter", sharey=False, col_order=thetanames)
g.map(sns.lineplot, "N", "mse",marker='o')
# to remove the prefix with variable name in the facet title
for item, ax in g.axes_dict.items():
     ax.grid(False, axis='x')
     ax.set_title(item) 
g.savefig("rmse_mecanoise_corr_theta1_params.pdf")

params_last = params_last.rename(columns={"variable": "Parameter"})
g = sns.FacetGrid(params_last, col="Parameter", sharey=False)
g.map(sns.boxplot, "N", "value", color="skyblue")

for ax, pos in zip(g.axes.flat, true_value['true_val_param']):
    ax.axhline(y=pos, color='r')
# to remove the prefix with variable name in the facet title
for item, ax in g.axes_dict.items():
     ax.grid(False, axis='x')
     ax.set_title(item) 
g.savefig("boxplot_estimates_mecanoise_corr_theta1_params.pdf")



## -------------------------------
# Results for theta_2
## -------------------------------
## Mecanistic noise and correlation
path = 'results/'
files = [f for f in os.listdir(path) if re.match(r'theta2_mecanoise1_d2_n*', f)]
params2_data = [load(os.path.join(path, f)) for f in files]
params2_data = pd.concat(params2_data)

# get the averaged value of the parameters
params2_last = params2_data.groupby(['N','rep']).mean().reset_index()
# transform back the parameters
params2_last[thetanames[0]] = params2_last[0]
params2_last[thetanames[1]] = params2_last[1]
params2_last[thetanames[2]] = np.log(1+np.exp(params2_last[2])) # std lambda
params2_last[thetanames[3]] = np.sqrt((np.log(1+np.exp(params2_last[3]))**2+params2_last[4]**2)/2) # std h
params2_last[thetanames[4]] = params2_last[4]/(np.sqrt(2)*params2_last[thetanames[3]]) # correlation between lambda and h
params2_last[thetanames[5]] = np.sqrt(np.log(1+np.exp(params2_last[5]))) # residual std


params2_last = pd.melt(params2_last.drop(['rep','iter',0,1,2,3,4,5], axis=1)
,id_vars=['N','d','noise'])
params2_last = pd.merge(params2_last,true_value2)
params2_last['bias'] = params2_last['value'] - params2_last['true_val_param']
params2_last['bias2'] = params2_last['bias']**2


rmse2 = params2_last.groupby(['N','variable'],as_index=False).agg({'bias2' : ['mean'], 'value' : ['var']})
rmse2 = rmse2.droplevel(axis=1, level=[1]).reset_index()
rmse2 = pd.merge(rmse2,true_value2)
rmse2['mse'] = rmse2['bias2'] + rmse2['value']
rmse2['relmse'] = np.sqrt(rmse2['mse'])/rmse2['true_val_param']
rmse2 = rmse2.drop(columns=['true_value','true_val_param'])
rmse2.columns = ['index','N','Parameter','Bias (squared)','Variance','mse','relmse']

# Plots
g = sns.FacetGrid(rmse2, col="Parameter", sharey=False, col_order=thetanames)
g.map(sns.lineplot, "N", "mse",marker='o')
for item, ax in g.axes_dict.items():
     ax.grid(False, axis='x')
     ax.set_title(item) 
g.savefig("rmse_mecanoise_corr_theta2_params.pdf")

params2_last = params2_last.rename(columns={"variable": "Parameter"})
g = sns.FacetGrid(params2_last, col="Parameter", sharey=False)
g.map(sns.boxplot, "N", "value",color="skyblue")

for ax, pos in zip(g.axes.flat, true_value2['true_val_param']):
    ax.axhline(y=pos, color='r')
for item, ax in g.axes_dict.items():
     ax.grid(False, axis='x')
     ax.set_title(item) 
g.savefig("boxplot_estimates_mecanoise_corr_theta2_params.pdf")


## Plots with both theta_1 and theta_2 on the same graph
## The part below contains technical adjustments for seaborn using functions in utils.py
import utils

rmse['Parameter set'] = r"$\theta_1$"
rmse2['Parameter set'] = r"$\theta_2$"
rmse_all = pd.concat([rmse,rmse2])

g = sns.FacetGrid(rmse_all, col="Parameter", hue="Parameter set", sharey=False, col_order=thetanames, col_wrap=3)
g.map(sns.lineplot, "N", "mse",marker='o')
for item, ax in g.axes_dict.items():
     ax.grid(False, axis='x')
     ax.set_title(item) 
     ax.secondary_xaxis('top', functions=(invert,invert))
     
# relative MSE
sns.set_style("whitegrid")
g = sns.FacetGrid(rmse_all, col="Parameter", hue="Parameter set", sharey=False, sharex = False, col_order=thetanames, col_wrap=3)
g.map(sns.lineplot, 'N', 'relmse',marker='o')
for item, ax in g.axes_dict.items():
	ax.set_title(item,loc='left') 
	ax.grid(False, axis='x')
	ax.set_title('')
	ax.set_ylabel("relative MSE")
	ax.set_xlabel(r'$N$ (nb individuals)')
	ax.set_xticks([20,40,60,80], labels=[20,40,60,80], color='grey') 
	ax.tick_params(axis='x')	
	secax = ax.secondary_xaxis('top', functions=(invert, invert))	
	secax.set_xlabel(r'$n$ (nb obs per individual)', color='grey')
	secax.set_xticks([25,12,8,6]) # Use the same ticks position
	secax.set_xticklabels([25,12,8,6], color='grey') 
	secax.tick_params(axis='x', labelcolor='grey')	
g.add_legend()
sns.move_legend(g, "center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
g.savefig("relMSE_mecanoise_corr_theta1_theta2_params.pdf")


# boxplots of estimates
params_last['Parameter set'] = r"$\theta_1$"
params2_last['Parameter set'] = r"$\theta_2$"
params_all = pd.concat([params_last,params2_last])


palette = {r"$\theta_1$": "tab:blue", r"$\theta_2$": "tab:orange"}
sns.set_style("whitegrid", {'axes.grid' : False})
g = sns.FacetGrid(params_all, col="Parameter", sharey=False, col_wrap=3)
g.map_dataframe(plot_boxplot_with_twinx_axis)

for ax, pos in zip(g.axes.flat, true_value['true_val_param']):
    ax.axhline(y=pos, color='tab:blue',linestyle="-")
for ax, pos in zip(g.axes.flat, true_value2['true_val_param']):
    ax.axhline(y=pos, color='tab:orange',linestyle="-")
for item, ax in g.axes_dict.items():
	ax.set_title(item,loc='left') 
	ax.grid(False, axis='x')
	ax.set_title('')

g.add_legend()
sns.move_legend(g, "center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout()
g.savefig("boxplot_estimates_mecanoise_corr_theta1_theta2_params.pdf")



