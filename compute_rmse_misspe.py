#!/usr/bin/env python
# coding: utf-8

# # Effect of a model misspecification
import models
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import algos
import pandas as pd

# load results 
def params_to_array(param):
    arr = jnp.concatenate([param.pop,param.indiv.mean_latent,param.indiv.cov_latent[jnp.triu_indices(2)],jnp.array([param.var_residual])])
    return arr

def theta_to_params(theta):
    arr = jnp.array([theta[0],theta[1],jnp.log(1+jnp.exp(theta[2]))**2,(jnp.log(1+jnp.exp(theta[3]))**2+theta[4]**2)/2,(jnp.log(1+jnp.exp(theta[2]))*theta[4])/jnp.sqrt(2),jnp.log(1+jnp.exp(theta[5]))])
    return arr
    

thetanames = [r"$\mu_{\lambda}$",r'$\mu_h$',r'$\sigma_{\lambda}^2$',r'$\sigma_h^2$',r'$\sigma_{\lambda,h}$',r'$\sigma_0^2$']

true_value = pd.DataFrame({'true_value':[0.7,0.5,0.07,0.05,0.03,0.1**2]})
true_value['variable'] = thetanames

import collections
ResEstim = collections.namedtuple("ResEstim",
        ("theta_wrong","theta_correct","ll_wrong","ll_correct","bic_wrong","bic_correct","y","t","z","zw","zc"))


def loadBIC(fn):
     with open(fn, 'rb') as f:
          bic = jnp.load(f)  

     nrep = len(bic)

     # extract information from filename
     str_split = re.split(r'[_.]', fn)   
     size = [idx[1:] for idx in str_split if idx[0] == "n" and not idx[1] == "o"]
     size = int(size[0])
     dim = [idx[1:] for idx in str_split if idx[0] == "d" and not idx[1] == "i" ]
     dim = int(dim[0])
     noise = [idx[-1] for idx in str_split if idx[0] == "m"]
     noise = int(noise[-1])
     modtype = "wrong" if 'bicwrong' in str_split else "correct"
          
     dbic = pd.DataFrame(bic)
     dbic.columns = ['BIC']
     dbic['N'] = size
     dbic['d'] = dim
     dbic['noise'] = noise
     dbic['rep'] = jnp.arange(1,nrep+1)
     dbic['Model'] = modtype

     return dbic


def loadtheta(fn):
     with open(fn, 'rb') as f:
          theta = jnp.load(f)  

     (nrep, nparams) = theta.shape

     # extract information from filename
     str_split = re.split(r'[_.]', fn)   
     size = [idx[1:] for idx in str_split if idx[0] == "n" and not idx[1] == "o"]
     size = int(size[0])
     dim = [idx[1:] for idx in str_split if idx[0] == "d" and not idx[1] == "i" ]
     dim = int(dim[0])
     noise = [idx[-1] for idx in str_split if idx[0] == "m"]
     noise = int(noise[0])
     
     theta = theta.reshape((nrep,nparams))
     dtheta = pd.DataFrame(theta)
     
     dtheta['N'] = jnp.repeat(size,nrep)
     dtheta['d'] = jnp.repeat(dim,nrep)
     dtheta['noise'] = jnp.repeat(noise,nrep)
     dtheta['rep'] = jnp.arange(1,nrep+1)

     return dtheta

# Import BIC
import os 
import re
path = 'results/'
files = [f for f in os.listdir(path) if re.match(r'modchoice_mecanoise1_d2_biccorrect*', f)]
bic_correct = [loadBIC(os.path.join(path, f)) for f in files]
bic_correct = pd.concat(bic_correct)

files = [f for f in os.listdir(path) if re.match(r'modchoice_mecanoise1_d2_bicwrong*', f)]
bic_wrong = [loadBIC(os.path.join(path, f)) for f in files]
bic_wrong = pd.concat(bic_wrong)

import seaborn as sns
bic_all = pd.concat([bic_correct,bic_wrong])
bic_all = bic_all.reset_index()

bp = sns.boxplot(x='N',y='BIC',hue='Model',gap=0.2,fliersize=3,data=bic_all)
sns.move_legend(bp, "lower center",
    bbox_to_anchor=(.5, 1), ncol=1, title=None, frameon=False,
)
plt.savefig("bic_missspe_corr_noise.pdf")
plt.show()

# compute difference in BIC
bic_correct = bic_correct.rename(columns={'BIC':'BIC correct'})
bic_correct = bic_correct.drop(columns=['Model'])
bic_wrong = bic_wrong.rename(columns={'BIC':'BIC wrong'})
bic_wrong = bic_wrong.drop(columns=['Model'])
bic_diff = bic_correct.merge(bic_wrong)
bic_diff['diffBIC'] = bic_diff['BIC correct'] - bic_diff['BIC wrong']

bic_diff['diffBICneg'] = bic_diff['diffBIC']<0
stat_bic = bic_diff.groupby('N').mean()
stat_bic = stat_bic.reset_index()
stat_bic = stat_bic[['N','diffBICneg']]
stat_bic.to_csv('prop_choice_bic_corr_noise.csv')

sns.barplot(data=stat_bic,x='N',y='diffBICneg',width=0.5)
plt.savefig("prop_choice_bic_corr_noise.pdf")

### Estimation
files = [f for f in os.listdir(path) if re.match(r'modchoice_mecanoise1_d2_thetacorrect*', f)]
theta_correct = [loadtheta(os.path.join(path, f)) for f in files]
theta_correct = pd.concat(theta_correct)

files = [f for f in os.listdir(path) if re.match(r'modchoice_mecanoise1_d2_thetawrong*', f)]
theta_wrong = [loadtheta(os.path.join(path, f)) for f in files]
theta_wrong = pd.concat(theta_wrong)

# Transform back the parameters
theta_correct[thetanames[0]] = theta_correct[0]
theta_correct[thetanames[1]] = theta_correct[1]
theta_correct[thetanames[2]] = np.log(1+np.exp(theta_correct[2]))**2
theta_correct[thetanames[3]] = (np.log(1+np.exp(theta_correct[3]))**2+theta_correct[4]**2)/2
theta_correct[thetanames[4]] = (np.log(1+np.exp(theta_correct[2]))*theta_correct[4])/np.sqrt(2)
theta_correct[thetanames[5]] = np.log(1+np.exp(theta_correct[5]))

theta_wrong[thetanames[0]] = theta_wrong[0]
theta_wrong[thetanames[1]] = theta_wrong[1]
theta_wrong[thetanames[2]] = np.log(1+np.exp(theta_wrong[2]))**2
theta_wrong[thetanames[3]] = (np.log(1+np.exp(theta_wrong[3]))**2+theta_wrong[4]**2)/2
theta_wrong[thetanames[4]] = (np.log(1+np.exp(theta_wrong[2]))*theta_wrong[4])/np.sqrt(2)
theta_wrong[thetanames[5]] = np.log(1+np.exp(theta_wrong[5]))


theta_correct = pd.melt(theta_correct.drop(['rep',0,1,2,3,4,5], axis=1)
,id_vars=['N','d','noise'])
theta_wrong = pd.melt(theta_wrong.drop(['rep',0,1,2,3,4,5], axis=1)
,id_vars=['N','d','noise'])



# Plot the estimates together
theta_correct["Model"] = "Correct"
theta_wrong["Model"] = "Wrong"
dtheta_all = pd.concat([theta_correct,theta_wrong])
dtheta_all = dtheta_all.reset_index()

palette = {"Correct": "tab:blue", "Wrong": "tab:orange"}

g = sns.FacetGrid(dtheta_all, col="variable", sharey=False, col_order=thetanames)
g.map_dataframe(sns.boxplot, x="N", y="value",hue="Model",palette=palette,dodge=True,gap=0.1,fliersize=2,width=0.5)
for ax, pos in zip(g.axes.flat, true_value['true_value']):
    ax.axhline(y=pos, color='r')
g.add_legend(title='')
sns.move_legend(g, bbox_to_anchor=(0.5,-0.1), loc='lower center', frameon=False, ncol=2)
# to remove the prefix with variable name in the facet title
for item, ax in g.axes_dict.items():
     ax.grid(False, axis='x')
     ax.set_title(item) 
g.savefig("estim_missspe_corr_noise.pdf")
