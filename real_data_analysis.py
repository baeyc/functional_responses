import config
import models
import jax.numpy as jnp
import jax
from tqdm import tqdm
import algos
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

# Import real data
data = pd.read_csv("schroder_et_al.csv")

data['ID']
yreal = pd.pivot_table(data,index='ID', columns='Density', values='Feeding').to_numpy()/2
n = yreal.shape[0]
dreal = jnp.tile(data['Density'].unique(), (n,1))


# Initialize algos
thetainit = {"pop" : jnp.array([]), 
             "indiv" : {"mean_latent" : jnp.array([0.15,0.008]), "cov_latent" : jnp.array([[0.02**2,0.00008],[0.00008,0.008**2]])}, 
             "var_residual" : 98}

thetanames = [r"$\mu_{\lambda}$",r'$\mu_h$',r'$\sigma_{\lambda}^2$',r'$\sigma_h^2$',r'$\sigma_{\lambda,h}$',r'$\sigma^2_0$']

n_heat = 1000
n_tot = 10000
Nrep = 10

# Define two random keys, one for the initialization of the SGD algo and one for the initialization of the parameter values
key = jax.random.split(jax.random.PRNGKey(0), Nrep)
keyinit = jax.random.split(jax.random.PRNGKey(100), Nrep)

theta0 = [jax.random.uniform(key=keyinit[i], shape=(models.parametrization.size,), minval=0.9, maxval=1.1)*models.parametrization.params_to_reals1d(thetainit) for i in range(Nrep)]


#####################         RUN ALGORITHMS       ###########################
res_nomecanoise = [algos.fisher_sgd(y=yreal,d=dreal,delta=1.,meca_noise=0,dim=3,prng_key=key[i],pre_heating=n_heat,Nmax=n_tot,theta0=theta0[i],optim_step='AdaGrad',factor=0.8,alpha=0.9,rho=0.9) for i in range(Nrep)]
# Check if any repetition lead to NaN values, and loop until we have 10 rep
res_nomecanoise = [x for x in res_nomecanoise if not x.theta[-1][0] == 0]
j = 0
while len(res_nomecanoise) < 10:
	j += 1
	key2 = jax.random.split(jax.random.PRNGKey(j),2)
	keyinit2 = jax.random.split(jax.random.PRNGKey(100+j),2)
	theta02 = jax.random.uniform(key=keyinit2[1], shape=(models.parametrization.size,), minval=0.9, maxval=1.1)*models.parametrization.params_to_reals1d(thetainit)
	x = algos.fisher_sgd(y=yreal,d=dreal,delta=1.,meca_noise=0,dim=3,prng_key=key2[1],pre_heating=n_heat,Nmax=n_tot,theta0=theta02,optim_step='Adam',factor=0.18,alpha=0.85,rho=0.75)
	if not x.theta[-1][0] == 0:
		res_nomecanoise.append(x)
		
res_mecanoise = [algos.fisher_sgd(y=yreal,d=dreal,delta=1.,meca_noise=1,dim=3,prng_key=key[i],pre_heating=n_heat,Nmax=n_tot,theta0=theta0[i],optim_step='Adam',factor=0.18,alpha=0.85,rho=0.75) for i in range(Nrep)]
res_mecanoise = [x for x in res_mecanoise if not x.theta[-1][0] == 0]
j = 0
while len(res_mecanoise) < 10:
	j += 1
	key2 = jax.random.split(jax.random.PRNGKey(j),2)
	keyinit2 = jax.random.split(jax.random.PRNGKey(100+j),2)
	theta02 = jax.random.uniform(key=keyinit2[1], shape=(models.parametrization.size,), minval=0.9, maxval=1.1)*models.parametrization.params_to_reals1d(thetainit3)		
	x = algos.fisher_sgd(y=yreal,d=dreal,delta=1.,meca_noise=1,dim=3,prng_key=key2[1],pre_heating=n_heat,Nmax=n_tot,theta0=theta02,optim_step='Adam',factor=0.18,alpha=0.85,rho=0.75)
	if not x.theta[-1][0] == 0:
		res_mecanoise.append(x)	


theta_nomecanoise = jnp.array([res.theta for res in res_nomecanoise])
theta_mecanoise = jnp.array([res.theta for res in res_mecanoise])

latent_nomecanoise = jnp.array([jnp.array(res.latent)[-1000:,:,:] for res in res_nomecanoise])
latent_mecanoise = jnp.array([jnp.array(res.latent)[-1000:,:,:] for res in res_mecanoise])

fim_nomecanoise = jnp.array([res.fim[-1] for res in res_nomecanoise])
fim_mecanoise = jnp.array([res.fim[-1] for res in res_mecanoise])

# average of FIM (otherwise, some runs of the algo lead to numerical instabilities)
fim_nonoise = jnp.array([fim_nomecanoise[i] for i in range(Nrep)])
fim_nonoise = jnp.mean(fim_nonoise,axis=0)
theta_nonoise = jnp.array([theta_nomecanoise[i][-1,] for i in range(Nrep)])
theta_nonoise_mean = jnp.mean(theta_nonoise,axis=0)
latent_nonoise = jnp.mean(latent_nomecanoise,axis=0)

fim_noise = jnp.array([fim_mecanoise[i] for i in range(Nrep)])
fim_noise = jnp.mean(fim_noise,axis=0)
theta_noise = jnp.array([theta_mecanoise[i][-1,] for i in range(Nrep)])
theta_noise_mean = jnp.mean(theta_noise,axis=0)
latent_noise = jnp.mean(latent_mecanoise,axis=0)


# Transform unconstrained parameters back into their original scale
params_nomecanoise = [[models.parametrization.reals1d_to_params(x) for x in theta_nomecanoise[i,:,:]] for i in range(Nrep)]
params_mecanoise = [[models.parametrization.reals1d_to_params(x) for x in theta_mecanoise[i,:,:]] for i in range(Nrep)]

# reshape to get an array
params_nomecanoise = [jnp.array([jnp.concatenate((p.indiv.mean_latent.reshape(2,),jnp.sqrt(jnp.diag(p.indiv.cov_latent)),p.indiv.cov_latent[0,1]/jnp.sqrt(jnp.prod(jnp.diag(p.indiv.cov_latent))).reshape(1,),
jnp.sqrt(p.var_residual.reshape(1,)))) for p in params_nomecanoise[i]]) for i in range(Nrep)]
params_mecanoise = [jnp.array([jnp.concatenate((p.indiv.mean_latent.reshape(2,),jnp.sqrt(jnp.diag(p.indiv.cov_latent)),p.indiv.cov_latent[0,1]/jnp.sqrt(jnp.prod(jnp.diag(p.indiv.cov_latent))).reshape(1,),jnp.sqrt(p.var_residual.reshape(1,)))) for p in params_mecanoise[i]]) for i in range(Nrep)]


#####################         PLOTS RESULTS       ###########################
fig,axs = plt.subplots(nrows=2,ncols=3,figsize=(9,6))
axs = axs.ravel()
for i in range(Nrep):
    for j in range(6):        
        axs[j].plot(params_nomecanoise[i][:,j],alpha=0.75,linewidth=1)   
	axs[j].title.set_text(thetanames[j])
fig.suptitle("No mecanistic noise")    
fig.tight_layout()  
plt.show()     
fig.savefig("realdata_nomecanoise.pdf")    

fig,axs = plt.subplots(nrows=2,ncols=3,figsize=(9,6))
axs = axs.ravel()
for i in range(Nrep):
    for j in range(6):
        axs[j].plot(params_dim3_mecanoise[i][:,j],alpha=0.75,linewidth=1)
        axs[j].title.set_text(thetanames[j])
fig.suptitle("Mecanistic noise")  
fig.tight_layout()       
fig.savefig("realdata_mecanoise.pdf")



#####################         PREDICTIONS       ###########################
dreal_all = jnp.linspace(jnp.min(dreal[0]),jnp.max(dreal[0]))
dreal_all = jnp.tile(dreal_all,(n,1))
key_simu = jax.random.PRNGKey(1)

# Produce predictions using the model
lamb30 = jnp.exp(latent_nonoise[-1][:,0][:, None])
h30 = jnp.exp(latent_nonoise[-1][:,1][:, None])
ypred30, mypred30, varypred30 = models.model(lamb30,h30,delta=1,var_res=0,d=dreal_all,meca_noise=0.,dim=3,prng_key=key_simu)   

lamb31 = jnp.exp(latent_noise[-1][:,0][:, None])
h31 = jnp.exp(latent_noise[-1][:,1][:, None])
ypred31, mypred31, varypred31 = models.model(lamb31,h31,delta=1,var_res=0,d=dreal_all,meca_noise=1.,dim=3,prng_key=key_simu)
# when mecanistic noise is present, produce several simulations 
key_simu_31 = jax.random.split(key_simu,100)
rep31 = [models.model(lamb31,h31,delta=1,var_res=0,d=dreal_all,meca_noise=1.,dim=3,prng_key=key_simu_31[i]) for i in range(100)]
ypred31 = jnp.array([rep[0] for rep in rep31])
ypred31_min = jnp.min(ypred31,axis=0)
ypred31_max = jnp.max(ypred31,axis=0)


## Plots of individual predictions
import collections
LEGEND_HEIGHT = 0.05

fig,axs = plt.subplots(nrows=7,ncols=7,figsize=(14,14),sharey=True)
axs = axs.ravel()
for j in range(n):
	axs[j].scatter(dreal[j,:],yreal[j,:],color="black")

for i in range(n):
    axs[i].plot(dreal_all[i,:],mypred30[i,:],label="No mechanistic noise")
    axs[i].plot(dreal_all[i,:],mypred31[i,:],label="Mechanistic noise")
    axs[i].fill_between(dreal_all[i,:],ypred31_min[i,:],ypred31_max[i,:],alpha=0.25,color="tab:orange")

# Collect legend labels from all plots.
entries = collections.OrderedDict()
for ax in axs.flatten():
  for handle, label in zip(*ax.get_legend_handles_labels()):
    entries[label] = handle

# Adjust spacing between plots and make space for legend.
fig.tight_layout(rect=(0, LEGEND_HEIGHT, 1, 1), h_pad=0.5, w_pad=0.5)

# Add legend below the grid of plots.
legend = fig.legend(
    entries.values(), entries.keys(),
    loc='upper center', bbox_to_anchor=(0.5, LEGEND_HEIGHT), ncols=2, frameon=False)
fig.savefig("pred_2corrRE_3D_7x7.pdf")    



#####################         COMPUTE BIC       ###########################
import collections
ResBIC = collections.namedtuple("ResBIC",("BIC","Loglik","Var_lik"))

# theta -> array of dim nb_iter*nb_params
# latent -> array of size nb_iter*nb_params
def bic(theta,latent,meca_noise,y,d,dim,N=5000):
    n_tot = theta.shape[0]
    (n,J) = d.shape
    indiv_lik, var_lik, ll = models.log_likelihood_is(theta,latent,y,d,delta=1,N=N,meca_noise=meca_noise,dim=dim)
    bic = -2*ll + 4*jnp.log(n) + jnp.log(n*J)
    return ResBIC(bic, ll, var_lik)


bic_nonoise = bic(theta_nonoise_mean,latent_nonoise,meca_noise=0,y=yreal,d=dreal,dim=3,N=10000)
bic_noise = bic(theta_noise_mean,latent_noise,meca_noise=1,y=yreal,d=dreal,dim=3,N=10000)


# BIC on each repetition
bic_nonoise_rep = [bic(theta_nonoise[i],latent_nomecanoise[i],meca_noise=0,y=yreal,d=dreal,dim=3,N=10000).BIC for i in range(Nrep)]
bic_noise_rep = [bic(theta_noise[i],latent_mecanoise[i],meca_noise=1,y=yreal,d=dreal,dim=3,N=10000).BIC for i in range(Nrep)]


dbp = {"bic":np.concatenate([bic_nonoise_rep,bic_noise_rep]),"model":np.repeat(["No meca noise", "Meca noise"],10)}
dbp = pd.DataFrame(dbp)
sns.boxplot(data=dbp,x="model",y="bic")
plt.savefig("bic_realdata.pdf")


#####################         CONFIDENCE INTERVALS      ###########################
def scaled_params(theta):
	p = jnp.array([theta[0],theta[1],jnp.log(1+jnp.exp(theta[2])),jnp.sqrt((jnp.log(1+jnp.exp(theta[3]))**2+theta[4]**2)/2),theta[4]/jnp.sqrt(jnp.log(1+jnp.exp(theta[3]))**2+theta[4]**2),jnp.sqrt(theta[5])])
	return p


def ci(theta,fim):
	p = scaled_params(theta)
	jac = jax.jacfwd(scaled_params)(theta)
	fim_inv = jnp.linalg.inv(n*fim)
	fim_inv_p = jac @ fim_inv @ jac.T
	ci_lower = p - scipy.stats.norm.ppf(0.975) * jnp.sqrt(jnp.diag(fim_inv_p))
	ci_upper = p + scipy.stats.norm.ppf(0.975) * jnp.sqrt(jnp.diag(fim_inv_p))
	df = pd.DataFrame([thetanames,p,ci_lower,ci_upper]).T
	df.columns = ['Parameter','Estimate','CI_lower','CI_upper']
	return df


ci(theta_nonoise_mean,fim_nonoise)
ci(theta_noise_mean,fim_noise)


#####################         VISUAL PREDICTIVE CHECK      ###########################
B = 5000

# Quantiles of observed data
yreal_25 = jnp.nanquantile(yreal,axis=0,q=0.05)
yreal_50 = jnp.nanquantile(yreal,axis=0,q=0.5)
yreal_975 = jnp.nanquantile(yreal,axis=0,q=0.95)

import numpy as np
def plot_vpc(yreal,dreal,ysim,fn):
	quant = np.quantile(ysim,[0.05,0.5,0.95],axis=1)
	quant = np.quantile(quant,[0.025,0.5,0.975],axis=1)

	f, ax = plt.subplots(figsize=(8,6))
	
	for i in range(n):	
	    ax.scatter(dreal[i,:],yreal[i,:],c="black")
	ax.set_xlabel("prey density (/2L)")
	ax.set_ylabel("ingested (/2min)")
	ax.plot(dreal[0,:],yreal_25,c="tab:blue",linestyle="--")
	ax.plot(dreal[0,:],yreal_50,c="tab:orange",linestyle="--")
	ax.plot(dreal[0,:],yreal_975,c="tab:blue",linestyle="--")	
	for i in range(3):
		ax.plot(dreal[0,:],quant[1,i,:],c="tab:orange" if i==1 else "tab:blue")
		for j in range(3):
			ax.fill_between(dreal[0,:],quant[0,i,:],quant[2,i,:],alpha=0.2,color='C1' if i==1 else 'C0')
	f.tight_layout()
	f.savefig(fn)
	return ax


keyy = jax.random.split(jax.random.PRNGKey(0), B)
ysim_noise = np.array([models.simu_data(jnp.mean(theta_noise[-1000:,:],axis=0),n=n,d=dreal,delta=1,meca_noise=1,dim=3,prng_key=key)[1] for key in keyy])


plot_vpc(yreal,dreal,ysim_noise,"vpc_real_noise.pdf")

