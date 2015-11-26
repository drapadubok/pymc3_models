# -*- coding: utf-8 -*-
"""
Code for ipynb in https://github.com/drapadubok/pymc3_models
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pymc3
from sklearn.cross_validation import KFold
import scipy.stats as sp

sns.set_style('darkgrid')

dataRU = np.array(pd.read_csv('http://www.becs.aalto.fi/~smirnod1/RusKarina.csv',header=None))
dataENG = np.array(pd.read_csv('http://www.becs.aalto.fi/~smirnod1/EngKarina.csv',header=None))

print(dataRU.shape)
print(dataENG.shape)

data1 = np.zeros(np.shape(dataRU))
data2 = np.zeros(np.shape(dataENG))
data1[dataRU == 7] = 1
data1[dataRU == 1] = 1
data2[dataENG == 7] = 1
data2[dataENG == 1] = 1

S = data1.shape[1]
s_idx = [i for i in range(S)]
s = 0.01 # shape for Gamma
r = 0.01 # rate for Gamma
nsamp = 100000 # How many samples to get from MCMC
njobs = 4 # how many jobs to run in parallel
s_idx = [i for i in range(S)]

def TrainHierarchicalModel(data1, data2, s_idx, njobs=2, s=0.01, r=0.01, nsamp=10000):
    with pymc3.Model() as model:
        # define the group hyperparameters
        k_minus_two1 = pymc3.Gamma('k_minus_two1', alpha=s, beta=r)
        k1 = pymc3.Deterministic('k1', k_minus_two1 + 2)
        w1 = pymc3.Beta('w1', alpha=1, beta=1)
        k_minus_two2 = pymc3.Gamma('k_minus_two2', alpha=s, beta=r)
        k2 = pymc3.Deterministic('k2', k_minus_two2 + 2)
        w2 = pymc3.Beta('w2', alpha=1, beta=1)    
        # define the prior for group, want to tie story specific theta to it's group theta
        theta1 = pymc3.Beta('theta1', alpha=(w1 * k_minus_two1) + 1, beta=((1-w1)*k_minus_two1)+1, shape=S)
        theta2 = pymc3.Beta('theta2', alpha=(w2 * k_minus_two2) + 1, beta=((1-w2)*k_minus_two2)+1, shape=S)
        # define the likelihood
        y1 = pymc3.Bernoulli('y1', p=theta1[s_idx], observed=data1)
        y2 = pymc3.Bernoulli('y2', p=theta2[s_idx], observed=data2)
    
        wdiff = pymc3.Deterministic('wdiff', w1 - w2)
        thetadiff = pymc3.Deterministic('thetadiff', theta1 - theta2)
        oddsratio = pymc3.Deterministic('oddsratio', (theta2/(1-theta2))/(theta1/(1-theta1)))
        
        step = pymc3.Metropolis() # Instantiate MCMC sampling algorithm
        trace = pymc3.sample(nsamp/njobs, step, njobs=njobs, progressbar=False)
    return model,trace



_, trace = TrainHierarchicalModel(data1, data2, s_idx=s_idx, nsamp=nsamp, njobs=njobs)
# Remove semicolon to see the numbers
pymc3.diagnostics.gelman_rubin(trace[5000:]);
pymc3.traceplot(trace[5000:]);

# Group RU
theta1_sample = np.mean(trace[5000:].get_values('theta1',combine = True), axis=1)
w1_sample = trace[5000:].get_values('w1',combine = True)
# Group ENG
theta2_sample = np.mean(trace[5000:].get_values('theta2',combine = True), axis=1)
w2_sample = trace[5000:].get_values('w2',combine = True)
# Difference of groups
thetadiff_sample = np.mean(trace[5000:].get_values('thetadiff',combine = True), axis=1)
wdiff_sample = trace[5000:].get_values('wdiff',combine = True)
# HPDs
hpd_theta1_sample = pymc3.stats.hpd(np.array(theta1_sample))
hpd_w1_sample = pymc3.stats.hpd(np.array(w1_sample))
hpd_theta2_sample = pymc3.stats.hpd(np.array(theta2_sample))
hpd_w2_sample = pymc3.stats.hpd(np.array(w2_sample))
hpd_thetadiff = pymc3.stats.hpd(np.array(thetadiff_sample))
hpd_wdiff = pymc3.stats.hpd(np.array(wdiff_sample))
# Histogram range
histrange = list()
histrange.append(np.min(np.concatenate((theta1_sample,theta2_sample))))
histrange.append(np.max(np.concatenate((theta1_sample,theta2_sample))))
histrange_w = list()
histrange_w.append(np.min(np.concatenate((w1_sample,w2_sample))))
histrange_w.append(np.max(np.concatenate((w1_sample,w2_sample))))


fig = plt.figure(figsize=(16, 8))

# Theta1
ax1 = plt.subplot2grid((3,2), (0,0))
ax1.set_autoscaley_on(True)
bins = plt.hist(theta1_sample, histtype='stepfilled', bins=30, alpha=0.85, label="posterior of $theta_1$", color="#A60628", normed=True)         
plt.vlines(hpd_theta1_sample[0],0,max(bins[0])+1,linestyle = "--",linewidth = 1,color = 'red',label='HDI ({0:.3f} – {1:.3f}), mean={2:.2f}'.format(hpd_theta1_sample[0],hpd_theta1_sample[1],np.mean(theta1_sample)))
plt.vlines(hpd_theta1_sample[1],0,max(bins[0])+1,linestyle = "--",linewidth = 1,color = 'red')
plt.title(r"""Posterior distributions of the variables
    $\theta_1 (RU),\;\theta_2 (ENG),\;Difference\ of\ thetas$""")
plt.legend(loc="upper right")
plt.xlim(histrange)

# Theta2
ax2 = plt.subplot2grid((3,2), (1,0))
ax2.set_autoscaley_on(True)
bins = plt.hist(theta2_sample, histtype='stepfilled', bins=30, alpha=0.85, label="posterior of $theta_2$", color="#7A68A6", normed=True)
plt.vlines(hpd_theta2_sample[0],0,max(bins[0])+1,linestyle = "--",linewidth = 1,color = 'purple',label='HDI ({0:.3f} – {1:.3f}), mean={2:.2f}'.format(hpd_theta2_sample[0],hpd_theta2_sample[1],np.mean(theta2_sample)))
plt.vlines(hpd_theta2_sample[1],0,max(bins[0])+1,linestyle = "--",linewidth = 1,color = 'purple')
plt.legend(loc="upper left")
plt.xlim(histrange)

# Thetadiff
ax3 = plt.subplot2grid((3,2), (2,0))
bins = plt.hist(thetadiff_sample, histtype='stepfilled', bins=30, alpha=0.85, label=r"posterior of $thetadiff$", color="#467821")
plt.vlines(hpd_thetadiff[0],0,max(bins[0])+1,linestyle = "--",linewidth = 1,color = 'green',label='HDI ({0:.3f} – {1:.3f}), mean={2:.2f}'.format(hpd_thetadiff[0],hpd_thetadiff[1],np.mean(thetadiff_sample)))
plt.vlines(hpd_thetadiff[1],0,max(bins[0])+1,linestyle = "--",linewidth = 1,color = 'green')
plt.vlines(0,0,max(bins[0])+1,linestyle = "-",linewidth = 2,color = 'green')
plt.legend(loc="upper right")
plt.xlim([min(thetadiff_sample),0.1]);

# Omega1
ax4 = plt.subplot2grid((3,2), (0,1))
ax4.set_autoscaley_on(True)
bins = plt.hist(w1_sample, histtype='stepfilled', bins=30, alpha=0.85, label="posterior of $\omega_1$", color="#A60628", normed=True)         
plt.vlines(hpd_w1_sample[0],0,max(bins[0])+1,linestyle = "--",linewidth = 1,color = 'red',label='HDI ({0:.3f} – {1:.3f}), mean={2:.2f}'.format(hpd_w1_sample[0],hpd_w1_sample[1],np.mean(w1_sample)))
plt.vlines(hpd_w1_sample[1],0,max(bins[0])+1,linestyle = "--",linewidth = 1,color = 'red')
plt.title(r"""Posterior distributions of the variables
    $\omega_1 (RU),\;\omega_2 (ENG),\;Difference\ of\ omegas$""")
plt.legend(loc="upper left")
plt.xlim(histrange_w)

# Omega2
ax5 = plt.subplot2grid((3,2), (1,1))
ax5.set_autoscaley_on(True)
bins = plt.hist(w2_sample, histtype='stepfilled', bins=30, alpha=0.85, label="posterior of $\omega_2$", color="#7A68A6", normed=True)
plt.vlines(hpd_w2_sample[0],0,max(bins[0])+1,linestyle = "--",linewidth = 1,color = 'purple',label='HDI ({0:.3f} – {1:.3f}), mean={2:.2f}'.format(hpd_w2_sample[0],hpd_w2_sample[1],np.mean(w2_sample)))
plt.vlines(hpd_w2_sample[1],0,max(bins[0])+1,linestyle = "--",linewidth = 1,color = 'purple')
plt.legend(loc="upper left")
plt.xlim(histrange_w)

# Omegadiff
ax6 = plt.subplot2grid((3,2), (2,1))
bins = plt.hist(wdiff_sample, histtype='stepfilled', bins=30, alpha=0.85, label=r"posterior of $wdiff$", color="#467821")
plt.vlines(hpd_wdiff[0],0,max(bins[0])+1,linestyle = "--",linewidth = 1,color = 'green',label='HDI ({0:.3f} – {1:.3f}), mean={2:.2f}'.format(hpd_wdiff[0],hpd_wdiff[1],np.mean(wdiff_sample)))
plt.vlines(hpd_wdiff[1],0,max(bins[0])+1,linestyle = "--",linewidth = 1,color = 'green')
plt.vlines(0,0,max(bins[0])+1,linestyle = "-",linewidth = 2,color = 'green')
plt.legend(loc="upper left")
plt.xlim([min(wdiff_sample),0.01]);


np.mean(wdiff_sample<0)

oddsratio = np.mean(trace[5000:].get_values('oddsratio',combine=True),axis=1)

fig = plt.figure(figsize=(8, 4));
plt.hist(oddsratio, label="posterior of $Odds ratio$,\n mean={0}".format(np.mean(oddsratio)), normed=True);  
plt.title(r"""Posterior distribution of the group-wise odds ratio between RU and ENG""");
plt.legend(loc='upper right');

w_oddsratio = (w2_sample/(1-w2_sample))/(w1_sample/(1-w1_sample))

fig = plt.figure(figsize=(8, 4));
plt.hist(w_oddsratio, label="posterior of $Odds ratio$,\n mean={0}".format(np.mean(w_oddsratio)), normed=True);  
plt.title(r"""Posterior distribution of the group-wise odds ratio between RU and ENG""");
plt.legend(loc='upper right');


nsim = 500 # How many simulated datasets to get
n_folds = 5 # How many folds for cross validation
# Use KFold partitioner from sklearn to create cross-validation indices
cv1 = list(KFold(data1.shape[0], n_folds=n_folds))
cv2 = list(KFold(data2.shape[0], n_folds=n_folds))

prop_1 = np.empty((S,n_folds),dtype=float) # Proportion of extreme answers for each story in Russian group
prop_2 = np.empty((S,n_folds),dtype=float) # Proportion of extreme answers for each story in English group
prop_yrep1 = np.empty((S,n_folds),dtype=float) # Proportions for Russian group in simulated datasets
prop_yrep2 = np.empty((S,n_folds),dtype=float) # Proportions for English group in simulated datasets

from collections import defaultdict

def run_ppc(trace, samples=100, model=None):
    """Generate Posterior Predictive samples from a model given a trace.
    """
    if model is None:
         model = pymc3.modelcontext(model)

    ppc = defaultdict(list)
    for idx in np.random.randint(0, len(trace), samples):
        param = trace[idx]
        for obs in model.observed_RVs:
            ppc[obs.name].append(obs.distribution.random(point=param))

    return ppc

for fold in range(n_folds):
    # Datasets are of unequal size, so we can't use single cv object
    data1_train = data1[cv1[fold][0],:]
    data1_test = data1[cv1[fold][1],:]
    data2_train = data2[cv2[fold][0],:]
    data2_test = data2[cv2[fold][1],:]
    # Store number of left out samples to know
    n1 = data1_test.shape[0]
    n2 = data2_test.shape[0]
    # Run the chain
    model, trace = TrainHierarchicalModel(data1_train, data2_train, s_idx=s_idx, nsamp=10000, njobs=1)
    # Get simulated datasets
    ppc = run_ppc(trace, samples=nsim, model=model)
    yrep1 = np.asarray(ppc['y1'])
    yrep2 = np.asarray(ppc['y2'])
    # Calculate statistics
    prop_yrep1[:,fold] = np.array([(yrep1[:,s]==1).mean() for s in range(S)])
    prop_yrep2[:,fold] = np.array([(yrep2[:,s]==1).mean() for s in range(S)])
    prop_1[:,fold] = np.array([(data1_test[:,s]==1).mean() for s in range(S)])
    prop_2[:,fold] = np.array([(data2_test[:,s]==1).mean() for s in range(S)])   

prop_1 = np.mean(prop_1,1)
prop_2 = np.mean(prop_2,1)
prop_yrep1 = np.mean(prop_yrep1,1)
prop_yrep2 = np.mean(prop_yrep2,1)


fig = plt.figure(figsize=(16, 8));
plt.plot(prop_yrep1,'#ff6969', alpha=1, lw=3., linestyle = '--', label='Simulated proportion of extreme answers, RU')
plt.plot(prop_1,'#9a0000', alpha=1, lw=3., label='Proportion of extreme answers, RU')

plt.plot(prop_yrep2,'#94b2fe',alpha=1, lw=3., linestyle = '--', label='Simulated proportion of extreme answers, ENG')
plt.plot(prop_2,'#4455dd', alpha=1, lw=3., label='Proportion of extreme answers, ENG')
plt.legend(loc="upper right");





