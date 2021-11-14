# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:55:47 2021

@author: svenb

In this skript, everything that was done to evaluate the estimation results for\
    Baseline model is included.

-Testing the results with KS-tests
-acceptance rate
-evaluating the normal distribtion of the SumStats
-testing other distributions for the SumStats
"""
import numpy as np
import pandas as pd
import os
import Shiller_ex_ante_price
from scipy.stats import kstest,ks_2samp
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import SumStats_MK1_log

Data_Lux=pd.read_excel(r'C:\Users\svenb\Master\Studium\Material\Masterarbeit\Shiller_data.xls')
#D sind die discontierten Dividenden'
D = np.array(Data_Lux['D'])
y = np.array(Data_Lux['P'])
Date = np.array(Data_Lux['Date'])
T =len(y)
M=50

funda = Shiller_ex_ante_price.fundamental(D,T)[0:1800]
ssy =  np.asarray(SumStats_MK1_log.sumstats(y, funda, T, M))

###############################################################################
import Basic_Model
import internal_func_Baseline_log

#set working directory to foulder Results
os.chdir(r'C:\Users\svenb\Master\Studium\Material\Masterarbeit\ABM-Masterarbeit\Fine_run_MK3\Results')
Baseline_results_log_fine=np.array(pd.read_csv('Baseline_results_log_fine.csv'))

Baseline_Likelihood_log_fine=np.array(pd.read_csv('Baseline_Likelihood_log_fine.csv'))
pd.plotting.autocorrelation_plot(Baseline_Likelihood_log_fine)
Baseline_Likelihood_prop_log_fine=np.array(pd.read_csv('Baseline_Likelihood_prop_log_fine.csv'))
pd.plotting.autocorrelation_plot(Baseline_Likelihood_prop_log_fine)


###############################################################################
#Plot of the Likelihood of the  Baseline Model
plt.plot(Baseline_Likelihood_log_fine, color='#0504aa')
plt.xlabel('Chain Steps')
plt.ylabel('Likelihood')
plt.title('Graph of the Baseline Model Likelihood')
plt.show()


################################################################################
#KS-Test for Baseline with SumStats-MK1
###############################################################################
x_Baseline = Basic_Model.Basic_Model(Baseline_results_log_fine[49998], T, funda, y)
x_Baseline_0 = Basic_Model.Basic_Model(Baseline_results_log_fine[0], T, funda, y)

#Performing a Kolmogorov-Smirnov test for the baseline model with alternative SumStats
print(ks_2samp(x_Baseline,y))
print(ks_2samp(x_Baseline_0,y))

#average of 50 simulations
x_i = np.zeros(shape=(T,50))
for i in range(0,50):
    x_i[:,i] = Basic_Model.Basic_Model(Baseline_results_log_fine[49998], T, funda, y)
x_average_Baseline = x_i.sum(axis=1)/50

#average of 50 simulations
x_i_0 = np.zeros(shape=(T,50))
for i in range(0,50):
    x_i_0[:,i] = Basic_Model.Basic_Model(Baseline_results_log_fine[0], T, funda, y)
x_average_Baseline_0 = x_i_0.sum(axis=1)/50

#Performing a Kolmogorov-Smirnov test for the baseline model with alternative SumStats
print(ks_2samp(x_average_Baseline,y))
print(ks_2samp(x_average_Baseline_0,y))

#retunrs
dx_Baseline = np.diff(x_Baseline)
dx_Baseline_0 = np.diff(x_Baseline_0)
dy = np.diff(y)
print(ks_2samp(dx_Baseline,dy))
print(ks_2samp(dx_Baseline_0,dy))

#average returns
dx_average_Baseline = np.diff(x_average_Baseline)
dx_average_Baseline_0 = np.diff(x_average_Baseline_0)
print(ks_2samp(dx_average_Baseline,dy))
print(ks_2samp(dx_average_Baseline_0,dy))

###############################################################################
#logged time series
x_Baseline_log = np.log(x_Baseline)
x_Baseline_log_0 = np.log(x_Baseline_0)

dx_Baseline_log = np.diff(x_Baseline_log)
dx_Baseline_log_0 = np.diff(x_Baseline_log_0)

y_log = np.log(y)
dy_log = np.diff(y_log)
print(ks_2samp(x_Baseline_log,y_log))
print(ks_2samp(dx_Baseline_log,dy_log))
print(ks_2samp(x_Baseline_log_0,y_log))
print(ks_2samp(dx_Baseline_log_0,dy_log))

x_average_Baseline_log = np.log(x_average_Baseline)
x_average_Baseline_log_0 = np.log(x_average_Baseline_0)
dx_average_Baseline_log = np.diff(x_average_Baseline_log)
dx_average_Baseline_log_0 = np.diff(x_average_Baseline_log_0)

print(ks_2samp(x_average_Baseline_log,y_log))
print(ks_2samp(dx_average_Baseline_log,dy_log))
print(ks_2samp(x_average_Baseline_log_0,y_log))
print(ks_2samp(dx_average_Baseline_log_0,dy_log))


###############################################################################
#acceptance rate of the propsals
unique, counts = np.unique(Baseline_results_log_fine, return_counts=True)
dict(zip(unique, counts))
updates = len(unique)/7#len(theta_0)
print(updates)
accaptence_rate = updates/50000
print(accaptence_rate)

theta_1_vector=Baseline_results_log_fine[:,0]

###############################################################################
#Normality test for the SumStats with estimated parameter
###############################################################################
M=1
s_para_Baseline=Parallel(n_jobs=-1)(delayed(internal_func_Baseline_log.syn_loop)(T,M,Baseline_results_log_fine[49998],funda,y) for _ in range(0,1000))
s_B= np.asarray(s_para_Baseline)
kstest(s_B[:,0],'norm',N=1000)
kstest(s_B[:,1], 'norm',N=1000)
kstest(s_B[:,2], 'norm',N=1000)
kstest(s_B[:,3], 'norm',N=1000)
kstest(s_B[:,4], 'norm',N=1000)
kstest(s_B[:,5], 'norm',N=1000)
kstest(s_B[:,6], 'norm',N=1000)
kstest(s_B[:,7], 'norm',N=1000)
kstest(s_B[:,8], 'norm',N=1000)
kstest(s_B[:,9], 'norm',N=1000)
kstest(s_B[:,10], 'norm',N=1000)
kstest(s_B[:,11], 'norm',N=1000)
kstest(s_B[:,12], 'norm',N=1000)
kstest(s_B[:,13], 'norm',N=1000)
kstest(s_B[:,14], 'norm',N=1000)
kstest(s_B[:,15], 'norm',N=1000)
kstest(s_B[:,16], 'norm',N=1000)
kstest(s_B[:,17], 'norm',N=1000)
kstest(s_B[:,18], 'norm',N=1000)
kstest(s_B[:,19], 'norm',N=1000)


###############################################################################
#Historgrams of the SumStats draws 
###############################################################################

n, bins, patches = plt.hist(x=s_B[:,0], bins='auto', color='#0504aa')
#plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the first SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
#plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_B[:,1], bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the second SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_B[:,2], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the third SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_B[:,3], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the fourth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_B[:,4], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the fifth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_B[:,5], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the sixth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


###############################################################################
n, bins, patches = plt.hist(x=s_B[:,6], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the seventh SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_B[:,7], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the eighth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_B[:,8], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the ninth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_B[:,9], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the tenth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_B[:,10], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the eleventh SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_B[:,11], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the twelfth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_B[:,12], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the thirteenth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_B[:,13], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the fourteen SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


###############################################################################
n, bins, patches = plt.hist(x=s_B[:,14], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the fiftieth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_B[:,15], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the sixteenth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_B[:,16], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the seventeenth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_B[:,17], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the eighteenth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_B[:,18], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the nineteenth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_B[:,19], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the twentieth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
###############################################################################

###############################################################################
#test for different distributions 
###############################################################################
"""Further tests of distributions for the first SumStat. \
 H1 had to be assumed for all of them, which means that none of the theoretical\
     distributions matched the empirical one."""

kstest(np.log(s_B[:,0]), 'powerlaw',args=(1000,2.5),N=1000)
#14 is the number of degrees of freedome (22-8)=14
kstest(s_B[:,0], 't',args=(10,1),N=1000)
kstest(s_B[:,0], 'nct',args=(10,1),N=1000)
kstest(s_B[:,0], 'chi',args=(0.1,100),N=1000)# 
kstest(s_B[:,0], 'chi2',args=(0.1,100),N=1000)# 
kstest(s_B[:,0], 'genpareto',args=(0.1,100),N=1000)# 
kstest(s_B[:,0], 'gamma',args=(0.1,100),N=1000)# 
#the 0.245 is test by hand and seems to be a optimum
kstest(s_B[:,0], 'pareto',args=(0.245,1),N=1000)# 
kstest(s_B[:,0], 'uniform',args=(0,1000),N=1000)
kstest(s_B[:,0], 'cauchy',args=(0,5),N=1000)

s_frequent= np.unique(s_B[:,0], return_counts=False)
