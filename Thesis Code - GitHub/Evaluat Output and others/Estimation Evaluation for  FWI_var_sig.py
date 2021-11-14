# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:34:16 2021

@author: svenb

In this skript, everything that was done to evaluate the estimation results for\
    Franke & Westerhoff I model with variable sigma is included.

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
import internal_func_FWI_var_sig_log
import FrankWesterhoffI_var_sig

#set working directory to foulder Results
os.chdir(r'C:\Users\svenb\Master\Studium\Material\Masterarbeit\ABM-Masterarbeit\Fine_run_MK3\Results')

FWI_results_var_sig_log_fine=np.array(pd.read_csv('FWI_results_var_sig_log_fine.csv'))

FWI_var_sig_Likelihood_log_fine=np.array(pd.read_csv('FWI_var_sig_Likelihood_log_fine.csv'))
pd.plotting.autocorrelation_plot(FWI_var_sig_Likelihood_log_fine)
FWI_var_sig_Likelihood_prop_log_fine=pd.read_csv('FWI_var_sig_Likelihood_prop_log_fine.csv')

#Plot of the Likelihood of the Franke Westerhoff variable sigma Model
plt.plot(FWI_var_sig_Likelihood_log_fine, color='#0504aa')
plt.xlabel('Chain Steps')
plt.ylabel('Likelihood')
plt.title('Graph of the FWI-var-sig Model Likelihood')
plt.show()

###############################################################################
###############################################################################

#KS-Test for Baseline with alternative SumStats
x_FWI_var = FrankWesterhoffI_var_sig.FrankeWestI(T, FWI_results_var_sig_log_fine[45000], funda, y)
x_FWI_var_0 = FrankWesterhoffI_var_sig.FrankeWestI(T, FWI_results_var_sig_log_fine[0], funda, y)

#Performing a Kolmogorov-Smirnov test for the baseline model with alternative SumStats
print(ks_2samp(x_FWI_var,y))
print(ks_2samp(x_FWI_var_0,y))

#average of 50 simulations
x_i = np.zeros(shape=(T,50))
for i in range(0,50):
    x_i[:,i] = FrankWesterhoffI_var_sig.FrankeWestI(T, FWI_results_var_sig_log_fine[45000], funda, y)
x_average_FWI_var = x_i.sum(axis=1)/50

print(ks_2samp(x_average_FWI_var,y))


#average of 50 simulations
x_i_0 = np.zeros(shape=(T,50))
for i in range(0,50):
    x_i_0[:,i] = FrankWesterhoffI_var_sig.FrankeWestI(T, FWI_results_var_sig_log_fine[0], funda, y)
x_average_FWI_var_0 = x_i_0.sum(axis=1)/50

print(ks_2samp(x_average_FWI_var_0,y))

#For the returns of the simulations
dx_FWI_var = np.diff(x_FWI_var)
dy = np.diff(y)
print(ks_2samp(dx_FWI_var,dy))

dx_average_FWI_var = np.diff(x_average_FWI_var)
print(ks_2samp(dx_average_FWI_var,dy))

dx_FWI_var_0 = np.diff(x_FWI_var_0)
print(ks_2samp(dx_FWI_var_0,dy))

dx_average_FWI_var_0 = np.diff(x_average_FWI_var_0)
print(ks_2samp(dx_average_FWI_var_0,dy))

###############################################################################
#acceptance rate of the propsals
unique, counts = np.unique(FWI_results_var_sig_log_fine, return_counts=True)
dict(zip(unique, counts))
updates = len(unique)/7#len(theta_0)
print(updates)
accaptence_rate = updates/50000
print(accaptence_rate)

###############################################################################
#logged time series
x_FWI_var_log = np.log(x_FWI_var)
x_FWI_var_log_0 = np.log(x_FWI_var_0)

dx_FWI_var_log = np.diff(x_FWI_var_log)
dx_FWI_var_log_0 = np.diff(x_FWI_var_log_0)

y_log = np.log(y)
dy_log = np.diff(y_log)
print(ks_2samp(x_FWI_var_log,y_log))
print(ks_2samp(dx_FWI_var_log,dy_log))
print(ks_2samp(x_FWI_var_log_0,y_log))
print(ks_2samp(dx_FWI_var_log_0,dy_log))

x_average_FWI_var_log = np.log(x_average_FWI_var)
x_average_FWI_var_log_0 = np.log(x_average_FWI_var_0)
dx_average_FWI_var_log = np.diff(x_average_FWI_var_log)
dx_average_FWI_var_log_0 = np.diff(x_average_FWI_var_log_0)

print(ks_2samp(x_average_FWI_var_log,y_log))
print(ks_2samp(dx_average_FWI_var_log,dy_log))
print(ks_2samp(x_average_FWI_var_log_0,y_log))
print(ks_2samp(dx_average_FWI_var_log_0,dy_log))

###############################################################################
#normallity check of the SumStats for FWI var sig with SumStats-MK1
###############################################################################
M=1
s_para_FWI=Parallel(n_jobs=-1)(delayed(internal_func_FWI_var_sig_log.syn_loop)(T,M,FWI_results_var_sig_log_fine[45000],funda,y) for _ in range(0,1000))
s_FWI = np.asarray(s_para_FWI)
kstest(s_FWI[:,0],'norm',N=1000)
kstest(s_FWI[:,1], 'norm',N=1000)
kstest(s_FWI[:,2], 'norm',N=1000)
kstest(s_FWI[:,3], 'norm',N=1000)
kstest(s_FWI[:,4], 'norm',N=1000)
kstest(s_FWI[:,5], 'norm',N=1000)
kstest(s_FWI[:,6], 'norm',N=1000)
kstest(s_FWI[:,7], 'norm',N=1000)
kstest(s_FWI[:,8], 'norm',N=1000)
kstest(s_FWI[:,9], 'norm',N=1000)
kstest(s_FWI[:,10], 'norm',N=1000)
kstest(s_FWI[:,11], 'norm',N=1000)
kstest(s_FWI[:,12], 'norm',N=1000)
kstest(s_FWI[:,13], 'norm',N=1000)
kstest(s_FWI[:,14], 'norm',N=1000)
kstest(s_FWI[:,15], 'norm',N=1000)
kstest(s_FWI[:,16], 'norm',N=1000)
kstest(s_FWI[:,17], 'norm',N=1000)
kstest(s_FWI[:,18], 'norm',N=1000)
kstest(s_FWI[:,19], 'norm',N=1000)

###############################################################################
#Historgrams of the SumStats draws 
###############################################################################

n, bins, patches = plt.hist(x=s_FWI[:,0], bins='auto', color='#0504aa')
#plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the first SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
#plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FWI[:,1], bins='auto', color='#0504aa',
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
n, bins, patches = plt.hist(x=s_FWI[:,2], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the third SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FWI[:,3], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the fourth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FWI[:,4], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the fifth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


###############################################################################
n, bins, patches = plt.hist(x=s_FWI[:,5], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the sixth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


###############################################################################
n, bins, patches = plt.hist(x=s_FWI[:,6], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the seventh SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FWI[:,7], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the eighth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FWI[:,8], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the ninth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FWI[:,9], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the tenth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FWI[:,10], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the eleventh SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FWI[:,11], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the twelfth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FWI[:,12], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the thirteenth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FWI[:,13], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the fourteen SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


###############################################################################
n, bins, patches = plt.hist(x=s_FWI[:,14], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the fiftieth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FWI[:,15], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the sixteenth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FWI[:,16], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the seventeenth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FWI[:,17], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the eighteenth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FWI[:,18], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the nineteenth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FWI[:,19], bins='auto', color='#0504aa')
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

kstest(np.log(s_FWI[:,0]), 'powerlaw',args=(1000,2.5),N=1000)
#14 is the number of degrees of freedome (22-8)=14
kstest(s_FWI[:,0], 't',args=(10,1),N=1000)
kstest(s_FWI[:,0], 'nct',args=(10,1),N=1000)
kstest(s_FWI[:,0], 'chi',args=(0.1,100),N=1000)# 
kstest(s_FWI[:,0], 'chi2',args=(0.1,100),N=1000)# 
kstest(s_FWI[:,0], 'genpareto',args=(0.1,100),N=1000)# 
kstest(s_FWI[:,0], 'gamma',args=(0.1,100),N=1000)# 
kstest(s_FWI[:,0], 'pareto',args=(0.245,1),N=1000)# 
kstest(s_FWI[:,0], 'uniform',args=(0,1000),N=1000)
kstest(s_FWI[:,0], 'cauchy',args=(0,5),N=1000)

s_frequent= np.unique(s_FWI[:,0], return_counts=False)