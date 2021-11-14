# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:30:54 2021

@author: svenb

In this skript, everything that was done to evaluate the estimation results for\
    Franke & Westerhoff I model is included.

-Testing the results with KS-tests
-acceptance rate
-evaluating the normal distribtion of the SumStats
-testing other distributions for the SumStats
"""
import numpy as np
import pandas as pd
import os
import Shiller_ex_ante_price
from scipy.stats import kstest,ks_2samp, powerlaw, epps_singleton_2samp
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
import FrankWesterhoffI
import internal_func_FWI_SumStats_MK1_log


#set working directory to foulder Results
os.chdir(r'C:\Users\svenb\Master\Studium\Material\Masterarbeit\ABM-Masterarbeit\Fine_run_MK3\Results')

FWI_results_log_fine=np.array(pd.read_csv('FWI_results_log_fine.csv'))

FWI_Likelihood_log_fine=np.array(pd.read_csv('FWI_Likelihood_log_fine.csv'))
pd.plotting.autocorrelation_plot(FWI_Likelihood_log_fine)
FWI_Likelihood_prop_log_fine=np.array(pd.read_csv('FWI_Likelihood_prop_log_fine.csv'))
pd.plotting.autocorrelation_plot(FWI_Likelihood_prop_log_fine)


#Plot of the Likelihood of the Franke Westerhoff Model
plt.plot(FWI_Likelihood_log_fine, color='#0504aa')
plt.xlabel('Chain Steps')
plt.ylabel('Likelihood')
plt.title('Graph of the FWI Model Likelihood')
plt.show()



###############################################################################
###############################################################################
#KS-Test for FWI with SumStats_MK1

x_FWI_0 = FrankWesterhoffI.FrankeWestI(T, FWI_results_log_fine[0], funda, y)
x_FWI = FrankWesterhoffI.FrankeWestI(T, FWI_results_log_fine[49998], funda, y)

#Performing a Kolmogorov-Smirnov test for the baseline model with alternative SumStats
print(ks_2samp(x_FWI,y))
print(ks_2samp(x_FWI_0,y))

#average of 50 simulations
x_i = np.zeros(shape=(T,50))
for i in range(0,50):
    x_i[:,i] = FrankWesterhoffI.FrankeWestI(T, FWI_results_log_fine[49998], funda, y)
x_average_FWI = x_i.sum(axis=1)/50

#average of 50 simulations
x_i_0 = np.zeros(shape=(T,50))
for i in range(0,50):
    x_i_0[:,i] = FrankWesterhoffI.FrankeWestI(T, FWI_results_log_fine[0], funda, y)
x_average_FWI_0 = x_i_0.sum(axis=1)/50

print(ks_2samp(x_average_FWI,y))
print(ks_2samp(x_average_FWI_0,y))

dx_FWI = np.diff(x_FWI)
dx_FWI_0 = np.diff(x_FWI_0)
dy = np.diff(y)
print(ks_2samp(dx_FWI,dy))
print(ks_2samp(dx_FWI_0,dy))

dx_average_FWI = np.diff(x_average_FWI)
dx_average_FWI_0 = np.diff(x_average_FWI_0)
print(ks_2samp(dx_average_FWI,dy))
print(ks_2samp(dx_average_FWI_0,dy))

###############################################################################
#logged time series
x_FWI_log = np.log(x_FWI)
x_FWI_log_0 = np.log(x_FWI_0)

dx_FWI_log = np.diff(x_FWI_log)
dx_FWI_log_0 = np.diff(x_FWI_log_0)

y_log = np.log(y)
dy_log = np.diff(y_log)
print(ks_2samp(x_FWI_log,y_log))
print(ks_2samp(dx_FWI_log,dy_log))
print(ks_2samp(x_FWI_log_0,y_log))
print(ks_2samp(dx_FWI_log_0,dy_log))

x_average_FWI_log = np.log(x_average_FWI)
x_average_FWI_log_0 = np.log(x_average_FWI_0)
dx_average_FWI_log = np.diff(x_average_FWI_log)
dx_average_FWI_log_0 = np.diff(x_average_FWI_log_0)

print(ks_2samp(x_average_FWI_var_log,y_log))
print(ks_2samp(dx_average_FWI_var_log,dy_log))
print(ks_2samp(x_average_FWI_var_log_0,y_log))
print(ks_2samp(dx_average_FWI_var_log_0,dy_log))

###############################################################################
#acceptance rate of the propsals
unique, counts = np.unique(FWI_results_log_fine, return_counts=True)
dict(zip(unique, counts))
updates = len(unique)/7#len(theta_0)
print(updates)
accaptence_rate = updates/50000
print(accaptence_rate)

##############################################################################
#normallity check of the SumStats for F&W with standard SumStats
##############################################################################
M=1
s_para=Parallel(n_jobs=-1)(delayed(internal_func_FWI_SumStats_MK1_log.syn_loop)(T,M,FWI_results_log_fine[49998],funda,y) for _ in range(0,1000))
s = np.asarray(s_para)
kstest(s[:,0],'norm',N=1000)
kstest(s[:,1], 'norm',N=1000)
kstest(s[:,2], 'norm',N=1000)
kstest(s[:,3], 'norm',N=1000)
kstest(s[:,4], 'norm',N=1000)
kstest(s[:,5], 'norm',N=1000)
kstest(s[:,6], 'norm',N=1000)
kstest(s[:,7], 'norm',N=1000)
kstest(s[:,8], 'norm',N=1000)
kstest(s[:,9], 'norm',N=1000)
kstest(s[:,10], 'norm',N=1000)
kstest(s[:,11], 'norm',N=1000)
kstest(s[:,12], 'norm',N=1000)
kstest(s[:,13], 'norm',N=1000)
kstest(s[:,14], 'norm',N=1000)
kstest(s[:,15], 'norm',N=1000)
kstest(s[:,16], 'norm',N=1000)
kstest(s[:,17], 'norm',N=1000)
kstest(s[:,18], 'norm',N=1000)
kstest(s[:,19], 'norm',N=1000)



###############################################################################
#Historgrams of the SumStats draws 
###############################################################################


n, bins, patches = plt.hist(x=s[:,0], bins='auto', color='#0504aa')
#plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the first SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
#plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s[:,1], bins='auto', color='#0504aa',
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
n, bins, patches = plt.hist(x=s[:,2], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the third SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s[:,3], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the fourth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s[:,4], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the fifth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s[:,5], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the sixth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


###############################################################################
n, bins, patches = plt.hist(x=s[:,6], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the seventh SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s[:,7], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the eighth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s[:,8], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the ninth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s[:,9], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the tenth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s[:,10], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the eleventh SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s[:,11], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the twelfth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s[:,12], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the thirteenth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s[:,13], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the fourteen SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


###############################################################################
n, bins, patches = plt.hist(x=s[:,14], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the fiftieth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s[:,15], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the sixteenth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s[:,16], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the seventeenth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s[:,17], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the eighteenth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s[:,18], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the nineteenth SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s[:,19], bins='auto', color='#0504aa')
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
###############################################################################
'Here the idea was followed that perhaps a logarithmic and sorted diagram could\
    give more information about the distribution. It was not found to be helpful in the study.'

Sumstats_sort_0=np.log(np.sort(s[:,0]))
%varexp --plot Sumstats_sort_0
Sumstats_sort_1=np.log(np.sort(s[:,1]))
%varexp --plot Sumstats_sort_1

Sumstats_sort_2=np.log(np.sort(s[:,2]))
%varexp --plot Sumstats_sort_2

Sumstats_sort_3=np.log(np.sort(s[:,3]))
%varexp --plot Sumstats_sort_3

Sumstats_sort_4=np.log(np.sort(s[:,4]))
%varexp --plot Sumstats_sort_4

Sumstats_sort_5=np.log(np.sort(s[:,5]))
%varexp --plot Sumstats_sort_5

Sumstats_sort_6=np.log(np.sort(s[:,6]))
%varexp --plot Sumstats_sort_6

Sumstats_sort_7=np.log(np.sort(s[:,7]))
%varexp --plot Sumstats_sort_7

Sumstats_sort_8=np.log(np.sort(s[:,8]))
%varexp --plot Sumstats_sort_8

Sumstats_sort_9=np.log(np.sort(s[:,9]))
%varexp --plot Sumstats_sort_9

Sumstats_sort_10=np.log(np.sort(s[:,10]))
%varexp --plot Sumstats_sort_10

Sumstats_sort_11=np.log(np.sort(s[:,11]))
%varexp --plot Sumstats_sort_11

Sumstats_sort_12=np.log(np.sort(s[:,12]))
%varexp --plot Sumstats_sort_12

Sumstats_sort_13=np.log(np.sort(s[:,13]))
%varexp --plot Sumstats_sort_13

Sumstats_sort_14=np.log(np.sort(s[:,14]))
%varexp --plot Sumstats_sort_14

Sumstats_sort_15=np.log(np.sort(s[:,15]))
%varexp --plot Sumstats_sort_15

Sumstats_sort_16=np.log(np.sort(s[:,16]))
%varexp --plot Sumstats_sort_16

Sumstats_sort_17=np.log(np.sort(s[:,17]))
%varexp --plot Sumstats_sort_17

Sumstats_sort_18=np.log(np.sort(s[:,18]))
%varexp --plot Sumstats_sort_18

Sumstats_sort_19=np.log(np.sort(s[:,19]))
%varexp --plot Sumstats_sort_19

###############################################################################
'The SumStats do also not follow a powerlaw distirbution.'
#mean, var, skew, kurt = powerlaw.stats(s[:,0], moments='mvsk')
kstest(s[:,0], 'powerlaw',args=(1000,1.6),N=1000)
kstest(s[:,1], 'powerlaw',args=(1000,1.6),N=1000)# 
kstest(s[:,2], 'powerlaw',args=(1000,1.6),N=1000)# 
kstest(s[:,3], 'powerlaw',args=(1000,1.6),N=1000)# 
kstest(s[:,4], 'powerlaw',args=(1000,1.6),N=1000)# 
kstest(s[:,5], 'powerlaw',args=(1000,1.6),N=1000)# 
kstest(s[:,6], 'powerlaw',args=(1000,1.6),N=1000)# 

###############################################################################
'Further tests of distributions for the first SumStat. \
 H1 had to be assumed for all of them, which means that none of the theoretical\
     distributions matched the empirical one.

kstest(np.log(s[:,0]), 'powerlaw',args=(1000,2.5),N=1000)
#14 is the number of degrees of freedome (22-8)=14
kstest(s[:,0], 't',args=(10,1),N=1000)
kstest(s[:,0], 'nct',args=(10,1),N=1000)
kstest(s[:,0], 'chi',args=(0.1,100),N=1000)# 
kstest(s[:,0], 'chi2',args=(0.1,100),N=1000)# 
kstest(s[:,0], 'genpareto',args=(0.1,100),N=1000)# 
kstest(s[:,0], 'gamma',args=(0.1,100),N=1000)# 
#the 0.245 is test by hand and seems to be a optimum
kstest(s[:,0], 'pareto',args=(0.245,1),N=1000)# 
kstest(s[:,0], 'uniform',args=(0,1000),N=1000)
kstest(s[:,0], 'cauchy',args=(0,5),N=1000)

s_frequent= np.unique(s[:,0], return_counts=False)
###############################################################################
