# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 09:46:17 2021

@author: svenb

In this skript some of the other sets of Sumstats were tested on normality
"""



###############################################################################
###############################################################################
###############################################################################
#The same Test for normality was again conducted for the alternativ SumStats.F&W_I
#Check for the alternative SumStats

import 
theta_BSL_FW_alternative_50000=pd.read_csv('F&WI_results_alternative_SumStats (1).csv')
theta_BSL_FW_alternative_50000=np.array(theta_BSL_FW_alternative_50000)

theta_1_BSL_FW_alternative_50000= theta_BSL_FW_alternative_50000[:,0]

x_FWI_alternative = FrankWesterhoffI.FrankeWestI(T,theta_BSL_FW_alternative_50000[49998],funda,y)


print(ks_2samp(x_FWI_alternative,y))
#print(epps_singleton_2samp(x_FWI,y))

s_FW_para=Parallel(n_jobs=-1)(delayed(internal_func_FWI_alternativeSumStats.syn_loop)(T,M,theta_BSL_FW_alternative_50000[49998],funda,y) for _ in range(0,M))
s_FW = np.asarray(s_FW_para)
#H0 = Normally distibuted | H1 = Not normally distributed
kstest(s_FW[:,0], 'norm',N=1000)
kstest(s_FW[:,1], 'norm',N=1000)
kstest(s_FW[:,2], 'norm',N=1000)
kstest(s_FW[:,3], 'norm',N=1000)
kstest(s_FW[:,4], 'norm',N=1000)
kstest(s_FW[:,5], 'norm',N=1000)
kstest(s_FW[:,6], 'norm',N=1000)
kstest(s_FW[:,7], 'norm',N=1000)
kstest(s_FW[:,8], 'norm',N=1000)
kstest(s_FW[:,9], 'norm',N=1000)
kstest(s_FW[:,10], 'norm',N=1000)
kstest(s_FW[:,11], 'norm',N=1000)
kstest(s_FW[:,12], 'norm',N=1000)
kstest(s_FW[:,13], 'norm',N=1000)
kstest(s_FW[:,14], 'norm',N=1000)
kstest(s_FW[:,15], 'norm',N=1000)
kstest(s_FW[:,16], 'norm',N=1000)



import matplotlib.pyplot as plt

# An "interface" to matplotlib.axes.Axes.hist() method
#n, bins, patches = plt.hist(x=np.log(s[:,0]), bins='auto', color='#0504aa')
n, bins, patches = plt.hist(x=s_FW[:,0], bins='auto', color='#0504aa')
#plt.grid(axis='y', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the first F&W SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FW[:,1], bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the second F&W SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FW[:,2], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the third F&W SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FW[:,3], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the fourth F&W SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FW[:,4], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the fifth F&W SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


###############################################################################
n, bins, patches = plt.hist(x=s_FW[:,5], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the sixth F&W SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


###############################################################################
n, bins, patches = plt.hist(x=s_FW[:,6], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the seventh F&W SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FW[:,7], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the eighth F&W SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FW[:,8], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the ninth F&W SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FW[:,9], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the tenth F&W SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s[:,10], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the eleventh F&W SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FW[:,11], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the twelfth F&W SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FW[:,12], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the thirteenth F&W SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FW[:,13], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the fourteen F&W SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


###############################################################################
n, bins, patches = plt.hist(x=s_FW[:,14], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the fiftieth F&W SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FW[:,15], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the sixteenth F&W SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################
n, bins, patches = plt.hist(x=s_FW[:,16], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the seventeenth F&W SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)



###############################################################################
###############################################################################
###############################################################################
#normallity check of the SumStats for Baseline with standard SumStats
import internal_func_Baseline
import Basic_Model

theta_BSL_Baseline_50000=np.asanyarray(pd.read_csv('Baseline_results.csv'))

theta_1_BSL_Baseline_50000= theta_BSL_Baseline_50000[:,0]


s_para=Parallel(n_jobs=-1)(delayed(internal_func_Baseline.syn_loop)(T,M,theta_BSL_Baseline_50000[49998],funda,y) for _ in range(0,M))
s = np.asarray(s_para)

kstest(s[:,0], 'norm',N=1000)
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
kstest(s[:,20], 'norm',N=1000)
kstest(s[:,21], 'norm',N=1000)

x_Baseline = Basic_Model.Basic_Model(theta_Baseline_0, T, funda, y)

x_Baseline = Basic_Model.Basic_Model(theta_BSL_Baseline_50000[49998], T, funda, y)
print(ks_2samp(x_Baseline,y))


############################################################################
import internal_func_Baseline_alternativeSumStats

theta_BSL_Baseline_alternative_50000=np.asanyarray(pd.read_csv('Baseline_results_alternative_SumStats (1).csv'))
s_para=Parallel(n_jobs=-1)(delayed(internal_func_Baseline_alternativeSumStats.syn_loop)(T,M,theta_BSL_Baseline_alternative_50000[49998],funda,y) for _ in range(0,M))
s = np.asarray(s_para)

kstest(s[:,0], 'norm',N=1000)
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

x_Baseline_alternative = Basic_Model.Basic_Model(theta_BSL_Baseline_alternative_50000[49998], T, funda, y)
print(ks_2samp(x_Baseline_alternative,y))



#################################################################################
##################################################################################
#################################################################################
#F&W_I with SumStats_MK1
import numpy as np
import FrankWesterhoffI
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kstest,ks_2samp, powerlaw, epps_singleton_2samp
M=1000



#FWI_FW_SumStats_25000=np.asanyarray(np.load('F&WI_FW_SumStats_results.npy'))


#Hill does not work, first half ok, second bad
import internal_func_FWI_PropSumStats
FWI_FW_SumStats_25000=np.asanyarray(pd.read_csv('F&WI_propSumStat_results_MK4.csv'))
#import internal_func_FWI_alternativeSumStats
s_para=Parallel(n_jobs=-1)(delayed(internal_func_FWI_PropSumStats.syn_loop)(T,M,FWI_FW_SumStats_25000[9998],funda,y) for _ in range(0,M))
s = np.asarray(s_para)

################################################################################
#works better then FWI_FW_SumStats_results but still room for improvment
FWI_SumStats_MK1_25000=np.asanyarray(pd.read_csv('FWI_SumStats_MK1_results.csv')) 
import internal_func_FWI_SumStats_MK1
#import internal_func_FWI_alternativeSumStats
s_para=Parallel(n_jobs=-1)(delayed(internal_func_FWI_SumStats_MK1.syn_loop)(T,M,FWI_SumStats_MK1_25000[24998],funda,y) for _ in range(0,M))
s = np.asarray(s_para)


x_FWI = FrankWesterhoffI.FrankeWestI(T,FWI_SumStats_MK1_25000[24998],funda,y)
print(ks_2samp(x_FWI,y))
dx_FWI = np.diff(x_FWI)
dy = np.diff(y)
print(ks_2samp(dx_FWI,dy))
#print(ks_2samp(y,y))
#fails
print(epps_singleton_2samp(x_FWI,y))


'acceptance rate of the propsals'
unique, counts = np.unique(FWI_SumStats_MK1_25000, return_counts=True)
dict(zip(unique, counts))
updates = len(unique)/8#len(theta_0)
print(updates)
accaptence_rate = updates/25000
print(accaptence_rate)

theta_1_vector=FWI_SumStats_MK1_25000[:,0]
################################################################################



#works quit well but room for imporvement
FWI_FW_SumStats_25000=np.asanyarray(pd.read_csv('FWI_FW_SumStats_results.csv')) 

x_FWI = FrankWesterhoffI.FrankeWestI(T,FWI_FW_SumStats_25000[24998],funda,y)
#dy = np.dif(y)
# dx_FWI = np.dif(x_FWI)
print(ks_2samp(x_FWI,y))
dy=np.diff(y)
dx_FWI=np.diff(x_FWI)
print(ks_2samp(dy, dx_FWI))
print(ks_2samp(y,y))
#fails
print(epps_singleton_2samp(x_FWI,y))


#normallity check of the SumStats for F&W with standard SumStats
import internal_func_FWI_SumStats_MK1

#import internal_func_FWI_alternativeSumStats
s_para=Parallel(n_jobs=-1)(delayed(internal_func_FWI_SumStats_MK1.syn_loop)(T,M,FWI_FW_SumStats_25000[24998],funda,y) for _ in range(0,M))
s = np.asarray(s_para)

#import internal_func_Baseline_alternativeSumStats
#s_para=Parallel(n_jobs=-1)(delayed(internal_func_Baseline_alternativeSumStats.syn_loop)(T,M,FWI_FW_SumStats_25000[24998],funda,y) for _ in range(0,M))
#s = np.asarray(s_para)


x_FWI = FrankWesterhoffI.FrankeWestI(T,FWI_SumStats_MK1_25000[24998],funda,y)
print(ks_2samp(x_FWI,y))
dx_FWI = np.diff(x_FWI)
dy = np.diff(y)
print(ks_2samp(dx_FWI,dy))
#print(ks_2samp(y,y))
#fails
print(epps_singleton_2samp(x_FWI,y))


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
kstest(s[:,20], 'norm',N=1000)
kstest(s[:,21], 'norm',N=1000)

print(np.corrcoef(s[:,1], s[:,0]))
print(np.corrcoef(s[:,1], s[:,2]))
print(np.corrcoef(s[:,1], s[:,3]))
print(np.corrcoef(s[:,1], s[:,4]))
print(np.corrcoef(s[:,1], s[:,5]))
print(np.corrcoef(s[:,1], s[:,6]))
print(np.corrcoef(s[:,1], s[:,7]))
print(np.corrcoef(s[:,1], s[:,8]))
print(np.corrcoef(s[:,1], s[:,9]))
print(np.corrcoef(s[:,1], s[:,10]))
print(np.corrcoef(s[:,1], s[:,11]))

kstest(s[:,0], 't',args=(1,1),N=1000)
#np.histogram(s[:,0])

import matplotlib.pyplot as plt

# An "interface" to matplotlib.axes.Axes.hist() method
#n, bins, patches = plt.hist(x=np.log(s[:,0]), bins='auto', color='#0504aa')
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

n, bins, patches = plt.hist(x=s[:,20], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the twentie-first SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
#plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
###############################################################################


n, bins, patches = plt.hist(x=s[:,21], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the twentie-second SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)

###############################################################################


n, bins, patches = plt.hist(x=s[:,22], bins='auto', color='#0504aa')
#plt.grid(axis='s 1', alpha=0.75)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the twentie-third SumStat')
#plt.text(23, 45, r'$\mu=15, b=3$')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
###############################################################################

