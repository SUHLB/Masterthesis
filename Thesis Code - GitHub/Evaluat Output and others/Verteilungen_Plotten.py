# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:23:54 2021

@author: svenb

Plotting of the distributions of the retruns
"""

import numpy as np
from pandas import read_excel
import matplotlib.pyplot as plt
Data_Lux=read_excel(r'C:\Users\svenb\Master\Studium\Material\Masterarbeit\Shiller_data.xls')
#D sind die discontierten Dividenden'
D = np.array(Data_Lux['D'])
y = np.array(Data_Lux['P'])
Date = np.array(Data_Lux['Date'])

dy = np.diff(y)
ly = np.log(y)
dly = np.diff(ly)


return_distribution = dy
fig, ax = plt.subplots()
plt.hist(return_distribution,bins=100,density=True)
plt.title("Return Distribution Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
fig.tight_layout()
plt.show()

###
logged_return_distribution = dly
plt.hist(logged_return_distribution,bins=100,density=True)
plt.title("logged Return Distribution Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()


#####################################
#Plot the Chart of the S&P500
y = Data_Lux["P"]

plt.plot(Date, y, label="Spot Price S&P500")
plt.title("The Spot Price of the S&P500 in Month")
plt.xlabel("Years")
plt.ylabel("Points")
plt.legend()
plt.xticks(rotation=45)

plt.show()

###
#Plot of the fundamentals the Chart of the S&P500
plt.plot(Date[0:1800], funda, label="fundamentel Values S&P500", color ='goldenrod')
plt.title("fundamental Value of the S&P500 in Month")
plt.xlabel("Years")
plt.ylabel("The fundamental Value in Dollar")
plt.legend()
plt.xticks(rotation=45)
plt.show()
#####

#Discrebtive Statistcs of the S&P500 Time series

y_min = np.min(y)
#appears on index 77
print(y_min, Date[77])
y_max = np.max(y)
#appears on index 1800 (the last one)
print(y_max, Date[1800])
y_mean = np.mean(y)
print(y_mean)


#largest and smalles Change
dy_min = np.min(dy)
print(dy_min)#, Date[77])
dy_max = np.max(dy)
print(dy_max)#, Date[77])
dy_mean = np.mean(dy)
print(dy_mean)


#largest and smalles logged Change
dly_min = np.min(dly)
print(dly_min)#, Date[77])
dly_max = np.max(dly)
print(dly_max)#, Date[77])
dly_mean = np.mean(dly)
print(dly_mean)


# The dividents
D_min = np.min(D)
#appears on index 77
print(D_min, Date[77])
D_max = np.max(D)
#appears on index 1800 (the last one)
print(D_max, Date[1800])
D_mean = np.mean(D)
print(D_mean)

import statsmodels as sm
import scipy.stats

#In console geben
#%varexp --plot y

ged_mean,ged_var,ged_kurt = scipy.stats.gennorm.fit(dy)
#scipy.stats.gennorm().pdf(ged_mean)
t_mean, t_var, t_kurt = scipy.stats.t.fit(dy)

"Evaluation and plotting of the Results"

import pandas as pd
import internal_func_FWI
import sumstats_short
import Shiller_ex_ante_price
import FrankWesterhoffI


a0 = 1.985
a_n=-3.498
a_p =0.00
#zeta is change to a'
a = 0.477
b = 0.004
gam = 2.65
epsi_c_sig=39.870
epsi_f_sig=4.749

#the data from Shiller, to find on his Yale website
Data_Shiller=df = pd.read_excel("Shiller_data.xls")
#D are the discounted dividends'
D = np.array(Data_Shiller['D'])
#the original price data from the S&P500
y = np.array(Data_Shiller['P'])
#length of the data, to match the synthetic series
T = len(y)

funda = Shiller_ex_ante_price.fundamental(D,T)[0:1800]
theta_0 = (a0, a, a_n, a_p, b, gam, epsi_c_sig, epsi_f_sig)

'loading the data'
theta_BSL_50000=pd.read_csv('FWI_results.csv')
N=50000
M=50
ssy =  np.asarray(sumstats_short.sumstats(y, funda, T, M))
theta_BSL_FWI_50000 = np.array(theta_BSL_50000)

'acceptance rate of the propsals'
unique, counts = np.unique(theta_BSL_FWI_50000, return_counts=True)
dict(zip(unique, counts))
updates = len(unique)/len(theta_0)
print(updates)
accaptence_rate = updates/N
print(accaptence_rate)

'realisation of the model with the calculated thetas'
print(theta_BSL_FWI_50000[N-2])
x = FrankWesterhoffI.FrankeWestI(T,theta_BSL_FWI_50000[N-2],funda,y)
%varexp --plot x

loglike_ind_theta_0 = internal_func_FWI.log_like_theta(ssy,theta_0,funda,y,M,T)
loglike_ind_prop = internal_func_FWI.log_like_theta(ssy,theta_BSL_FWI_50000[N-2],funda,y,M,T)
print(loglike_ind_prop)
print(loglike_ind_theta_0)

