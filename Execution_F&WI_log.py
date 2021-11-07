# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:40:41 2021

@author: svenb
This script runs the code for the Franke & Westerhoff I variable sigma model,\
    with the special properties, SumStats-MK1, the standard deviations,\
        covariances, and the given N and m.
The here used parameters are for a search run. The final run was conducted in the\
    Run_all skript.

"""
#Load libaries and skripts
import numpy as np
from pandas import read_excel
import statsmodels.stats.moment_helpers as SSMH
import Shiller_ex_ante_price
import SumStats_MK1_log
import time
import internal_func_FWI_SumStats_MK1_log
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.ar_model.AR', FutureWarning)

#start the counter to see how long the calculations take
time_start = time.perf_counter()

#the data from Shiller, to find on his Yale website
Data_Shiller=df = read_excel("Shiller_data.xls")
#D are the discounted dividends'
D = np.array(Data_Shiller['D'])
#the original price data from the S&P500
y = np.array(Data_Shiller['P'])
#y = np.load('F&WI_log_synthetic_Data.npy')
#length of the data, to match the synthetic series
T = len(y)


#setting up the nested loops

#Number of iteration of the MCMC
#the outer loop
N=25000
#Number of iteration of the simulation with a theta
#the inner loop
M=50

#set seed 
np.random.seed(seed=8)

#the inital values to feed in the MCMC as starting values
#Lux 2020 , FWI estimated parameters 

a0 = 1.985*2
a_n=--3.498*2
a_p =0.00
#zeta is change to a'
a = 0.477*2
b = 0.004*2
gam = 2.65*2
epsi_c_sig=39.870
epsi_f_sig=4.749

theta_0 = (a0, a, a_n, a_p, b, gam, epsi_c_sig, epsi_f_sig)

#the CoVarianz-Matrix needed for the MCMC'
cov_rw = np.zeros(shape=(8,8))

#setting up the hyperparameters Parameter'
#std_rw = np.array([0.13772871274527865,0.019571819506462324,0.18102109292238044,0.0002646514690123987,0.004275963601713833,0.16638047337077172,0.5586838822744291,0.7480378430144291])#np.array([0.1, 0.01, 0.1,0.0001,0.001,0.1,0.5,0.5])
std_rw = np.array([1, 1, 1,1,1,1,1,1])
#following the paper by Price et al. 2018 for corr_rw 
#corr_rw = np.array([[1,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5],
 #                   [0.5,1,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5],
  #                  [0.5,0.5,1,-0.5,-0.5,-0.5,-0.5,-0.5],
   #                 [0.5,0.5,0.5,1,-0.5,-0.5,-0.5,-0.5],
    ##                [-0.5,-0.5,-0.5,-0.5,1,0.5,0.5,0.5],
      #              [-0.5,-0.5,-0.5,-0.5,-0.5,1,0.5,0.5],
       #             [-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,1,0.5],
        #            [-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,1]])

#etrem schlechte Matrix
#corr_rw = np.array([[1,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05],
 #                   [0.05,1,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05],
  #                  [0.05,0.05,1,-0.05,-0.05,-0.05,-0.05,-0.05],
   #                 [0.05,0.05,0.05,1,-0.05,-0.05,-0.05,-0.05],
    #                [-0.05,-0.05,-0.05,-0.05,1,0.05,0.05,0.05],
     #               [-0.05,-0.05,-0.05,-0.05,-0.05,1,0.05,0.05],
      #              [-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,1,0.05],
       #             [-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,-0.05,1]])

corr_rw = np.array([[0.0189882,-0.000717473,0.00547957,1.05062e-05,-1.14082e-05,0.00842927,-0.0378984,-0.0824975],
                    [-0.000717473,0.00038344,0.00235383,-3.51392e-06,-5.88235e-05,0.00105657,0.00553395,-0.0042477],
                    [0.00547957,0.00235383,0.0328014,-2.03637e-05,-0.000451279,0.0123971,0.0333486,-0.0881162],
                    [1.05062e-05,-3.51392e-06,-2.03637e-05,7.01105e-08,5.75207e-07,-8.58431e-06,-8.71947e-05,2.05033e-05],
                    [-1.14082e-05,-5.88235e-05,-0.000451279,5.75207e-07,1.83022e-05,-0.000415165,-6.59681e-05,0.000992807],
                    [0.00842927,0.00105657,0.0123971,-8.58431e-06,-0.000415165,0.0277102,0.0106744,-0.0767823],
                    [-0.0378984,0.00553395,0.0333486,-8.71947e-05,-6.59681e-05,0.0106744,0.31244,-0.00614337],
                    [-0.0824975,-0.0042477,-0.0881162,2.05033e-05,0.000992807,-0.0767823,-0.00614337,0.560121]])

#convert correlation matrix to covariance matrix given standard deviation'
cov_rw=SSMH.corr2cov(corr_rw,std_rw)

#length of the theta vector
lenTheta = len(theta_0)
#prepering the output matrix of the parameter set (thetas)
theta_BSL = np.zeros(shape=(N,lenTheta))
#...matrix at every drawn parameter set
theta_drawn= np.zeros(shape=(N,lenTheta))
#...likelihood matrix at every iteration
likelihood = np.zeros(N)
#...likelihoods of the propsted parameter sets
likelihood_prop = np.zeros(N)

#number of SumStats 
Num_sums = 20

def bayes_sl_ricker_wood(y,M,N,T,cov_rw,theta_0,funda):
    """
    This function includes the main loop of the BSL. The function is simular to a MCMC.\
        The function is tuned to be used with the Franke&Westerhoff I Model

    Parameters
    ----------
    y : Array of float64
        original data
    M : int
        inner loop length.
    N : int
        outer loop length.
    T : int
        time series length.
    cov_rw : Array of float64 (matrix)
        Covariance matrix of the MCMC.
    theta_0 : tuple
        inital conditions/values.
    funda : Array of float64
        fundamental values.

    Returns
    -------
    theta_BSL : float64
        calculated parameters.
    loglike_ind_curr : float64
        likelihood for the last calculated parameters.

    """
    #inserting the start values
    theta_curr = theta_0
    #drawing u outside the loop should speed up the main loop 
    u = np.random.uniform(0,1,size=N)
    #calculating the summary statistics of the original data (y)
    ssy =  np.asarray(SumStats_MK1_log.sumstats(np.log(y), funda, T, M))
    #Calculating the reference likelihood of the original data
    loglike_ind_curr = internal_func_FWI_SumStats_MK1_log.log_like_theta(ssy,theta_curr,funda,y,M,T)
    #Main loop of the algorithmn'
    for i in range(0,N):
        #draw the new proposed theta
        #np.random.seed(seed=11)
        theta_prop = np.random.multivariate_normal(theta_curr,cov_rw)
        theta_drawn[i] =theta_prop
        #check if theta is acceptible (the two variances have to be larger than 0)
        if theta_prop[6]<0 or theta_prop[7]<0:
            theta_prop = theta_curr
            
        #the next line does all of the heavy lifting
        #the internal function calculate the likelihood of the given parameter
        #therefor they calculate the hole background of the method
        loglike_ind_prop = internal_func_FWI_SumStats_MK1_log.log_like_theta(ssy,theta_prop,funda,y,M,T)
        #reading out the propsed likelihood
        likelihood_prop[i]= loglike_ind_prop
        
         #check if the new theta should be accepted
        if np.exp(loglike_ind_prop - loglike_ind_curr) > u[i]:
            #if TRUE, accept theta proposed as theta current
            theta_curr=theta_prop
            #if TRUE, accept likelihood proposed as likelihood current
            loglike_ind_curr = loglike_ind_prop
   
        #storing the accepted thetas. regatless if there are the same as befor.
        theta_BSL[i,:] = theta_curr
        #save liklihoods
        likelihood[i]=loglike_ind_curr
    return theta_BSL, likelihood, likelihood_prop, theta_drawn

funda = Shiller_ex_ante_price.fundamental(D,T)[0:1800]

#the execution of the function 
theta_BSL,likelihood, likelihood_prop, theta_drawn = bayes_sl_ricker_wood(y,M,N,T,cov_rw,theta_0,funda)
#printing the results
print(theta_BSL,likelihood)
time_stop = time.perf_counter()

runningTime = time_stop-time_start
#time with number of interations of the inner and outer loop
print(runningTime,N,M)

#save the theta_BSL results as a csv file
#np.savetxt('F&WI_log_results.csv',theta_BSL, delimiter=',', fmt='%s')

#this can be used to directly plot the last calculated theta values

"""
a0 = 2
a_n= -2
a_p =0
#zeta is change to a'
a = 0.8
b = 0.01
gam = 0.1
epsi_c_sig=20 #39.870
epsi_f_sig=2 #4.749

theta_0 = (a0, a, a_n, a_p, b, gam, epsi_c_sig, epsi_f_sig)
"""

import FrankWesterhoffI
x_i = np.zeros(shape=(20,T))

for i in range(0,20):
    x_i[i] = FrankWesterhoffI.FrankeWestI(T,theta_BSL[N-1],funda,y)
    #x_i[i] = FrankWesterhoffI.FrankeWestI(T,theta_0,funda,y)
x_mean =  x_i.sum(axis=0)/20
%varexp --plot x_mean

#plot the mean 
%varexp --plot x_mean
 
#acceptance rate of the propsals
unique, counts = np.unique(theta_BSL, return_counts=True)
dict(zip(unique, counts))
updates = len(unique)/len(theta_0)
#print numberof acceptd draws
print(updates)
accaptence_rate = updates/N
#print aceptance rate
print(accaptence_rate)

theta_BSL_0 = theta_BSL[:,0]
%varexp --plot theta_BSL_0
#printing the last theta
print(theta_BSL[N-1])


#instant KS-test of the results and returns
from scipy.stats import ks_2samp
print(ks_2samp(x,y))
dy=np.diff(y)
diff_x = np.diff(x)
print(ks_2samp(diff_x,dy))

#instant KS-test of the average results and returns
print(ks_2samp(x_mean,y))
dy=np.diff(y)
diff_x_mean = np.diff(x_mean)
print(ks_2samp(diff_x_mean,dy))

