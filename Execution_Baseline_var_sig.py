# -*- coding: utf-8 -*-
"""
Created on Tue May  4 16:40:41 2021

@author: svenb
This script runs the code for the Baseline variable sigma model, with the special\
    properties, SumStats-MK1, the standard deviations, covariances, and\
        the given N and m.
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
import internal_func_Baseline_var_sig_log
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
#length of the data, to match the synthetic series
T = len(y)


#setting up the nested loops

#Number of iteration of the MCMC
#the outer loop
N=10
#Number of iteration of the simulation with a theta
#the inner loop
M=50

#set seed
np.random.seed(seed=4)

#The inital values to feed in the MCMC as starting values
#inital values randomly chosen for test reasons
a1 =1
a2=1
a3=0.00000005895
b1=0.207 #1.4543
b2=0.00882 #0.0553677
b3=0.00000311
sig = 22.167
theta_0=(a1,a2,a3,b1,b2,b3,sig)


#the CoVarianz-Matrix needed for the MCMC'
cov_rw = np.zeros(shape=(7,7))
#setting up the hyperparameters Parameter'
std_rw = np.array([1, 1, 1, 1, 1, 1,0.1])

corr_rw = np.array([[0.00322664,8.34222e-05,2.88312e-06,0.0013966,7.60337e-05,4.26027e-06,0.0273076],
                    [8.34222e-05,3.30929e-06,8.41113e-08,4.30088e-05,3.20805e-06,1.53177e-07,0.0009459],
                    [2.88312e-06,8.41113e-08,3.3002e-09,1.68994e-06,7.44328e-08,4.15088e-09,2.61734e-05],
                    [0.0013966,4.30088e-05,1.68994e-06,0.00265985,5.94212e-05,1.67209e-06,0.0137112],
                    [7.60337e-05,3.20805e-06,7.44328e-08,5.94212e-05,4.20631e-06,1.41825e-07,0.000771804],
                    [4.26027e-06,1.53177e-07,4.15088e-09,1.67209e-06,1.41825e-07,8.23986e-09,4.27522e-05],
                    [0.0273076,0.0009459,2.61734e-05,0.0137112,0.000771804,4.27522e-05,0.515633]])


#std_rw = np.array([0.05, 0.001, 0.00005,0.05,0.001,0.00005,0.05])
#corr_rw = np.array([[1,0.5,0.5,0.5,0.5,0.5,0.5],
 #                   [0.5,1,0.5,0.5,0.5,0.5,0.5],
  #                  [0.5,0.5,1,0.5,0.5,0.5,0.5],
   #                 [0.5,0.5,0.5,1,0.5,0.5,0.5],
    #                [0.5,0.5,0.5,0.5,1,0.5,0.5],
     #               [0.5,0.5,0.5,0.5,0.5,1,0.5],
      #              [0.5,0.5,0.5,0.5,0.5,0.5,1]])
      
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
        The function is tuned to be used with the Baseline Model

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
    ssy =  np.asarray(SumStats_MK1_log.sumstats(y, funda, T, M))
    #Calculating the reference likelihood of the original data
    loglike_ind_curr = internal_func_Baseline_var_sig_log.log_like_theta(ssy,theta_curr,funda,y,M,T)
    #Main loop of the algorithmn
    for i in range(0,N):
        #draw the new proposed theta
        theta_prop = np.random.multivariate_normal(theta_curr,cov_rw)
        #check if theta is acceptible (the two variances have to be larger than 0)
        if theta_prop[6]<0: #or theta_prop[5]<-0.01 or theta_prop[4]<-0.5 or theta_prop[2]>0.01 or theta_prop[1]>0.5:
            theta_prop = theta_curr
            
        #the next line does all of the heavy lifting
        #the internal function calculate the likelihood of the given parameter
        #therefor they calculate the hole background of the method
        loglike_ind_prop = internal_func_Baseline_var_sig_log.log_like_theta(ssy,theta_prop,funda,y,M,T)
        #reading out the propsed likelihood
        likelihood_prop[i]= loglike_ind_prop
        
         #check if the new theta should be accepted
        if np.exp(loglike_ind_prop - loglike_ind_curr) > u[i]:
            #if TRUE accept parameters and likelihood 
            theta_curr=theta_prop
            #if TRUE, accept likelihood proposed as likelihood current
            loglike_ind_curr = loglike_ind_prop
   
        #storing the accepted thetas regatless if they are the same as befor.
        theta_BSL[i,:] = theta_curr
        #save liklihoods
        likelihood[i]=loglike_ind_curr
        #retun the matrix of thetas and the last likelihood
    return theta_BSL, loglike_ind_curr

funda = Shiller_ex_ante_price.fundamental(D,T)[0:1800]

#the execution of the function 
theta_BSL,likelihood = bayes_sl_ricker_wood(y,M,N,T,cov_rw,theta_0,funda)
#printing the results
print(theta_BSL,likelihood)
time_stop = time.perf_counter()

runningTime = time_stop-time_start
#time with number of interations of the inner and outer loop
print(runningTime,N,M)

#save the theta_BSL results as a csv file
#np.savetxt('Baseline_constant_var_sig_25000.csv',theta_BSL, delimiter=',', fmt='%s')

#this can be used to directly plot the last calculated theta values
import Baseline_Model_var_sig
#simulate model with given parameters
x = Baseline_Model_var_sig.Basic_Model(theta_BSL[N-1],T,funda,y)
x_i = np.zeros(shape=(10,T))
for i in range(0,10):
    x_i[i] = Baseline_Model_var_sig.Basic_Model(theta_BSL[N-1],T,funda,y)
x_mean =  x_i.sum(axis=0)/10
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

#Print results
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
