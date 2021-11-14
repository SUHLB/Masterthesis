# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 14:39:56 2021

@author: Guest

This skript was used to conduced the final runs of the estimations. Here are all\
    models included, so they can run one after the other over night on the high \
        powered computer of the share. 
The here seen Inputed and calibrations are the once used in the final estimation\
    run for this thesis.
    
The Models are:
    -FWI
    -FWI var sig
    -Baseline 
    -Baseline var sig
"""

###############################################################################
#Model 1
#FWI_SumStats_MK1
###############################################################################

#clear the console and all saved variables
from IPython import get_ipython
get_ipython().magic('reset -sf')

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
#length of the data, to match the synthetic series
T = len(y)


#setting up the nested loops

#Number of iteration of the MCMC
#the outer loop
N=50000

#Number of iteration of the simulation with a theta
#the inner loop
M=50

#setting a seed for the random variable u
np.random.seed(seed=4)

#the inital values to feed in the MCMC as starting values
#Parameter von Lux 2020'

a0 = 1.985*2
a_n=-3.498*2
a_p =0.00
#zeta is change to a'
a = 0.477*2
b = 0.004*2
gam = 2.65*2
epsi_c_sig=5#39.870
epsi_f_sig=5 #4.749

theta_0 = (a0, a, a_n, a_p, b, gam, epsi_c_sig, epsi_f_sig)

#the CoVarianz-Matrix needed for the MCMC'
cov_rw = np.zeros(shape=(8,8))
#setting up the hyperparameters Parameter'
#std_rw = np.array([0.13772871274527865,0.019571819506462324,0.18102109292238044,0.0002646514690123987,0.004275963601713833,0.16638047337077172,0.5586838822744291,0.7480378430144291])#np.array([0.1, 0.01, 0.1,0.0001,0.001,0.1,0.5,0.5])
std_rw = np.array([ 1, 1, 1, 1, 1, 1, 1, 1])
#following the paper by Price et al. 2018 for corr_rw 

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

lenTheta = len(theta_0)
#theta_curr = np.zeros(lenTheta)
theta_BSL = np.zeros(shape=(N,lenTheta))

likelihood = np.zeros(N)

likelihood_prop = np.zeros(N)

theta_drawn= np.zeros(shape=(N,lenTheta))

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
    theta_curr = theta_0
    #drawing u outside the loop should speed up the main loop 
    u = np.random.uniform(0,1,size=N)
    #calculating the summary statistics of the original data (y)
    ssy =  np.asarray(SumStats_MK1_log.sumstats(y, funda, T, M))
    #Calculating the reference likelihood of the original data
    loglike_ind_curr = internal_func_FWI_SumStats_MK1_log.log_like_theta(ssy,theta_curr,funda,y,M,T)
    #Main loop of the algorithmn'
    for i in range(0,N):
        #draw the new proposed theta
        theta_prop = np.random.multivariate_normal(theta_curr,cov_rw)
        theta_drawn[i] =theta_prop
        #check if theta is acceptible (the two variances have to be larger than 0)
        if theta_prop[6]<0 or theta_prop[7]<0:
            theta_prop = theta_curr
            
        #the next line does all of the heavy lifting
        #the internal function calculate the likelihood of the given parameter
        #therefor they calculate the hole background of the method
        loglike_ind_prop = internal_func_FWI_SumStats_MK1_log.log_like_theta(ssy,theta_prop,funda,y,M,T)
        likelihood_prop[i]= loglike_ind_prop
         #check if the new theta should be accepted
        if np.exp(loglike_ind_prop - loglike_ind_curr) > u[i]:
            theta_curr=theta_prop
            loglike_ind_curr = loglike_ind_prop
   
        #storing the accepted thetas, regatless if there are the same as before.
        theta_BSL[i,:] = theta_curr
        #storing the accepted likelihoods, regatless if there are the same as before.
        likelihood[i]= loglike_ind_curr
    return theta_BSL, likelihood,likelihood_prop, theta_drawn

funda = Shiller_ex_ante_price.fundamental(D,T)[0:1800]

#the execution of the function 
theta_BSL_FWI,likelihood, likelihood_prop,theta_drawn = bayes_sl_ricker_wood(y,M,N,T,cov_rw,theta_0,funda)
#printing the results
#print(theta_BSL_FWI,likelihood, likelihood_prop)
time_stop = time.perf_counter()

runningTime = time_stop-time_start
#time with number of interations of the inner and outer loop
print(runningTime,N,M)

#save the theta_BSL results as a csv file
np.savetxt('FWI_results_log_fine.csv',theta_BSL_FWI, delimiter=',', fmt='%s')

np.savetxt('FWI_drawn_log_fine.csv',theta_drawn, delimiter=',', fmt='%s')

np.savetxt('FWI_Likelihood_log_fine.csv',likelihood, delimiter=',', fmt='%s')

np.savetxt('FWI_Likelihood_prop_log_fine.csv',likelihood_prop, delimiter=',', fmt='%s')


###############################################################################
###############################################################################
#Model 2
#var_sig FWI_log
###############################################################################

#clear the console and all saved variables
from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
from pandas import read_excel
import statsmodels.stats.moment_helpers as SSMH
import Shiller_ex_ante_price
import SumStats_MK1_log
import time
import internal_func_FWI_var_sig_log
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
#length of the data, to match by the synthetic series
T = len(y)


#setting up the loops

#Number of iteration of the MCMC
#the outer loop
N=50000
#Number of iteration of the simulation with a theta
#the inner loop
M=50

#setting a seed for the random variable u
np.random.seed(seed=4)

#the inital values to feed in the MCMC as starting values
#Parameter von Lux 2020

a0 = 1.985*2
a_n=-3.498*2
a_p =0.00
#zeta is change to a
a = 0.477*2
b = 0.004*2
gam = 2.65*2
epsi_c_sig=1#1
epsi_f_sig=1#1.1

#combining in a vector
theta_0 = (a0, a, a_n, a_p, b, gam, epsi_c_sig, epsi_f_sig)

#the covarianz matrix of the MCMC
cov_rw = np.zeros(shape=(8,8))
#setting up the hyperparameters Parameter
#following Price et al. 2018 for the standard diviation
std_rw = np.array([ 1, 1, 1, 1, 1, 1, 0.1, 0.1])
#std_rw = np.array([1, 1, 1,1,1,1,1,1])
#following the paper by Price et al. 2018 for corr_rw 
#corr_rw = np.array([[1,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5],
 #                   [0.5,1,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5],
  #                  [0.5,0.5,1,-0.5,-0.5,-0.5,-0.5,-0.5],
   #                 [0.5,0.5,0.5,1,-0.5,-0.5,-0.5,-0.5],
    #                [-0.5,-0.5,-0.5,-0.5,1,0.5,0.5,0.5],
     #               [-0.5,-0.5,-0.5,-0.5,-0.5,1,0.5,0.5],
      #              [-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,1,-0.5],
       #             [-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,1]])

#My estimated covariance matrix
corr_rw = np.array([[0.0189882,-0.000717473,0.00547957,1.05062e-05,-1.14082e-05,0.00842927,-0.0378984,-0.0824975],
                    [-0.000717473,0.00038344,0.00235383,-3.51392e-06,-5.88235e-05,0.00105657,0.00553395,-0.0042477],
                    [0.00547957,0.00235383,0.0328014,-2.03637e-05,-0.000451279,0.0123971,0.0333486,-0.0881162],
                    [1.05062e-05,-3.51392e-06,-2.03637e-05,7.01105e-08,5.75207e-07,-8.58431e-06,-8.71947e-05,2.05033e-05],
                    [-1.14082e-05,-5.88235e-05,-0.000451279,5.75207e-07,1.83022e-05,-0.000415165,-6.59681e-05,0.000992807],
                    [0.00842927,0.00105657,0.0123971,-8.58431e-06,-0.000415165,0.0277102,0.0106744,-0.0767823],
                    [-0.0378984,0.00553395,0.0333486,-8.71947e-05,-6.59681e-05,0.0106744,0.31244,-0.00614337],
                    [-0.0824975,-0.0042477,-0.0881162,2.05033e-05,0.000992807,-0.0767823,-0.00614337,0.560121]])

#convert correlation matrix to covariance matrix given standard deviation
cov_rw=SSMH.corr2cov(corr_rw,std_rw)

#length of the theta vector
lenTheta = len(theta_0)
#prepering the output matrix of the thetas
theta_BSL = np.zeros(shape=(N,lenTheta))
#likelihoods 
likelihood = np.zeros(N)

likelihood_prop = np.zeros(N)

theta_drawn= np.zeros(shape=(N,lenTheta))

Num_sums = 20

def BSL(y,M,N,T,cov_rw,theta_0,funda):
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
    #inserting the start value
    theta_curr = theta_0
    #drawing u outside the loop should speed up the main loop 
    u = np.random.uniform(0,1,size=N)
    #calculating the summary statistics of the original data (y)
    ssy =  np.asarray(SumStats_MK1_log.sumstats(np.log(y), funda, T, M))
    #Calculating the frist likelihood of the original data as reference
    loglike_ind_curr =internal_func_FWI_var_sig_log.log_like_theta(ssy,theta_curr,funda,y,M,T)
    #Main loop of the algorithmn
    for i in range(0,N):
        #draw the new proposed theta
        theta_prop = np.random.multivariate_normal(theta_curr,cov_rw)
        theta_drawn[i] =theta_prop
        #check if theta is acceptible (the two variances have to be larger than 0)
        if theta_prop[6]<0 or theta_prop[7]<0:
            theta_prop = theta_curr
            
        #the next line does all of the heavy lifting
        #the internal function calculate the likelihood of the given parameter
        #therefor they calculate the hole background of the method
        loglike_ind_prop = internal_func_FWI_var_sig_log.log_like_theta(ssy,theta_prop,funda,y,M,T)
        #
        likelihood_prop[i]= loglike_ind_prop
        #check if the new theta should be accepted
        if np.exp(loglike_ind_prop - loglike_ind_curr) > u[i]:
            #if given, accept theta proposed as theta current
            theta_curr=theta_prop
            #if given, accept likelihood proposed as likelihood current
            loglike_ind_curr = loglike_ind_prop
   
        #storing the accepted thetas regatless if they are the same as befor.
        theta_BSL[i,:] = theta_curr
        #save liklihoods
        likelihood[i]=loglike_ind_curr
        #retun the matrix of thetas and the last likelihood
    return theta_BSL, likelihood, likelihood_prop, theta_drawn

#estimating the fundamental values following Shiller
#see Shiller_ex_ante_price
funda = Shiller_ex_ante_price.fundamental(D,T)[0:1800]

#the execution of the function 
theta_BSL_FWI,likelihood,likelihood_prop,theta_drawn = BSL(y,M,N,T,cov_rw,theta_0,funda)

#printing the results#printing the results
#print(theta_BSL,likelihood, likelihood_prop)
time_stop = time.perf_counter()

runningTime = time_stop-time_start
#time with number of interations of the inner and outer loop
print(runningTime,N,M)


#save the theta_BSL results as a csv file
np.savetxt('FWI_results_var_sig_log_fine.csv',theta_BSL_FWI, delimiter=',', fmt='%s')

np.savetxt('FWI_var_sig_drawn_log_fine.csv',theta_drawn, delimiter=',', fmt='%s')

np.savetxt('FWI_var_sig_likelihood_log_fine.csv',likelihood, delimiter=',', fmt='%s')

np.savetxt('FWI_var_sig_likelihood_prop_log_fine.csv',likelihood_prop, delimiter=',', fmt='%s')


###############################################################################
###############################################################################
#Model 3
#var_sig Baseline model
###############################################################################

#clear the console and all saved variables
from IPython import get_ipython
get_ipython().magic('reset -sf')

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
N=50000
#Number of iteration of the simulation with a theta
#the inner loop
M=50

#set seed
np.random.seed(seed=8)

#the inital values to feed in the MCMC as starting values
#Lux 2020, simple linear Model, Model 4
a1 = 0.019*2
a2=-0.00007163*2
a3=0.00000005895*2
b1=0.207*2
b2=0.00882*2
b3=0.00000311*2
sig = 1
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
      
#convert correlation matrix to covariance matrix given standard deviation'
cov_rw=SSMH.corr2cov(corr_rw,std_rw)


lenTheta = len(theta_0)
#theta_curr = np.zeros(lenTheta)
theta_BSL = np.zeros(shape=(N,lenTheta))

likelihood = np.zeros(N)

likelihood_prop = np.zeros(N)

theta_drawn= np.zeros(shape=(N,lenTheta))

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
        theta_drawn[i] =theta_prop
        #check if theta is acceptible (the two variances have to be larger than 0)
        if theta_prop[6]<0: #or theta_prop[5]<-0.01 or theta_prop[4]<-0.5 or theta_prop[2]>0.01 or theta_prop[1]>0.5:
            theta_prop = theta_curr
            
        #the next line does all of the heavy lifting
        #the internal function calculate the likelihood of the given parameter
        #therefor they calculate the hole background of the method
        loglike_ind_prop = internal_func_Baseline_var_sig_log.log_like_theta(ssy,theta_prop,funda,y,M,T)
        
        likelihood_prop[i]= loglike_ind_prop
         #check if the new theta should be accepted
        if np.exp(loglike_ind_prop - loglike_ind_curr) > u[i]:
            theta_curr=theta_prop
            loglike_ind_curr = loglike_ind_prop
   
        #storing the accepted thetas, regatless if there are the same as before.
        theta_BSL[i,:] = theta_curr
        #storing the accepted likelihoods, regatless if there are the same as before.
        likelihood[i]= loglike_ind_curr
    return theta_BSL, likelihood, likelihood_prop, theta_drawn

funda = Shiller_ex_ante_price.fundamental(D,T)[0:1800]

#the execution of the function 
theta_BSL_Baseline,likelihood_Baseline, likelihood_prop_Baseline, theta_drawn_Baseline = bayes_sl_ricker_wood(y,M,N,T,cov_rw,theta_0,funda)
#printing the results
#print(theta_BSL_FWI_alt,likelihood, likelihood_prop)
#stopping the time
time_stop = time.perf_counter()

#estimate the running time
runningTime = time_stop-time_start
#print time with number of interations of the inner and outer loop
print(runningTime,N,M)


#save the theta_BSL results as a csv file
np.savetxt('Baseline_results_var_sig_log_fine.csv',theta_BSL_Baseline, delimiter=',', fmt='%s')

np.savetxt('Baseline_drawn_var_sig_log_fine.csv',theta_drawn_Baseline, delimiter=',', fmt='%s')

np.savetxt('Baseline_var_sig_likelihood_log_fine.csv',likelihood_Baseline, delimiter=',', fmt='%s')

np.savetxt('Baseline_var_sig_likelihood_prop_log_fine.csv',likelihood_prop_Baseline, delimiter=',', fmt='%s')


###############################################################################
###############################################################################
#model 4 
#Baseline model log
###############################################################################

#clear the console and all saved variables
from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
from pandas import read_excel
import statsmodels.stats.moment_helpers as SSMH
import Shiller_ex_ante_price
import SumStats_MK1_log
import time
import internal_func_Baseline_log
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
N=50000
#Number of iteration of the simulation with a theta
#the inner loop
M=50

np.random.seed(seed=8)

#the inital values to feed in the MCMC as starting values
#Lux 2020, simple linear Model, Model 4
a1 = 0.019*2
a2=-0.00007163*2
a3=0.00000005895*2
b1=0.207*2
b2=0.00882*2
b3=0.00000311*2
sig = 22.167*2
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

#convert correlation matrix to covariance matrix given standard deviation'
cov_rw=SSMH.corr2cov(corr_rw,std_rw)

lenTheta = len(theta_0)
#theta_curr = np.zeros(lenTheta)
theta_BSL = np.zeros(shape=(N,lenTheta))
#likelihoods 
likelihood = np.zeros(N)

likelihood_prop = np.zeros(N)

theta_drawn= np.zeros(shape=(N,lenTheta))

#Number von Sumstats 
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
    theta_curr = theta_0
    #drawing u outside the loop should speed up the main loop 
    u = np.random.uniform(0,1,size=N)
    #calculating the summary statistics of the original data (y)
    ssy =  np.asarray(SumStats_MK1_log.sumstats(y, funda, T, M))
    #Calculating the reference likelihood of the original data
    loglike_ind_curr = internal_func_Baseline_log.log_like_theta(ssy,theta_curr,funda,y,M,T)
    #Main loop of the algorithmn
    for i in range(0,N):
        #draw the new proposed theta
        theta_prop = np.random.multivariate_normal(theta_curr,cov_rw)
        theta_drawn[i] =theta_prop
        #check if theta is acceptible (the two variances have to be larger than 0)
        if theta_prop[6]<0: #or theta_prop[5]<-0.01 or theta_prop[4]<-0.5 or theta_prop[2]>0.01 or theta_prop[1]>0.5:
            theta_prop = theta_curr
            
        #the next line does all of the heavy lifting
        #the internal function calculate the likelihood of the given parameter
        #therefor they calculate the hole background of the method
        loglike_ind_prop = internal_func_Baseline_log.log_like_theta(ssy,theta_prop,funda,y,M,T)
        likelihood_prop[i]= loglike_ind_prop
         #check if the new theta should be accepted
        if np.exp(loglike_ind_prop - loglike_ind_curr) > u[i]:
            theta_curr=theta_prop
            loglike_ind_curr = loglike_ind_prop
   
        #storing the accepted thetas, regatless if there are the same as before.
        theta_BSL[i,:] = theta_curr
        #storing the accepted likelihoods, regatless if there are the same as before.
        likelihood[i]= loglike_ind_curr
    return theta_BSL, likelihood, likelihood_prop, theta_drawn

funda = Shiller_ex_ante_price.fundamental(D,T)[0:1800]

#the execution of the function 
theta_BSL_Baseline_var,likelihood_Baseline_var,likelihood_prop_Baseline_var, theta_drawn_Baseline_var = bayes_sl_ricker_wood(y,M,N,T,cov_rw,theta_0,funda)
#printing the results
print(theta_BSL,likelihood)
time_stop = time.perf_counter()

runningTime = time_stop-time_start
#time with number of interations of the inner and outer loop
print(runningTime,N,M)

#save the theta_BSL results as a csv file
np.savetxt('Baseline_results_log_fine.csv',theta_BSL_Baseline_var, delimiter=',', fmt='%s')

np.savetxt('Baseline_drawn_log_fine.csv', theta_drawn_Baseline_var, delimiter=',', fmt='%s')

np.savetxt('Baseline_Likelihood_log_fine.csv',likelihood_Baseline_var, delimiter=',', fmt='%s')

np.savetxt('Baseline_Likelihood_prop_log_fine.csv',likelihood_prop_Baseline_var, delimiter=',', fmt='%s')
