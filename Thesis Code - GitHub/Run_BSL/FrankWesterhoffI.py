# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 09:52:31 2021

@author: svenb
-Franke & Westerhoff Model-
"""

import numpy as np

#the Franke & Westerhoff I Model from the Lux (2020) paper
def FrankeWestI (T,thetas,funda,y):
    """
    The Model from Franke, R., and Westerhoff, F. (2012). Structural stochastic\
        volatility in asset pricing dynamics: Estimation and model contest.\
            Journal of Economic Dynamics and Control, 36(8), 1193-1211

    Parameters
    ----------
    T : int
        length of the time series.
    thetas : tuple
        parameter for the model.
    funda : Array of float64
        fundamental values.
    y : Array of float64
        real world data.

    Returns
    -------
    p_t : Array of float64
        calculated prices of the time series.

    """
    
    ind_set   =range(T)       #length of the time series
    
    #setting up vectors
    ED_c = np.zeros(len(ind_set)) 
    ED_f = np.zeros(len(ind_set))
    p_t = np.zeros(len(ind_set))
    dU = np.zeros(len(ind_set))
    n_c = np.zeros(len(ind_set))
    n_f = np.zeros(len(ind_set))

    #beta = 1 is from the paper
    beta = 1
    #bringing the fundamental price into the equation as reverence
    p_f=funda
    #the inital results to calculate the t-1 are taken from the original data
    p_t[0]= y[0]
    p_t[1]= y[1]
    #inital distribution of fundamentalist and chartist
    dU[0] = 0.5
    #randome seed for the error terms
    for i in range (1,T-1):
        #np.random.seed(seed=111)
        epsi_c = np.random.normal(0,thetas[6])
        #np.random.seed(seed=112)
        epsi_f = np.random.normal(0,thetas[7])
        n_c[i] = 1/np.exp(beta*(dU[i-1]))
        n_f[i] = 1-n_c[i]
        ED_f[i] = (thetas[1]*(p_f[i]-p_t[i])+epsi_f)*n_f[i]
        ED_c[i] = (thetas[4]*(p_t[i]-p_t[i-1])+epsi_c)*n_c[i]
        dU[i] = thetas[0] + thetas[2]*(n_f[i]-n_c[i])+thetas[3]*(p_t[i]-p_f[i])**2
        p_t[i+1] = p_t[i]+beta*(ED_f[i]+ED_c[i])
        #this should provide a littel bit of an cath in case something goes wrong
        #it is very unlike that the S&P500 will drop below zeros, those if prevented that from happining with this if-loop
        if p_t[i+1] < 0:
            p_t[i+1]=0.01
              
        #here the if-loop that prevents the system to explode and produces NaNs. 
        if p_t[i+1] > 10000000:
            p_t[i+1]=p_t[i]
    return p_t 
