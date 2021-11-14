# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 17:50:35 2021

@author: svenb
This skript has all the inner functions of the algorithm in it.
"""
import numpy as np
import FrankWesterhoffI_var_sig
import SumStats_MK1_log
from joblib import Parallel, delayed

#the loop that gets the synthetic data and the summary statistics of those
def syn_loop(T,M,theta,funda,y):
    """
    The inner loop of the BSL Method. Simulates data (x) and instantly
        calculates the summary statistics of the simulated data (ssx).

    Parameters
    ----------
    T : int
        length of the time series.
    M : int
        length of the inner loop.
    theta : tuple
        parameters.
    funda : Array of float64
        fundamental values.
    y : Array of float64
        real world data.

    Returns
    -------
    ssx : Array of float64 (matrix)
        Array of summary statistics of all simulated runns.
        
    """
    
    #simulates the data from the model 
    x = FrankWesterhoffI_var_sig.FrankeWestI(T, theta, funda, y)
    #calculates the summary statistics of the data
    ssx = SumStats_MK1_log.sumstats(x, funda, T, M)
    return ssx

#log-likelihood to the given data, from the given theta
def log_prob(ssy, the_mu, the_cov):
    """
    calculates the log likelihood function for the given parameters

    Parameters
    ----------
    ssy : Array of float64
        summary statistics of the real wordl data.
    the_mu : float64
        the mean of the summary statistics.
    the_cov : Array of float64 (matrix)
        covariance martix of the summary statistics.

    Returns
    -------
    return: float64
    log likelihood function of the summary statistics of the given parameters
    
    """
    return  -0.5*np.log(np.linalg.det(the_cov))-0.5*(ssy-the_mu)@np.linalg.pinv(the_cov)@(ssy-the_mu).T
    
#the algorithm that brings the syn_loop and log_prop together
#also it uses parallel computing 
def log_like_theta(ssy,theta,funda,y,M,T):
    """
    simmulates the data, takes the mean and variance of the data and plugs it
    in the log likelihood function.

    Parameters
    ----------
    ssy : Array of float64 (matrix)
        summary statistics of the real world data.
    theta : tuple
        parameters.
    funda : Array of float64
        fundamental value.
    y : Array of float64
        real world data.
    M : int
        length inner loop.
    T : int
        length time series.

    Returns
    -------
    float64
        the log_prob function, or the log likelihood for the given thetas 

    """
    #the parallel computing of the simulations and summary statistics
    s_para=Parallel(n_jobs=61)(delayed(syn_loop)(T,M,theta,funda,y) for _ in range(0,M))
    #changes s_para to a numpy matrix
    s = np.asarray(s_para)
    #the mean and the variance/covariance matrix are calculated
    the_mu = s.sum(axis=0)/M
    the_cov = np.cov(s.T)
    #returns the likelihood of the theta, given the summary statisitcs of the original data
    return log_prob(ssy, the_mu, the_cov)

#this is the same fuction as log_like_theta, but gives out more data.
#It is needed for the effient sample size estimation 
def ESS_theta_func(ssy,theta,funda,y,M,T):
    """
    This function can calculate the efficent sample size for the model.

    Parameters
    ----------
    ssy : Array of float64 (matrix)
        summary statistics of the real world data.
    theta : tuple
        parameters.
    funda : Array of float64
        fundamental values.
    y : Array of float64
        real world data.
    M : int
        inner loop.
    T : int
        length of the time series.

    Returns
    -------
    float64
        log likelihood.
    the_mu : float64
        the mean of the M time simulated summary statistics.

    """
    s_para=Parallel(n_jobs=60)(delayed(syn_loop)(T,M,theta,funda,y) for _ in range(0,M))
    #changes s_para to a numpy matrix
    s = np.asarray(s_para)
    #the mean and the variance/covariance matrix are calculated
    the_mu = s.sum(axis=0)/M
    the_cov = np.cov(s.T) 
    return log_prob(ssy, the_mu, the_cov), the_mu
