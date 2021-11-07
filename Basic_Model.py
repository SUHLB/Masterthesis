# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 12:14:34 2021

@author: svenb
"""

import numpy as np


def Basic_Model(theta,N,funda,y):
    """
    The baseline model form Lux 2020

    Parameters
    ----------
    theta : tulpe
        parameters for the model.
    N : int
        length of the time series.
    funda : Array of float64
        fundamental values.
    y : Array of float64
        real world time series.

    Returns
    -------
    p : Array of float64
        simulated time series of points of the S&P500 with the baseline model.

    """
    #shorten the length by one
    N=N-1
    #set up all the variables 
    index_set   =range(N+1)
    ind_set   =range(N)       #die Länge des Vektors
    EDc = np.zeros(len(ind_set)) 
    EDf = np.zeros(len(ind_set))
    ED = np.zeros(len(ind_set))
    p = np.zeros(len(index_set))
    epsilon = np.zeros(len(ind_set))
    #for the inital values, the original once are pluged in
    p[0]=y[0]
    p[1]=y[1]
    beta = 1
    #Fundamental Value
    pf = funda
    #the main function
    for n in range(1,N):
        epsilon = np.random.normal(0,theta[6])
        EDf[n] = theta[0]*(pf[n]-p[n]) + theta[1]*(pf[n]-p[n])**2 + theta[2]*(pf[n]-p[n])**3
        EDc[n] = theta[3]*(p[n]-p[n-1]) + theta[4]*(p[n]-p[n-1])**2 + theta[5]*(p[n]-p[n-1])**3
        ED[n] = EDf[n] + EDc[n] + epsilon
        p[n+1] = p[n] + beta*ED[n]
        #limitations for the price, Without the boundaries the algorithm would crash.
        #The Model ist highly non-linear and instable, therefore the model could blow up.
        if p[n+1]<0:
            p[n+1]=0.5
        if p[n+1]>1000000:
            p[n+1]=p[n] #-np.random.normal(0,theta[6])
    return p

