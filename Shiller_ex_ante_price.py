# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 10:18:21 2021

@author: svenb
"""
#Shiller ex-post price
import numpy as np


def fundamental(D,T):
    """
    This function calculates the fundamental value, used in the models.
    The fundamental price calculated following Shiller(1981),
    "Do Stock Prices Move Too Much to be Justified by Subsequent Changes in Dividends"
    most of the calculations follow page 425 Table 1 and page 424 equation (2) and equation (3)

    Parameters
    ----------
    D : Array of float64
        discounted devidents from the Shiller data.
    T : int
        lenght of the time series.

    Returns
    -------
    p_f : Array of float64
        The calculated fundamental values.

    """
    #prepering the variables
    ind_set   =range(T)         
    ex_p = np.zeros(len(ind_set)) 
    p_f = np.zeros(len(ind_set))
    #the same lamda like Franke Westerhoff (2012)
    lamda = 1+0.051
    #r is taken from shiller
    r=0.076
    #calculating the discounting factor 
    gamma = 1/(1+r)
    gamma_bar = lamda*gamma
    W=T-1
    
    #ex-ante rational Price / equivalent to equation (3)'
    for t in range(0,W):
        for k in range(0,W-t):
            ex_p[k] = (gamma_bar**(k+1)*D[t+k])
        p_f[t] = ex_p.sum(axis=0)
    return p_f

