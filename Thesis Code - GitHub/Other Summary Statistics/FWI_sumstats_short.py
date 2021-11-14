# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 15:23:05 2021

@author: svenb
"""
#Summary Statitics like in Franke Westerhoff (2012)
import numpy as np
import statsmodels.tsa.stattools as statsm


def sumstats(x,funda,T,M):
    """
    The function in that the summary statistics of the simulated time series are\
        calculated.

    Parameters
    ----------
    x : float64
        raw data/time series data.
    funda : TYPE
        fundamental values.
    T : TYPE
        length of the time series.
    M : TYPE
        number of inner loops.

    Returns
    -------
    sumstat : Array of float64
        sumary statistics of the time series.

    """
    short_ind = range(T-2) 
    dlx = np.zeros(len(short_ind))
    
    x = np.nan_to_num(x, copy=True, nan=0.01, posinf=None, neginf=None)
    x = x[0:1800]
    lx = np.log(x)
    
    #this second sourting can be used, the check the list of too small values, those can be a problem later
    #TODO #lx = np.nan_to_num(lx, copy=True, nan=0.0001, posinf=None, neginf=None)
    
    #log of the fundamental values 
    logfunda = np.log(funda)
        
    dx = np.diff(x)
    #log difference oder of one
    dlx = np.diff(lx)
    #first oder autocorrelation coefficent of the raw returns
    autocorr_dx = np.corrcoef(np.array([dx[:-1], dx[1:]]))
    
    #Hill estimator for the absolut returns 
    #alternative Version https://github.com/ivanvoitalov/tail-estimation
    ysort = np.sort(x)   # sort the returns
    CT = 1740   # set the threshold
    iota = 1/(np.mean(np.log(ysort[0:CT]/ysort[CT]))) # get the tail index
    #print(iota)
    
    #ACF of absolut returns for 60 mounth (bei mir vlt so 12 Monate?)
    #https://www.statsmodels.org/devel/generated/statsmodels.tsa.stattools.acf.html
    acf= statsm.acf(dx, nlags=13)
    #one month
    acf_1= acf[1]
    #three month
    acf_3= (acf[2]+acf[3]+acf[4])/3
    #six month
    acf_6= (acf[5]+acf[6]+acf[7])/3
    #one year
    acf_12= (acf[11]+acf[12]+acf[13])/3
    # two years
    #acf_24= acf[24]
    # five years
    #acf_100= acf[60]
    
    #Also ACF of absolut returns for 1,5,10,25,50,100 days but for a 3 day average (bei mir vlt so 12 Monate?)
    
    #mean of the logged returns 
    dlx_mean = np.mean(dlx)
    #variance of the logged returns 
    dlx_var = np.var(dx)
   
    #Insgesammt 9 summary statistics #[iota,autocorr_dx,dlx_mean,acf_1,acf_5,acf_10,acf_25,acf_50,acf_100]
    
    #My aditions
    #absolut Difference of the log difference between spot price and fundamental value
    div=np.abs(lx-logfunda)
    dp_pf_mean = np.mean(div)
    #dp_pf_max = np.amax(div)
    #dp_pf_min = np.amin(div)
    
    
    #first oder autocorrelation coefficent of the raw returns
    autocorr_div = np.corrcoef(np.array([div[:-1], div[1:]]))
    
    #ACF of the absolut Difference of log difference between spot price and fundamental value
    #acf_dp_pf= statsm.acf(div, nlags=24)
    #one month
    #acf_dp_pf_1= acf_dp_pf[1]
    #three month
    #acf_dp_pf_3= (acf_dp_pf[2]+acf_dp_pf[3]+acf_dp_pf[4])/3
    #six month
    #acf_dp_pf_6= (acf_dp_pf[5]+acf_dp_pf[6]+acf_dp_pf[7])/3
    #one year
    #acf_dp_pf_12= (acf_dp_pf[11]+acf_dp_pf[12]+acf_dp_pf[13])/3
    
    #Cubic regession to explain the log diff by the log fundamental values'
    #Cubicy = np.poly1d(np.polyfit(lx[0:T-1],logfunda[0:T-1],3))
    
    #Sumstats 9 long
    sumstat=[iota,acf_1,acf_3,acf_6,acf_12,autocorr_div[0,1],\
             dp_pf_mean,dlx_mean,dlx_var]
    return sumstat  