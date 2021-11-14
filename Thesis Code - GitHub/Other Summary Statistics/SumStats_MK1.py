# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 15:23:05 2021

@author: svenb
"""
#Summary Statitics like in Franke Westerhoff (2012)
import numpy as np
import statsmodels.tsa.stattools as statsm
from scipy.stats import skew,kurtosis

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
    #short_ind = range(T-2) 
    #dlx = np.zeros(len(short_ind))
    
    x = np.nan_to_num(x, copy=True, nan=0.01, posinf=None, neginf=None)
    x = x[0:1800]
    lx = np.log(x)
    
    #this second sourting can be used, the check the list of too small values, those can be a problem later
    #TODO #lx = np.nan_to_num(lx, copy=True, nan=0.0001, posinf=None, neginf=None)
    
    #log of the fundamental values 
    logfunda = np.log(funda)
        
    dx = np.diff(x)
    #log difference oder of one
    #dlx = np.diff(lx)
    #first oder autocorrelation coefficent of the raw returns
    autocorr_dx = np.corrcoef(np.array([dx[:-1], dx[1:]]))
    
    #Hill estimator for the absolut returns 
    #alternative Version https://github.com/ivanvoitalov/tail-estimation
    #ysort = np.sort(dx)   # sort the returns
    #CT = 1740   # set the threshold for ~5%
    #iota = 1/(np.mean(np.log(ysort[0:CT]/ysort[CT]))) # get the tail index
    #print(iota)
    
    #ACF of absolut returns for 60 mounth (bei mir vlt so 12 Monate?)
    #https://www.statsmodels.org/devel/generated/statsmodels.tsa.stattools.acf.html
    acf= statsm.acf(dx, nlags=24)
    #one month
    acf_1= acf[1]
    #three month
    acf_3= acf[3]
    #six month
    acf_6= acf[6]
    #one year
    acf_12= acf[12]
    # two years
    acf_24= acf[24]
   
    #Insgesammt 9 summary statistics #[iota,autocorr_dx,dlx_mean,acf_1,acf_5,acf_10,acf_25,acf_50,acf_100]
    
    #My aditions
    #Absolut Difference of log difference between spot price and fundamental value
    div=np.abs(lx-logfunda)
    #absdiv=np.abs(div)
    #dp_pf_mean = np.mean(div)
    dp_pf_var = np.var(div)
    dp_pf_skew = skew(div)
    dp_pf_kurt = kurtosis(div)
    
    #the autocorrelation of the differences between spot price and fundamental value
    #autocorr_dp_pf = np.corrcoef(np.array([div[:-1], div[1:]]))
    
    #ACF of the absolut Difference of log difference between spot price and fundamental value
    #Also ACF of differences between spot price and fundamental value for 1,3,6,12,24 mounth
    acf_dp_pf= statsm.acf(div, nlags=24)
    #one month
    acf_dp_pf_1= acf_dp_pf[1]
    #three month
    acf_dp_pf_3= acf_dp_pf[3]
    #six month
    acf_dp_pf_6= acf_dp_pf[6]
    #one year
    acf_dp_pf_12= acf_dp_pf[12]
    # two years
    acf_dp_pf_24= acf_dp_pf[24]
    
    
    #evaluate the log differences of oder one
    #The Mean was also included in FW moments
    #dlx_mean = np.mean(dlx)
    #dlx_max = np.amax(dlx)
    #dlx_min = np.amin(dlx)
    dlx_var = np.var(dx)
    dlx_skew = skew(dx)
    dlx_kurt = kurtosis(dx)
    
    #The variance of the raw returns
    #x_var = np.var(x)
    
    #Cubic regession to explain the log diff by the log fundamental values'
    Cubicy = np.poly1d(np.polyfit(lx[0:T-1],logfunda[0:T-1],3))
    
    #autocorr_dp_pf[0,1],\
    #Sumstats 20 long
    sumstat=[autocorr_dx[0,1],acf_1,acf_3,acf_6,acf_12,acf_24,\
             dlx_var,dlx_skew,dlx_kurt,\
                 dp_pf_var,dp_pf_skew,dp_pf_kurt,\
                     acf_dp_pf_1,acf_dp_pf_3,acf_dp_pf_6,acf_dp_pf_12,acf_dp_pf_24,\
                         Cubicy[0],Cubicy[1],Cubicy[2]]
    
    #sumstat=[iota,autocorr_dx[0,1],acf_1,acf_3,acf_6,acf_12,acf_24,\
     #        dlx_mean,dlx_var,dlx_skew,dp_pf_mean,dp_pf_var,dp_pf_skew,Cubicy[0],Cubicy[1],Cubicy[2]]
        
        
    return sumstat  