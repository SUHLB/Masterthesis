# -*- coding: utf-8 -*-
"""
Created on Fri May  7 15:53:35 2021

@author: svenb
"""
import numpy as np
from scipy.stats import gennorm
from statsmodels.tsa import ar_model
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.ar_model.AR', FutureWarning)

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
        
    #log difference oder of one
    dlx = np.diff(lx)
    
    #the AR-Model
    #source of the package 'https://arch.readthedocs.io/en/latest/univariate/introduction.html#arch.univariate.arch_model'
    AR = ar_model.AR(x).fit(3,method='cmle')#,trend='nc',transparams=True)
    #AR parameters (I get 3 summary statistcs out of this)
    AR_res = AR.params
    #AR standart diviation if the parameters (get 3 summary statistcs out of this)
    AR_std = AR.bse
    #AR scale (get 1 summary statistcs out of this)
    AR_scale = AR.scale
    #AR Variance on the laggs, only use the many diagunal (get 3 summary statistcs out of this)
    AR_cov = AR.cov_params()
    
    #log diff between P_f and P'
    div=lx-logfunda
    dp_pf_mean = np.mean(div)
    dp_pf_max = np.amax(div)
    dp_pf_min = np.amin(div)
    
    #evaluate the log differences of oder one
    dlx_max = np.amax(dlx)
    dlx_min = np.amin(dlx)
    dlx_mean = np.mean(dlx)
    
    #Cubic regession to explain the log diff by the log fundamental values'
    Cubicy = np.poly1d(np.polyfit(lx[0:T-1],logfunda[0:T-1],3))
    #Die Koeffizenten werden absteigend vom höhsten Wert ausgegeben'
    
    
    #Fit GED to logged differences'
    ged_mean,ged_var,ged_kurt = gennorm.fit(dlx)

    #The Matrix that includes all the values and parameters that will be used to\
       # claculate the parameters
    sumstat=[AR_res[0],AR_res[1],AR_res[2],AR_std[0],AR_std[1],AR_std[2],\
             AR_scale,AR_cov[0,0],AR_cov[1,1],AR_cov[2,2],Cubicy[0],Cubicy[1],\
                 Cubicy[2], dlx_max, dlx_min,dlx_mean,
                             dp_pf_mean,dp_pf_max,dp_pf_min,\
                                 ged_mean,ged_var,ged_kurt]#,dx_mean,dx_max,dx_min]
    return sumstat  