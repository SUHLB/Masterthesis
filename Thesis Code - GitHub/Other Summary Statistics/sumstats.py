# -*- coding: utf-8 -*-
"""
Created on Fri May  7 15:53:35 2021

@author: svenb
"""
import numpy as np
#import pandas as pd
from scipy.stats import gennorm
from arch import arch_model
from statsmodels.tsa import ar_model
#import statsmodels.api as smapi
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.ar_model.AR', FutureWarning)

def sumstats(x,funda,T,M):
    #ind = range(T-1) 
    short_ind = range(T-2)
    #ind_set   =range(T)   
    #Number = np.arange(0,T-1,1) 
    dlx = np.zeros(len(short_ind))
    #lx = np.zeros(len(ind))
    dx = np.zeros(len(short_ind))
    #div = np.zeros(len(ind))
    #logging the data'
    # TODO
    x = np.nan_to_num(x, copy=True, nan=0.0001, posinf=None, neginf=None)
    x = x[0:1800]
    lx = np.log(x)
    # TODO why should there ever be log of non-positive number
    lx = np.nan_to_num(lx, copy=True, nan=0.0001, posinf=None, neginf=None)
    logfunda = np.log(funda)
    #series = pd.DataFrame(lx,Number)
    
    #difference oder of one'
    dx = np.diff(x)
    
    #for n in range(1,T-1):
     #   dx[n-1] = x[n]-x[n-1]
        
    #log difference oder of one'
    dlx = np.diff(lx)
    
    #for n in range(1,T-1):
     #   dlx[n-1] = lx[n]-lx[n-1]

    #datay = pd.concat([series.shift(1),series.shift(2),series.shift(3)\
     #                  ,series], axis=1)
    #datay.columns = ['t-1','t-2','t-3','t+1']
    #ARresult = datay.corr()
    #ARre = ARresult['t+1']
    #AR3 = ARresult['t-3']
    #AR2 = ARresult['t-2']
    
    #'https://arch.readthedocs.io/en/latest/univariate/introduction.html#arch.univariate.arch_model'
    AR = ar_model.AR(x).fit(3,method='cmle')#,trend='nc',transparams=True)
    AR_res = AR.params
    AR_std = AR.bse
    AR_scale = AR.scale
    AR_cov = AR.cov_params()
    #distingtions about the log diff'
    #dx_mean = dlx.sum(axis=0)/(T-2)
    #dx_max = np.amax(dlx)
    #dx_min = np.amin(dlx)
    
    #diff between P_f and P'
    #hier kommt der Fehler'
    div=lx-logfunda
    dp_pf_mean = np.mean(div)
    dp_pf_max = np.amax(div)
    dp_pf_min = np.amin(div)
    
    #Cubic regession to explain the log diff by the log fundamental values'
    Cubicy = np.poly1d(np.polyfit(lx[0:T-1],logfunda[0:T-1],3))
    #Die Koeffizenten werden absteigend vom höhsten Wert ausgegeben'
    
    #ARIMA-Model'
    #https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.ArmaProcess.html'
    #ARMA_model=sm.tsa.ARMA(lx,(1,1)).fit(trend='nc',disp=0)
    #ARMA_one,ARMA_two=ARMA_model.params
    
    #GARCH
    #https://arch.readthedocs.io/en/latest/univariate/univariate.html
    GARCH =arch_model(dlx,vol='GARCH',p=1,q=1, dist='StudentsT')
    Garch = GARCH.fit(0,cov_type='robust', show_warning=False)
    #Garch_mu, Garch_omega, Garch_alpha, Garch_beta, Garch_nu= Garch.params
    Garch_mu, Garch_omega, Garch_alpha, Garch_beta, Garch_nu= Garch.params
    #Garch_BIC=Garch.bic
    #Garch_con_vol = Garch.conditional_volatility.sum(axis=0)/(T-2)
    #Garch_nobs = Garch.nobs
    #Garch_param_cov = Garch.param_cov
    Garch_std = Garch.std_err
   
    #Fit GED'
    ged_mean,ged_var,ged_kurt = gennorm.fit(dx)

    #The Matrix that includes all the values and parameters that will be used to\
       # claculate the parameters
    sumstat=[AR_res[0],AR_res[1],AR_res[2],AR_std[0],AR_std[1],AR_std[2],\
             AR_scale,AR_cov[0,0],AR_cov[1,1],AR_cov[2,2],Cubicy[0],Cubicy[1],\
                 Cubicy[2],Garch_mu, Garch_omega,Garch_alpha,Garch_beta,Garch_nu,\
                     #Garch_BIC,#Garch_con_vol,
                     Garch_std[0],Garch_std[1],\
                         Garch_std[2],Garch_std[3],Garch_std[4],\
                             dp_pf_mean,dp_pf_max,dp_pf_min,\
                                 ged_mean,ged_var,ged_kurt]#,dx_mean,dx_max,dx_min]
    return sumstat  