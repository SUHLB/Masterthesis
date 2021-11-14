# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 19:28:17 2021

@author: svenb
"""


"""Testing the likelihood of the parameters from the Paper by Lux 2021"""
import numpy as np
import pandas as pd
import Shiller_ex_ante_price
from scipy.stats import kstest,ks_2samp, powerlaw, epps_singleton_2samp
import SumStats_MK1_log

Data_Lux=pd.read_excel(r'C:\Users\svenb\Master\Studium\Material\Masterarbeit\Shiller_data.xls')
#D sind die discontierten Dividenden'
D = np.array(Data_Lux['D'])
y = np.array(Data_Lux['P'])
Date = np.array(Data_Lux['Date'])
T =len(y)
M=50
funda = Shiller_ex_ante_price.fundamental(D,T)[0:1800]
ssy =  np.asarray(SumStats_MK1_log.sumstats(y, funda, T, M))

###############################################################################
"Baseline model"
import internal_func_Baseline_log
import Basic_Model

a1 = 0.019
a2=-0.00007163
a3=0.00000005895
a=(a1,a2,a3)
b1=0.207
b2=0.00882
b3=0.00000311
b=(b1,b2,b3)
sig = 22.167
theta_0_Baseline=(a1,a2,a3,b1,b2,b3,sig)

Lux_para_Baseline = internal_func_Baseline_log.log_like_theta(ssy, theta_0_Baseline, funda, y, M, T)

###############################################################################
"Franke Westerhoff model"
import internal_func_FWI_SumStats_MK1_log
import FrankWesterhoffI

a0 = 1.985
a_n=-3.498
a_p =0.00
#zeta is change to a'
a = 0.477
b = 0.004
gam = 2.65
epsi_c_sig=39.870
epsi_f_sig=4.749
theta_0_FW = (a0, a, a_n, a_p, b, gam, epsi_c_sig, epsi_f_sig)

Lux_para_FWI = internal_func_FWI_SumStats_MK1_log.log_like_theta(ssy, theta_0_FW, funda, y, M, T)
