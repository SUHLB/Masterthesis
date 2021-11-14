# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 13:14:15 2021

@author: svenb
"""
"""In the following there are some calculations for the hyperparameters of the MCMC
though estimation are not complet, because not all data was saved as a csv-file"""
#Different Starting values

import numpy as np
import pandas as pd
import FrankWesterhoffI
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import os
M=1000
#os.chdir('C:\Users\svenb\Master\Studium\Material\Masterarbeit\ABM-Masterarbeit\BSL_FWI_Review\Ergebnisse')

Prop_FW_std_divided_3_0=pd.read_csv('F&WI_propSumStat_results_std_divided_3_0.csv')
Prop_FW_std_divided_3_0=np.array(Prop_FW_std_divided_3_0)

Prop_FW_std_divided_3_1=pd.read_csv('F&WI_propSumStat_results_std_divided_3_1.csv')
Prop_FW_std_divided_3_1=np.array(Prop_FW_std_divided_3_1)

Prop_FW_std_divided_3_2=pd.read_csv('F&WI_propSumStat_results_std_divided_3_2.csv')
Prop_FW_std_divided_3_2=np.array(Prop_FW_std_divided_3_2)

Prop_FW_std_divided_3_3=pd.read_csv('F&WI_propSumStat_results_std_divided_3_3.csv')
Prop_FW_std_divided_3_3=np.array(Prop_FW_std_divided_3_3)

Prop_FW_std_divided_3_4=pd.read_csv('F&WI_propSumStat_results_std_divided_3_4.csv')
Prop_FW_std_divided_3_4=np.array(Prop_FW_std_divided_3_4)

#Prop_FW_std_divided_3_0 is 495 long

Prop_FW_std_divided_3 = np.vstack((Prop_FW_std_divided_3_0,Prop_FW_std_divided_3_1,Prop_FW_std_divided_3_2,\
                                 Prop_FW_std_divided_3_3,Prop_FW_std_divided_3_4))
    
std_theta_BSL=np.std(Prop_FW_std_divided_3[:,7])
0.01799039751408792, 0.00031812604955285586, 0.019865735942129207, 5.8841505372586675e-08, 1.6131929729243085e-05,\
    0.016988341524195174,0.16822115588705472, 0.6204289409123215

cov_Prop_FW_std_divided_3=np.cov(Prop_FW_std_divided_3.T)



Prop_FW_std_divided_5_0=pd.read_csv('F&WI_propSumStat_results_std_divided_5_0.csv')
Prop_FW_std_divided_5_0=np.array(Prop_FW_std_divided_5_0)

Prop_FW_std_divided_5_1=pd.read_csv('F&WI_propSumStat_results_std_divided_5_1.csv')
Prop_FW_std_divided_5_1=np.array(Prop_FW_std_divided_5_1)

Prop_FW_std_divided_5_2=pd.read_csv('F&WI_propSumStat_results_std_divided_5_2.csv')
Prop_FW_std_divided_5_2=np.array(Prop_FW_std_divided_5_2)

Prop_FW_std_divided_5_3=pd.read_csv('F&WI_propSumStat_results_std_divided_5_3.csv')
Prop_FW_std_divided_5_3=np.array(Prop_FW_std_divided_3_3)

Prop_FW_std_divided_5_4=pd.read_csv('F&WI_propSumStat_results_std_divided_5_4.csv')
Prop_FW_std_divided_5_4=np.array(Prop_FW_std_divided_5_4)

Prop_FW_std_divided_5 = np.vstack((Prop_FW_std_divided_5_0,Prop_FW_std_divided_5_1,Prop_FW_std_divided_5_2,\
                                 Prop_FW_std_divided_5_3,Prop_FW_std_divided_5_4))
    
std_theta_BSL=np.std(Prop_FW_std_divided_5[:,7])
print(std_theta_BSL)

0.008390733305409542,0.0003144274434277491, 0.01127129822465302, 4.4068881176237534e-08, 1.3860546086961832e-05\
    0.00962045140383799,0.1350330911335896, 0.2619735945785377

cov_Prop_FW_std_divided_5=np.cov(Prop_FW_std_divided_5.T)
