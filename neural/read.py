# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 14:33:40 2021

@author: Rodrigo
"""

import pandas as pd
import numpy as np

#%%
info = pd.read_csv('../csv/Houses_info.csv')

# =============================================================================
# LER CASAS
# =============================================================================

Weather_YVR = pd.read_csv('../csv/Weather_YVR.csv')
dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in Weather_YVR['date']])
dates['hour']=Weather_YVR['hour']
Weather_YVR['date'] = pd.to_datetime(dates)
Weather_YVR = Weather_YVR.drop(['hour'],axis=1)

Weather_WYj = pd.read_csv('../csv/Weather_WYj.csv')
dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in Weather_WYj['date']])
dates['hour']=Weather_WYj['hour']
Weather_WYj['date'] = pd.to_datetime(dates)
Weather_WYj = Weather_WYj.drop(['hour'],axis=1)

Solar = pd.read_csv('../csv/Solar.csv')
dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in Solar['date']])
Solar = pd.concat([dates.drop(columns =['year']), Solar.drop(columns=['date'])],axis=1)
Solar = pd.concat([Solar.assign(year=np.repeat(i, len(Solar)))for i in range(2012,2021)]).reset_index(drop=True) 
Solar['date'] = pd.to_datetime(Solar[['year','month','day','hour']])
Solar = Solar.drop(['year','month','day','hour'],axis=1)

#%%
for i in range(1,29):
    
    path = info['HouseType'][i-1]   
    
    h = pd.read_csv('../csv/Residential_' + str(i) + '.csv')
    dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in h['date']])
    dates['hour']=h['hour']
    h['date'] = pd.to_datetime(dates)
    h = h.drop(['hour'],axis=1)
    
    if i == 15:
        df = pd.merge(Weather_WYj,Solar, on=['date'])
        df = pd.merge(df,h, on=['date'])
    else:
        df = pd.merge(Weather_YVR,Solar, on=['date'])
        df = pd.merge(df,h, on=['date'])
        
               
    if i != 7:           
        df.to_csv(path + '/Residential_' + str(i) + '.csv',index=False)
    else:
        df.to_csv('Residential_' + str(i) + '.csv',index=False)
        
#%%
for i in range(1,29):
    
    path = info['HouseType'][i-1]   
    
    h = pd.read_csv('../treated_data/Residential_' + str(i) + '.csv')
    dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in h['date']])
    dates['hour']=h['hour']
    h['date'] = pd.to_datetime(dates)
    h = h.drop(['hour'],axis=1)
    
    if i == 15:
        df = pd.merge(Weather_WYj,Solar, on=['date'])
        df = pd.merge(df,h, on=['date'])
    else:
        df = pd.merge(Weather_YVR,Solar, on=['date'])
        df = pd.merge(df,h, on=['date'])
        
               
    if i != 7:           
        df.to_csv(path + '/Residential_' + str(i) + '_treated.csv',index=False)
    else:
        df.to_csv('Residential_' + str(i) + '_treated.csv',index=False)
        
                
#%%
# =============================================================================
# União
# =============================================================================
import os

folders = ['apartment', 'bungalow', 'character', 'duplex', 'laneway', 'modern', 'special']

for path in folders:
    arr = os.listdir('./' + path)
    l=[]
    for residential in arr:
        df = pd.read_csv(path + '/' + residential)
        df = df.rename(columns={'energy_kWh':'energy_' + str(residential)})
        l.append(df)
        print(path +' '+ residential)