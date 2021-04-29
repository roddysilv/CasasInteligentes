# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 14:33:40 2021

@author: Rodrigo
"""

import pandas as pd
import numpy as np

info = pd.read_csv('../csv/Houses_info.csv')

apartment = info[info['HouseType'] == 'apartment']['House'].values

bungalow = info[info['HouseType'] == 'bungalow']['House'].values

duplex = info[info['HouseType'] == 'duplex']['House'].values

modern = info[info['HouseType'] == 'modern']['House'].values

character = info[info['HouseType'] == 'character']['House'].values

special = info[info['HouseType'] == 'special']['House'].values

laneway = info[info['HouseType'] == 'laneway']['House'].values

# =============================================================================
# LER CASAS
# =============================================================================

Weather = pd.read_csv('../csv/Weather_YVR.csv')
dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in Weather['date']])
dates['hour']=Weather['hour']
Weather['date'] = pd.to_datetime(dates)
Weather = Weather.drop(['hour'],axis=1)

Weather_w = pd.read_csv('../csv/Weather_WYj.csv')
dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in Weather_w['date']])
dates['hour']=Weather_w['hour']
Weather_w['date'] = pd.to_datetime(dates)
Weather_w = Weather_w.drop(['hour'],axis=1)

Solar = pd.read_csv('../csv/Solar.csv')
dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in Solar['date']])
Solar = pd.concat([dates.drop(columns =['year']), Solar.drop(columns=['date'])],axis=1)
Solar = pd.concat([Solar.assign(year=np.repeat(i, len(Solar)))for i in range(2012,2021)]).reset_index(drop=True) 
Solar['date'] = pd.to_datetime(Solar[['year','month','day','hour']])
Solar = Solar.drop(['year','month','day','hour'],axis=1)

for i in apartment:
    
    h = pd.read_csv('../csv/Residential_' + str(i) + '.csv')
    dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in h['date']])
    dates['hour']=h['hour']
    h['date'] = pd.to_datetime(dates)
    h = h.drop(['hour'],axis=1)
    
    df = pd.merge(Weather,Solar, on=['date'])
    df = pd.merge(df,h, on=['date'])
    
    dummie = pd.get_dummies(df['weather'])
    df = df.drop(columns =['weather'])
    df = pd.concat([df,dummie],axis=1)
    
    df.to_csv('apartment/Residential_' + str(i) + '.csv',index=False)
    
for i in bungalow:
    
    h = pd.read_csv('../csv/Residential_' + str(i) + '.csv')
    dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in h['date']])
    dates['hour']=h['hour']
    h['date'] = pd.to_datetime(dates)
    h = h.drop(['hour'],axis=1)
    
    if i == 15:
        df = pd.merge(Weather_w,Solar, on=['date'])
        df = pd.merge(df,h, on=['date'])
    else:
        df = pd.merge(Weather,Solar, on=['date'])
        df = pd.merge(df,h, on=['date'])
        
    dummie = pd.get_dummies(df['weather'])
    df = df.drop(columns =['weather'])
    df = pd.concat([df,dummie],axis=1)
        
    df.to_csv('bungalow/Residential_' + str(i) + '.csv',index=False)
    
for i in duplex:
    
    h = pd.read_csv('../csv/Residential_' + str(i) + '.csv')
    dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in h['date']])
    dates['hour']=h['hour']
    h['date'] = pd.to_datetime(dates)
    h = h.drop(['hour'],axis=1)
    
    df = pd.merge(Weather,Solar, on=['date'])
    df = pd.merge(df,h, on=['date'])
    
    dummie = pd.get_dummies(df['weather'])
    df = df.drop(columns =['weather'])
    df = pd.concat([df,dummie],axis=1)
    
    df.to_csv('duplex/Residential_' + str(i) + '.csv',index=False)
    
for i in modern:
    
    h = pd.read_csv('../csv/Residential_' + str(i) + '.csv')
    dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in h['date']])
    dates['hour']=h['hour']
    h['date'] = pd.to_datetime(dates)
    h = h.drop(['hour'],axis=1)   
    
    df = pd.merge(Weather,Solar, on=['date'])
    df = pd.merge(df,h, on=['date'])
    
    dummie = pd.get_dummies(df['weather'])
    df = df.drop(columns =['weather'])
    df = pd.concat([df,dummie],axis=1)
    
    df.to_csv('modern/Residential_' + str(i) + '.csv',index=False)
    
for i in character:
    
    h = pd.read_csv('../csv/Residential_' + str(i) + '.csv')
    dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in h['date']])
    dates['hour']=h['hour']
    h['date'] = pd.to_datetime(dates)
    h = h.drop(['hour'],axis=1)
    
    df = pd.merge(Weather,Solar, on=['date'])
    df = pd.merge(df,h, on=['date'])
    
    dummie = pd.get_dummies(df['weather'])
    df = df.drop(columns =['weather'])
    df = pd.concat([df,dummie],axis=1)
    
    df.to_csv('character/Residential_' + str(i) + '.csv',index=False)
    
for i in special:
    
    h = pd.read_csv('../csv/Residential_' + str(i) + '.csv')
    dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in h['date']])
    dates['hour']=h['hour']
    h['date'] = pd.to_datetime(dates)
    h = h.drop(['hour'],axis=1)  
    
    df = pd.merge(Weather,Solar, on=['date'])
    df = pd.merge(df,h, on=['date'])
    
    dummie = pd.get_dummies(df['weather'])
    df = df.drop(columns =['weather'])
    df = pd.concat([df,dummie],axis=1)
    
    df.to_csv('special/Residential_' + str(i) + '.csv',index=False)
    
for i in laneway:
    
    h = pd.read_csv('../csv/Residential_' + str(i) + '.csv')
    dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in h['date']])
    dates['hour']=h['hour']
    h['date'] = pd.to_datetime(dates)
    h = h.drop(['hour'],axis=1)
    
    df = pd.merge(Weather,Solar, on=['date'])
    df = pd.merge(df,h, on=['date'])
    
    dummie = pd.get_dummies(df['weather'])
    df = df.drop(columns =['weather'])
    df = pd.concat([df,dummie],axis=1)
    
    df.to_csv('laneway/Residential_' + str(i) + '.csv',index=False)
    