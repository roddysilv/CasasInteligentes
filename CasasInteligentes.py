#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 15:49:32 2021

@author: rodrigo
"""

import pandas as pd
import matplotlib.pyplot as plt
# import numpy as np

from sklearn.ensemble import RandomForestRegressor
from fbprophet import Prophet


# =============================================================================
# Lê informações das casas
# Transforma algumas info em variáveis tipo dummie
# =============================================================================
def infoCasas():
    info = pd.read_csv('./csv/Houses_info.csv')
    
    dummie_HouseType = pd.get_dummies(info['HouseType'])
    dummie_Facing = pd.get_dummies(info['Facing'])
    dummie_Region = pd.get_dummies(info['Region'])
    
    info = info.drop(columns =['HouseType','Facing','Region'])
    
    info = pd.concat([info,dummie_HouseType,dummie_Region,dummie_Facing],axis=1)
    
    # names = ['FAGF','HP','FPG','FPE','IFRHG','NAC','FAC','PAC','BHE','IFRHE','WRHIR','GEOTH']

    # info[names] = info[names].astype('bool')
        
    return info

# =============================================================================
# Retorna um Dataframe especifico dos feriados para o fbprophet 
# =============================================================================
def holiday():
    df = pd.read_csv('./csv/Holidays.csv')
    df = df.drop(columns=['day','weekend','dst'])
    df = df.rename(columns=({'date':'ds'}))
    df['ds'] =  pd.to_datetime(df['ds'])
    df = df.dropna()
    return df

# =============================================================================
# Lê arquivos das casas, tempo e incidencia solar
# Retorna um DataFrame X e uma série y
# =============================================================================
def read_data(fn='./csv/Residential_1.csv'):
# =============================================================================
# Casa 7 Possui dados faltantes sobre a região
# Casa 15 é a unica casa na região WYJ
# Todas as outras casas pertencem a região YVR
# =============================================================================
    res = pd.read_csv(fn)
        
    # holidays = pd.read_csv('Holidays.csv')
       
    dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in res['date']])
    res['year'],res['month'],res['day'] = dates['year'],dates['month'],dates['day']
    # res = res[['date','year','month','day','hour','energy_kWh']]
    
    res['weekday'] = pd.to_datetime(res['date']).dt.dayofweek  # monday = 0, sunday = 6
    res['weekend_indi'] = 0          # Initialize the column with default value of 0
    res.loc[res['weekday'].isin([5, 6]), 'weekend_indi'] = 1  # 5 and 6 correspond to Sat and Sun
      
    weather_yvr = pd.read_csv('./csv/Weather_YVR.csv')
    weather_yvr.hour=weather_yvr.hour.replace(24,0)
    
    dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in weather_yvr['date']])
    weather_yvr['year'],weather_yvr['month'],weather_yvr['day'] = dates['year'],dates['month'],dates['day']
    
    dummie = pd.get_dummies(weather_yvr['weather'])
    weather_yvr = weather_yvr.drop(columns =['weather'])
    weather_yvr = pd.concat([weather_yvr,dummie],axis=1)
    
    df = pd.merge(weather_yvr,res, on=["year","month","day","hour"])
    
    solar = pd.read_csv('./csv/Solar.csv')
    dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in solar['date']])
    solar = pd.concat([dates.drop(columns =['year']), solar.drop(columns=['date'])],axis=1)
    
    df = pd.merge(df,solar,on=['month','day','hour'],how='left')
    
    df = df.drop(columns=['date_y'])   
    
    dates = pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in df['date_x']])
    
    dates['hour']=res['hour']
    
    # df.index=pd.to_datetime(dates)
    df['ds'] = pd.to_datetime(dates)
    df = df.drop(columns =['date_x','hour','year','month','day'])
    # df = df.drop(columns =['date'])
    
    df = df.dropna()
    df = df.loc[(df!=0).any(1), (df!=0).any(0)]

# =============================================================================
#     X = df.drop(columns=['energy_kWh'])
#     y = df.energy_kWh
#     return X, y
# =============================================================================

# =============================================================================
#     for c in df.columns:
#         plt.figure(figsize=(10,5))
#         df[c].plot(label=c)
#         plt.legend()
#         plt.show()
# =============================================================================
        
    return df

#%%    
# =============================================================================
# FbProphet
# =============================================================================
df = read_data()
df = df.rename(columns=({'energy_kWh':'y'}))

train_size = int(df.shape[0]*.9)
train, test = df[0:train_size], df[train_size:]

m = Prophet()

for col in train.drop(columns=(['y','ds'])).columns: 
    m.add_regressor(col,standardize='auto')

m.fit(train)
       
forecast = m.predict(test.drop(columns=('y')))
    
fig1 = m.plot(forecast)

fig2 = m.plot_components(forecast)
    
plt.figure()
plt.scatter(test.y,forecast.yhat,c='r',marker='o',edgecolors='k',label='Previsto')
plt.plot(test.y,test.y,'k-',label='Real')
plt.title("Prophet - Scatter")
plt.show()

plt.figure()
plt.plot(test.y.values,'k-',label="Real")
plt.plot(forecast.yhat.values,'b--',label="Previsto")
plt.title("Prophet - Previsão")
plt.legend()
plt.show()

#%%   
# =============================================================================
# RandomForestRegressor
# =============================================================================
df = read_data()

train_size = int(df.shape[0]*.9)
train, test = df[0:train_size], df[train_size:]

reg=RandomForestRegressor()
reg.fit(train.drop(columns=(['energy_kWh','ds'])), train['energy_kWh'])

forecast = reg.predict(test.drop(columns=(['energy_kWh','ds'])))

plt.figure()
plt.scatter(test['energy_kWh'],forecast,c='r',marker='o',edgecolors='k',label='Previsto')
plt.plot(test['energy_kWh'],test['energy_kWh'],'k-',label='Real')
plt.title("RandomForestRegressor - Scatter")
plt.legend()
plt.show()

plt.figure()
plt.plot(test.energy_kWh.values,'k-',label="Real")
plt.plot(forecast,'b--',label="Previsto")
plt.title("RandomForestRegressor - Previsão")
plt.legend()
plt.show()
