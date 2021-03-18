#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 15:49:32 2021

@author: rodrigo
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import r2_score
import pylab as pl
import matplotlib.pyplot as plt
from fbprophet import Prophet
import numpy as np
# import numpy as np


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
# Lê arquivos das casas, tempo e incidencia solar
# Retorna um DataFrame X e uma série y
# =============================================================================
def read_data(fn='./csv/Residential_1.csv'):
# =============================================================================
# Casa 7 Possui dados faltantes sobre a região
# Casa 15 é a unica casa na região WYJ
# Todas as outras casas pertencem a região YVR
# =============================================================================
    #%%
    res = pd.read_csv(fn)
    res = pd.read_csv('./csv/Residential_2.csv')
    
    # res = pd.read_csv('csv\Residential_2.csv')
    
    # holidays = pd.read_csv('csv\Holidays.csv')
    
       
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
    df.index = df['ds']
    
    for s in ['energy_kWh', 'temperature', 'humidity', 'pressure','dc_output','ac_output']:
        df[s]=df[s].rolling(window=6, min_periods=1).mean()
        
    #(df['energy_kWh']).plot()    
    #df['energy_kWh'] = [ np.log(x) if x> 1e-3 else 0 for x in df['energy_kWh'] ]
    #df['energy_kWh'] = np.log(df['energy_kWh']).plot()
    #%%
    df = df.dropna()
    # for c in df.columns:
    #     plt.figure(figsize=(10,5))
    #     df[c].plot(label=c)
    #     plt.legend()
    #     pl.show()
        
    #%%        

    # X = df.drop(columns=['energy_kWh'])

    # y = df.energy_kWh

    # return X, y
    return df

def train():
    
    X, y = read_data()

    train_size = int(X.shape[0]*.8)
    X_train, X_test = X[0:train_size], X[train_size:]
    y_train, y_test = y[0:train_size], y[train_size:]

    reg = RandomForestRegressor()
    reg.fit(X_train, y_train)
    prediction = reg.predict(X_test)
    #%%    
    plt.figure()
    plt.plot(y_test,y_test,'-',y_test,prediction,'o')
    plt.show()
    
    plt.figure()
    plt.plot(y_test.values,'r-',prediction,'b-')
    plt.show()
#%%    



# =============================================================================
# Prophet
# =============================================================================
#%%
df = read_data()

df = df.rename(columns=({'energy_kWh':'y'}))

train_size = int(df.shape[0]*.7)
train, test = df[0:train_size], df[train_size:]
        
m = Prophet()

aux = df.drop(columns=(['y','ds']))
for col in aux.columns: 
    m.add_regressor(col,standardize=False)

m.fit(train)
# m.fit(df)
       
forecast = m.predict(test.drop(columns=(['y'])))
    
fig1 = m.plot(forecast)

fig2 = m.plot_components(forecast)
    
pl.figure()
pl.plot(test.y,test.y,'-',test.y,forecast.yhat,'o')
pl.show()

pl.figure()
pl.plot(test.y.values,'k-',label="Real")
pl.plot(forecast.yhat.values,'b--',label="Previsto")
pl.legend()
pl.show()
#%%
# =============================================================================
# from sklearn.ensemble import  RandomForestRegressor
# reg=RandomForestRegressor()
# reg.fit(aux, df['y'])
# 
# forecast = reg.predict(test.drop(columns=(['y','ds'])))
# 
# pl.figure()
# pl.plot(test.y,test.y,'-',test.y,forecast,'o')
# pl.show()
# 
# pl.figure()
# pl.plot(test.y.values,'k-',label="Real")
# pl.plot(forecast,'b--',label="Previsto")
# pl.legend()
# pl.show()
# =============================================================================

#%%


