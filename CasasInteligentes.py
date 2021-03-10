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
# import numpy as np


# =============================================================================
# Lê informações das casas
# Transforma algumas info em variáveis tipo dummie
# =============================================================================
def infoCasas():
    info = pd.read_csv('Houses_info.csv')
    
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
def read_data():
# =============================================================================
# Casa 7 Possui dados faltantes sobre a região
# Casa 15 é a unica casa na região WYJ
# Todas as outras casas pertencem a região YVR
# =============================================================================
    res = pd.read_csv('Residential_2.csv')
    
    # res = pd.read_csv('Residential_2.csv')
    
    # holidays = pd.read_csv('Holidays.csv')
           
    dates = pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in res['date']])
    res['year'],res['month'],res['day'] = dates['year'],dates['month'],dates['day']
    # res = res[['date','year','month','day','hour','energy_kWh']]
        
    res['weekday'] = pd.to_datetime(res['date']).dt.dayofweek  # monday = 0, sunday = 6
    res['weekend_indi'] = 0          # Initialize the column with default value of 0
    res.loc[res['weekday'].isin([5, 6]), 'weekend_indi'] = 1  # 5 and 6 correspond to Sat and Sun
      
    weather_yvr = pd.read_csv('Weather_YVR.csv')
    weather_yvr.hour=weather_yvr.hour.replace(24,0)
    
    dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in weather_yvr['date']])
    weather_yvr['year'],weather_yvr['month'],weather_yvr['day'] = dates['year'],dates['month'],dates['day']
    
    dummie = pd.get_dummies(weather_yvr['weather'])
    weather_yvr = weather_yvr.drop(columns =['weather'])
    weather_yvr = pd.concat([weather_yvr,dummie],axis=1)
    
    df = pd.merge(weather_yvr,res, on=["year","month","day","hour"])
    
    solar = pd.read_csv('Solar.csv')
    dates=pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in solar['date']])
    solar = pd.concat([dates.drop(columns =['year']), solar.drop(columns=['date'])],axis=1)
    
    df = pd.merge(df,solar,on=['month','day','hour'],how='left')
    
    df = df.drop(columns=['date_y'])   
    
    dates = pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in df['date_x']])
    
    dates['hour']=res['hour']
    
    df.index=pd.to_datetime(dates)
    df = df.drop(columns =['date_x','hour','year','month','day'])
    # df = df.drop(columns =['date'])
    
    df = df.dropna()

    X = df.drop(columns=['energy_kWh'])

    y = df.energy_kWh

    return X, y

def train():
    
    X, y = read_data()

    train_size = int(X.shape[0]*.8)
    X_train, X_test = X[0:train_size], X[train_size:]
    y_train, y_test = y[0:train_size], y[train_size:]

    reg = RandomForestRegressor()
    reg.fit(X_train, y_train)
    prediction = reg.predict(X_test)
    
    pl.plot(y_test,y_test,'-',y_test,prediction,'o')
    pl.figure()
    pl.plot(y_test.values,'r-',prediction,'b-')

# train()

# i = infoCasas()