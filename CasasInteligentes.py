#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 15:49:32 2021

@author: rodrigo
"""

import pandas as pd
# import matplotlib.pyplot as pl
# import numpy as np

h = []

for i in range(28):
    
    h.append(pd.read_csv('Residential_' + str(i+1) + '.csv'))
    

holidays = pd.read_csv('Holidays.csv')

solar = pd.read_csv('Solar.csv')

weather_wyj = pd.read_csv('Weather_WYJ.csv')

weather_yvr = pd.read_csv('Weather_YVR.csv')

info = pd.read_csv('Houses_info.csv')


names = ['FAGF','HP','FPG','FPE','IFRHG','NAC','FAC','PAC','BHE','IFRHE','WRHIR','GEOTH']


info[names] = info[names].astype('bool')


def read_data():
    
    res = pd.read_csv('Residential_1.csv')
    
    # holidays = pd.read_csv('Holidays.csv')
    
    # solar = pd.read_csv('Solar.csv')
    
    weather_yvr = pd.read_csv('Weather_YVR.csv')
    
    # info = pd.read_csv('Houses_info.csv')
    
    
    res['weekday'] = pd.to_datetime(res['date']).dt.dayofweek  # monday = 0, sunday = 6
    res['weekend_indi'] = 0          # Initialize the column with default value of 0
    res.loc[res['weekday'].isin([5, 6]), 'weekend_indi'] = 1  # 5 and 6 correspond to Sat and Sun
    
    weather_yvr.hour=weather_yvr.hour.replace(24,0)
    dummie = pd.get_dummies(weather_yvr['weather'])
    weather_yvr = weather_yvr.drop(columns =['weather'])
    weather_yvr = pd.concat([weather_yvr,dummie],axis=1)
    
    df = pd.merge(weather_yvr,res, on=["date","hour"])
    
    dates = pd.DataFrame([dict(zip(['year','month', 'day'],a.split('-'))) for a in df['date']])
    
    dates['hour']=res['hour']
    
    df.index=pd.to_datetime(dates)
    # df = df.drop(columns =['date','hour'])
    df = df.drop(columns =['date'])
    
    df = df.dropna()

    X = df.drop(columns=['energy_kWh'])

    y = df.energy_kWh

    return X, y