# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 21:03:19 2021

@author: Rodrigo
"""
# =============================================================================
# Transforma dados faltantes em energy_kWh para 0 e troca todos os valores 0
# para a média da hora. 
# =============================================================================

import pandas as pd

def media(aux):
    for i in range(aux.shape[0]):
        if(aux.iloc[i,2] == 0):
           if(i + 24 < aux.shape[0] and i - 24 >= 0):
               aux.iloc[i, 2] = (aux.iloc[i + 24, 2] + aux.iloc[i - 24, 2]) / 2
           else:
               aux.iloc[i,2] = (aux.iloc[i - 48, 2] + aux.iloc[i - 24, 2]) / 2

for i in range(1,15):   
    aux = pd.read_csv('csv/Residential_' + str(14 + i) + '.csv')
    aux = aux.fillna(0)
    media(aux)
    # print(aux.describe())
    aux.to_csv('treated_data/Residential_' + str(i + 14) + '.csv')