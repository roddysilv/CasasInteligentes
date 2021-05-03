# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:00:33 2021

@author: Rodrigo
"""

import time
import datetime
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy
from keras.optimizers import Adam
import keras
from matplotlib import pyplot
from keras.callbacks import EarlyStopping
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

start = time.time()

df = pd.read_csv("special/Residential_9_treated.csv")
df.pop('weather')
#t = pd.read_csv("teste.csv")


X = df.drop(['energy_kWh','date'],axis=1)
y = df['energy_kWh']

# X_train, X_test, y_train, y_test = train_test_split(X, y)


train_size = int(df.shape[0]*.7)
dim = X.shape[1]
X_train, X_test = X[0:train_size].values, X[train_size:].values
y_train, y_test = y[0:train_size].values, y[train_size:].values


model = Sequential()
model.add(Dense(64, activation="relu", input_dim=dim))
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))

model.add(Dense(1))


model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-3, decay=1e-3 / 200))

# Patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

# Fit the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10000000, batch_size=100, verbose=2, callbacks=[es])
# history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=] Calculate predictions
PredTestSet = model.predict(X_train)
PredValSet = model.predict(X_test)

# Save predictions
numpy.savetxt("trainresults.csv", PredTestSet, delimiter=",")
numpy.savetxt("valresults.csv", PredValSet, delimiter=",")

# Plot training history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# Plot actual vs prediction for training set
TestResults = numpy.genfromtxt("trainresults.csv", delimiter=",")
plt.plot(y_train,TestResults,'ro')
plt.plot(y_train,y_train,'k')
plt.title('Training Set')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

# Compute R-Square value for training set
TestR2Value = r2_score(y_train,TestResults)
print("Training Set R-Square=", TestR2Value)

print('Tempo decorrido:',str(datetime.timedelta(seconds=(time.time() - start))))


#%%
# PredValSet.sort()
# y_test.sort()

plt.plot(PredValSet[:50],'k--',label='predicted')
plt.plot(y_test[:50],label='test')
plt.title('Prediction 50')
plt.legend()
plt.show()

#%%
plt.plot(PredValSet[:100],'k--',label='predicted')
plt.plot(y_test[:100],label='test')
plt.title('Prediction 100')
plt.legend()
plt.show()

#%%
plt.plot(PredValSet[:500],'k--',label='predicted')
plt.plot(y_test[:500],label='test')
plt.title('Prediction 500')
plt.legend()
plt.show()

