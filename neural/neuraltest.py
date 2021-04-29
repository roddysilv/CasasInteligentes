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

start = time.time()

# t = pd.read_csv("bungalow/Residential_1.csv")
t = pd.read_csv("teste.csv")

X = t.drop(['energy_kWh'],axis=1).values
y = t['energy_kWh'].values

# X_train, X_test, y_train, y_test = train_test_split(X, y)


train_size = int(t.shape[0]*.9)
X_train, X_test = X[0:train_size], X[train_size:]
y_train, y_test = y[0:train_size], y[train_size:]


model = Sequential()
model.add(Dense(512, activation="relu"))
model.add(Dense(128))
model.add(Dense(32))
model.add(Dense(8))


model.add(Dense(1))


model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-3, decay=1e-3 / 200))

# Patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

# Fit the model
# history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10000000, batch_size=100, verbose=2, callbacks=[es])
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10000)

# Calculate predictions
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

#%%
# PredValSet.sort()
# y_test.sort()

plt.plot(PredValSet[:50])
plt.plot(y_test[:50])
plt.show()

#%%

plt.plot(PredValSet[:100])
plt.plot(y_test[:100])

print('Tempo decorrido:',str(datetime.timedelta(seconds=(time.time() - start))))
