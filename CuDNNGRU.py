# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:28:51 2020

@author: Prabhav Pandey
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('GOOG.csv', date_parser = True)
data.tail()
#%%
data_training = data[data['Date']<'2019-01-01'].copy()
data_test = data[data['Date']>='2019-01-01'].copy()
data_t=data_training

data_training = data_training.drop(['Date', 'Adj Close'], axis = 1)

scaler = MinMaxScaler()
data_training = scaler.fit_transform(data_training)
data_training

data_training[0:10]


X_train = []
y_train = []

for i in range(60, data_training.shape[0]):
    X_train.append(data_training[i-60:i])
    y_train.append(data_training[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)
X_train.shape
#%%
#LSTM


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,  Dropout, GRU ,CuDNNGRU

regressior = Sequential()

regressior.add(CuDNNGRU(units = 60, return_sequences = True, input_shape = (X_train.shape[1], 5)))
regressior.add(Dropout(0.2))

regressior.add(CuDNNGRU(units = 60, return_sequences = True))
regressior.add(Dropout(0.2))

regressior.add(CuDNNGRU(units = 80, return_sequences = True))
regressior.add(Dropout(0.2))

regressior.add(CuDNNGRU(units = 120))
regressior.add(Dropout(0.2))

regressior.add(Dense(units = 1))

regressior.summary()

regressior.compile(optimizer='adam', loss = 'mean_squared_error'
                   , metrics = ['accuracy'])
                   

regressior.fit(X_train, y_train, epochs=50, batch_size=32)


#TEST DATA
#%%

past_60_days = data_t.tail(60)
df = past_60_days.append(data_test, ignore_index = True)
df = df.drop(['Date', 'Adj Close'], axis = 1)
df.head()

inputs = scaler.transform(df)
inputs

X_test = []

y_test = []

for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    y_test.append(inputs[i, 0])
    

X_test, y_test = np.array(X_test), np.array(y_test)
X_test.shape, y_test.shape

y_pred = regressior.predict(X_test)

scaler.scale_

scale = 1/1.33067199e-03
scale

y_pred = y_pred*scale
y_test = y_test*scale


# Visualising the results
plt.figure(figsize=(14,5))
plt.plot(y_test, color = 'red', label = 'Real Google Stock Price')
plt.plot(y_pred, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

