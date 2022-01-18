# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 23:59:45 2020

@author: Prabhav Pandey
"""
#importing library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#import dataset
data = pd.read_csv('GOOG.csv', date_parser = True)
data.tail()
#%%
data_training = data

data_t=data_training

data_training = data_training.drop(['Date', 'Adj Close'], axis = 1)

scaler = MinMaxScaler()
data_training = scaler.fit_transform(data_training)
data_training

data_training[0:10]


X_train = []
y_train = []
#X_train Y_train initialisation
for i in range(60, data_training.shape[0]):
    X_train.append(data_training[i-60:i])
    y_train.append(data_training[i, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train.shape)
print(y_train.shape)

#%%
#LSTM

#import keras 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout ,CuDNNLSTM  #CuDNNLSTM GPU based LSTM

regressior = Sequential()
#traing data using LSTM
regressior.add(CuDNNLSTM(units = 60, return_sequences = True, input_shape = (X_train.shape[1], 5)))
regressior.add(Dropout(0.2))

regressior.add(CuDNNLSTM(units = 60, return_sequences = True))
regressior.add(Dropout(0.2))

regressior.add(CuDNNLSTM(units = 80, return_sequences = True))
regressior.add(Dropout(0.2))

regressior.add(CuDNNLSTM(units = 120))
regressior.add(Dropout(0.2))

regressior.add(Dense(units = 1))

regressior.summary()

regressior.compile(optimizer='adam', loss = 'mean_squared_error'
                   ,metrics=['accuracy'])

regressior.fit(X_train, y_train, epochs=50, batch_size=32)

#%%
#futurevalueio prediction
y_test = np.empty(900)
y_test=y_test.reshape(3,60,5)
y_test[0]=data_training[3928]
y_test[1]=data_training[3929]
y_test[2]=data_training[3930]


y_pred = regressior.predict(y_test)

print(scaler.scale_)


print(y_pred)

scale = 1/6.77662810e-04
scale

y_pred = y_pred*scale

print("Pridicted stocks")
print(y_pred)

#%%
#twitter sementic analyser called
from TweetSA import SentimentAnalysis 

sa = SentimentAnalysis()
polarity=sa.DownloadData()

#%%
#Alogorithem to findout stock future profit loss using values from stock prediction and TW sementic analyser
if(y_pred[0]<y_pred[1]<y_pred[2]):
     x=1
     x=(x+polarity)/2*100
    

elif(y_pred[0]<y_pred[1]>y_pred[2]):
     x=0
     x=(x+polarity)/2*100
    

elif(y_pred[0]>y_pred[1]<y_pred[2]):
     x=.50
     x=(x+polarity)/2*100
    
      
elif(y_pred[0]>y_pred[1]>y_pred[2]):
     x=-.50
     x=(x+polarity)/2*100
    
if(x>0):
    print(x,"% of profit")
else:
    print(x*-1,"% of loss")