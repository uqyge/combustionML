from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers import Dropout

from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP

import matplotlib.pyplot as plt

#############################
# input data generation

# vector length
nT = 20000
nP = 100
T_min = 50
T_max = 250

fluid = 'nitrogen'

# get critical pressure
p_c = CP.PropsSI(fluid,'pcrit')

p_vec = np.linspace(1,3,nP)
p_vec = (p_vec)*p_c

T_vec = np.zeros((nT,1,1))
T_vec[:, 0, 0] = np.linspace(T_min, T_max, nT)

rho_vec = np.zeros((nT,1))

print('Generate data ...')
for i in range(0,nT):
    rho_vec[i,0] = CP.PropsSI('D','T',T_vec[i,0,0],'P',1.1*p_c,fluid)

# normalize
T_max = max(T_vec)
rho_max = max(rho_vec)

T_norm = T_vec/T_max
rho_norm = rho_vec/rho_max


#############################
# ANN parameters

batch_size = 2
epochs = 100

in_out_neurons = 1
hidden_neurons = 300

# model

model = Sequential()
model.add(LSTM(hidden_neurons, return_sequences=True,  input_shape=(None, in_out_neurons)))
model.add(LSTM(500, input_dim=300, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(200, input_dim=500, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(in_out_neurons,input_dim=200))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop") #optimizer='adam'


#############################
# plots
plt.plot(predict)
plt.plot(rho_norm)
plt.show()
