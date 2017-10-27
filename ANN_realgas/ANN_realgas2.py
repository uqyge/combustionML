import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import LSTM
from keras.layers import Dropout

# vector length
nT = 300
nP = 100
T_min = 110
T_max = 170

# ANN parameters
dim = 1
batch_size = 2
epochs = 100

fluid = 'nitrogen'

# get critical pressure
p_c = CP.PropsSI(fluid, 'pcrit')

p_vec = np.linspace(1, 3, nP)
p_vec = (p_vec) * p_c

#T_vec2 = np.zeros((nT))
T_vec2 = np.linspace(T_min, T_max, nT)

rho_vec2 = np.zeros((nT))

# for i in range(0,len(T_vec)):


print('Generate data ...')
for i in range(0, nT):
    rho_vec2[i] = CP.PropsSI('D', 'T', T_vec2[i], 'P', 1.1*p_c, fluid)

# normalize
T_max = max(T_vec2)
rho_max = max(rho_vec2)

T_norm2 = T_vec2/T_max
rho_norm2 = rho_vec2/rho_max


print('set up ANN')
'''
model1 = Sequential()
model1.add(Dense(20, activation="sigmoid", kernel_initializer="uniform", input_dim=1))
model1.add(Dense(1, activation="linear", kernel_initializer="uniform"))
model1.compile(loss='mse',optimizer='rmsprop', metrics=['accuracy'])

print('fit the ANN')
model1.fit(T_vec2,p_vec,nb_epoch=epochs,batch_size=2,  validation_split=0.1)
'''
######################
model = Sequential()
# Input layer with dimension 1 and hidden layer i with 128 neurons. 
model.add(Dense(128, input_dim=1, activation='relu'))
# Dropout of 10% of the neurons and activation layer.
model.add(Dropout(.1))
model.add(Activation("relu"))
# Hidden layer j with 64 neurons plus activation layer.
model.add(Dense(10, activation='relu'))
model.add(Activation("relu"))
# Hidden layer k with 64 neurons.
#model.add(Dense(64, activation='relu'))
# Output Layer.
model.add(Dense(1,activation='softmax'))

# Model is derived and compiled using mean square error as loss
# function, accuracy as metric and gradient descent optimizer.
model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])

# fit the model
history = model.fit(T_norm2,rho_norm2,epochs=200,batch_size=20,validation_split=0.2,shuffle=True)


predict = model.predict(T_norm2,batch_size=50)

plt.plot(predict)
plt.plot(rho_norm2)
plt.show()



