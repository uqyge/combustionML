import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import LSTM
from keras.layers import Dropout

# vector length
nT = 20000
nP = 100
T_min = 50
T_max = 250

# ANN parameters
dim = 1
batch_size = 2
epochs = 100

fluid = 'nitrogen'

# get critical pressure
p_c = CP.PropsSI(fluid,'pcrit')

p_vec = np.linspace(1,3,nP)
p_vec = (p_vec)*p_c

#T_vec = np.zeros((nT,1,1))
T_vec = np.zeros((nT))
T_vec[:] = np.linspace(T_min, T_max, nT)
#T_vec[:, 0, 0] = np.linspace(T_min, T_max, nT)

#rho_vec = np.zeros((nT,1))
rho_vec = np.zeros((nT))

#for i in range(0,len(T_vec)):

print('Generate data ...')
for i in range(0,nT):
    rho_vec[i] = CP.PropsSI('D','T',T_vec[i],'P',1.1*p_c,fluid)

# normalize
T_max = max(T_vec)
rho_max = max(rho_vec)

T_norm = T_vec/T_max
rho_norm = rho_vec/rho_max

####################################
# sinus data
cos = np.zeros((50000, 1, 1))
for i in range(len(cos)):
    cos[i, 0, 0] = 100 * np.cos(2 * np.pi * i / 1000)
    cos[i, 0, 0] = cos[i, 0, 0] * np.exp(-0.0001 * i)

expected_output = np.zeros((len(cos), 1))
for i in range(len(cos)):
    expected_output[i, 0] = np.mean(cos[i + 1:i + 1])
####################################

print('set up ANN')

model1 = Sequential()
model1.add(Dense(20, input_dim=1, activation='relu'))
model1.add(Dense(10, init='uniform',activation='relu'))
model1.add(Dense(1,activation='softmax'))
model1.compile(loss='mse',optimizer='rmsprop', metrics=['accuracy'])

print('fit the ANN')
model1.fit(T_norm,rho_norm,nb_epoch=epochs,batch_size=2000,  validation_split=0.1)

'''
######################
model = Sequential()
# Input layer with dimension 1 and hidden layer i with 128 neurons. 
model.add(Dense(100, input_dim=1, activation='relu'))
# Dropout of 20% of the neurons and activation layer.
model.add(Dropout(.1))
model.add(Activation("linear"))
# Hidden layer j with 64 neurons plus activation layer.
model.add(Dense(64, activation='relu'))
model.add(Activation("linear"))
# Hidden layer k with 64 neurons.
model.add(Dense(64, activation='relu'))
# Output Layer.
model.add(Dense(1))
 
# Model is derived and compiled using mean square error as loss
# function, accuracy as metric and gradient descent optimizer.
model.compile(loss='mse', optimizer='adam', metrics=["accuracy"])


for i in range(epochs):
    print('Epoch', i, '/', epochs)

    # Note that the last state for sample i in a batch will
    # be used as initial state for sample i in the next batch.
    # Thus we are simultaneously training on batch_size series with
    # lower resolution than the original series contained in cos.
    # Each of these series are offset by one step and can be
    # extracted with cos[i::batch_size].

    model.fit(T_vec, p_vec,
              batch_size=batch_size,
              epochs=1,
              verbose=1,
              shuffle=False)
    model.reset_states()
'''

predict = model1.predict(T_norm,batch_size=2000)

plt.plot(predict)
plt.plot(p_vec)
plt.show()



