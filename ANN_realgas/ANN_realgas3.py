import numpy as np
import CoolProp.CoolProp as CP

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

import matplotlib.animation as animation

# vector length
nT = 20000
nP = 100
T_min = 100
T_max = 160

# ANN parameters
dim = 1
batch_size = 2
epochs = 100

fluid = 'nitrogen'

# get critical pressure
p_c = CP.PropsSI(fluid,'pcrit')

p_vec = np.linspace(1,3,nP)
p_vec = (p_vec)*p_c

T_vec = np.zeros((nT))
T_vec[:] = np.linspace(T_min, T_max, nT)

rho_vec = np.zeros((nT))

print('Generate data ...')
for i in range(0,nT):
    rho_vec[i] = CP.PropsSI('D','T',T_vec[i],'P',1.1*p_c,fluid)

# normalize
T_max = max(T_vec)
rho_max = max(rho_vec)

T_norm = T_vec/T_max
rho_norm = rho_vec/rho_max


print('set up ANN')
######################
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=2, init='uniform'))
model.add(Dense(100, activation='relu', init='uniform'))
model.add(Dense(100, activation='relu', init='uniform'))
model.add(Dense(units=1, activation='linear'))

#sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=False)
#model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# fit the model
history = model.fit(T_norm,rho_norm,
                    epochs=300,
                    batch_size=256,
                    validation_split=0.1,
                    verbose=2,
                    shuffle=True)

predict = model.predict(T_norm)

plt.plot(predict)
plt.plot(rho_norm)
plt.show()

fig = plt.figure()
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.title('mse')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
