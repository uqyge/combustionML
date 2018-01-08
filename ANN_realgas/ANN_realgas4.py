import numpy as np
import CoolProp.CoolProp as CP
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.layers import Dropout

import matplotlib.animation as animation

# vector length
nT = 2000

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

rho_vec = np.zeros((nT))    # density
cp_vec = np.zeros((nT))     # cp
mu_vec = np.zeros((nT))     # viscosity
lamb_vec = np.zeros((nT))   # lambda
A_vec = np.zeros((nT))      # speed of sound
Z_vec = np.zeros((nT))      # compressibility

print('Generate data ...')
for i in range(0,nT):
    rho_vec[i] = CP.PropsSI('D','T',T_vec[i],'P',1.1*p_c,fluid)
    cp_vec[i] = CP.PropsSI('C','T',T_vec[i],'P',1.1*p_c,fluid)
    mu_vec[i] = CP.PropsSI('V','T',T_vec[i],'P',1.1*p_c,fluid)
    lamb_vec[i] = CP.PropsSI('L','T',T_vec[i],'P',1.1*p_c,fluid)
    A_vec[i] = CP.PropsSI('A','T',T_vec[i],'P',1.1*p_c,fluid)
    Z_vec[i] = CP.PropsSI('Z','T',T_vec[i],'P',1.1*p_c,fluid)


# normalize

# normalize train data
rhoTP_scaler = preprocessing.MinMaxScaler()
rho_TP_train = rhoTP_scaler.fit_transform(rho_vec.reshape(-1, 1))

cpTP_scaler = preprocessing.MinMaxScaler()
cp_TP_train = cpTP_scaler.fit_transform(cp_vec.reshape(-1, 1))

muTP_scaler = preprocessing.MinMaxScaler()
mu_TP_train = muTP_scaler.fit_transform(mu_vec.reshape(-1, 1))

lambTP_scaler = preprocessing.MinMaxScaler()
lamb_TP_train = lambTP_scaler.fit_transform(lamb_vec.reshape(-1, 1))

ATP_scaler = preprocessing.MinMaxScaler()
A_TP_train = ATP_scaler.fit_transform(A_vec.reshape(-1, 1))

ZTP_scaler = preprocessing.MinMaxScaler()
Z_TP_train = ZTP_scaler.fit_transform(Z_vec.reshape(-1, 1))

T_scaler = preprocessing.MinMaxScaler()
T_train = T_scaler.fit_transform(T_vec.reshape(-1, 1))

# set up the trainings vector
y_train = np.concatenate((rho_TP_train,cp_TP_train,mu_TP_train,lamb_TP_train,A_TP_train,Z_TP_train),axis=1)

print('set up ANN')
######################
model = Sequential()
model.add(Dense(500, activation='relu', input_dim=1, init='uniform'))
model.add(Dense(500, activation='relu', init='uniform'))
model.add(Dense(500, activation='relu', init='uniform'))
model.add(Dense(500, activation='relu', init='uniform'))
model.add(Dense(200, activation='relu', init='uniform'))
model.add(Dense(100, activation='relu', init='uniform'))
model.add(Dense(units=6, activation='linear'))

#sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=False)
#model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# fit the model
history = model.fit(T_train,y_train,
                    epochs=400,
                    batch_size=100,
                    validation_split=0.2,
                    verbose=2,
                    shuffle=True)

T_test = np.zeros(nT)

for i in range(len(T_train)):
    T_test[i] = T_train[i]+np.random.random()

T_test = np.sort(T_test/max(T_test))


predict = model.predict(T_test)

plt.figure(1)
plt.plot(T_test,predict[:,0])
plt.plot(T_train,rho_TP_train)

plt.figure(2)
plt.plot(T_test,predict[:,1])
plt.plot(T_train,cp_TP_train)

plt.figure(3)
plt.plot(T_test,predict[:,2])
plt.plot(T_train,mu_TP_train)

plt.figure(4)
plt.plot(T_test,predict[:,3])
plt.plot(T_train,lamb_TP_train)

plt.figure(5)
plt.plot(T_test,predict[:,4])
plt.plot(T_train,A_TP_train)

plt.figure(6)
plt.plot(T_test,predict[:,5])
plt.plot(T_train,Z_TP_train)

plt.show(block=False)
#



